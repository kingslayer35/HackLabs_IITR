"""Semantic case retrieval audit, then a cached-control matched selector pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import OpenRouterClient
from .common import ARTIFACTS, SYSTEMS, read_json
from .comparisons import ReservedBudget
from .context_memory import MAX_EXAMPLES, MAX_SYSTEM_BYTES
from .memory_eval import body as lexical_body, verify as verify_lexical
from .selector500_eval import MODEL, PROVIDER, FIELDS, validate, sha, write, cluster_interval
from .semantic_memory import SemanticMemory

EXPERIMENT=ARTIFACTS/'semantic_memory_v1'
MEMORY=ARTIFACTS/'memory_context_v1'
DATA=ARTIFACTS/'diverse500_v1'
MODES=('lexical','semantic')
CAP=1.0
WORKERS=4


def body(prepared,mode):
    if mode not in MODES:raise ValueError('Unknown semantic-memory condition')
    proxy=dict(base=prepared['base'],examples={'context':prepared['examples'][mode]})
    return lexical_body(proxy,'context')


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def cached_response(iid,request_sha):
    # Fixed priority is independent of correctness. In particular, unchanged semantic
    # evidence shares its exact lexical-control response instead of sampling again.
    for mode in ['context','mention','contrast']:
        path=MEMORY/'responses'/mode/(iid+'.json')
        if path.exists():
            saved=read_json(path)
            if saved['request_sha256']==request_sha:return path,saved
    return None,None


def prepare():
    if (EXPERIMENT/'prepared.json').exists():
        for path,h in read_json(EXPERIMENT/'prepared.json')['hashes'].items():assert sha(Path(path))==h,path
        return
    verify_lexical()
    embedding=read_json(EXPERIMENT/'embeddings.json')
    for path,h in embedding['hashes'].items():assert sha(Path(path))==h,path
    plan=read_json(EXPERIMENT/'plan.json');ids=read_json(MEMORY/'prepared.json')['input_ids']
    memory=SemanticMemory();stats={s:Counter() for s in SYSTEMS};blind=[]
    for n,iid in enumerate(ids,1):
        old=read_json(MEMORY/'evidence'/(iid+'.json'));examples={'lexical':old['examples']['context'],'semantic':{}};diagnostics={}
        for s in SYSTEMS:
            codes={c['code'] for c in old['base']['payload']['candidates'][s]}
            selected=memory.retrieve(iid,s,codes);examples['semantic'][s]=selected['examples'];diagnostics[s]={k:v for k,v in selected.items() if k!='examples'}
            a,b=examples['lexical'][s],examples['semantic'][s]
            t=stats[s];t['inputs']+=1;t['different_order']+=a!=b
            t['different_set']+={e['example_id'] for e in a}!={e['example_id'] for e in b}
            t['semantic_examples']+=len(b);t['lexical_examples']+=len(a)
            assert len(b)<=MAX_EXAMPLES and len(json.dumps(b,ensure_ascii=False,separators=(',',':')).encode())<=MAX_SYSTEM_BYTES
            assert all(e['code'] in codes for e in b)
        prepared=dict(input_id=iid,base=old['base'],examples=examples,diagnostics=diagnostics)
        write(EXPERIMENT/'evidence'/(iid+'.json'),prepared)
        if iid in plan['audit_input_ids']:
            for s in SYSTEMS:
                # Hide retrieval-system names and present unique top examples in hash order.
                alternatives={e['example_id']:e for m in MODES for e in examples[m][s][:1]}
                options=sorted(alternatives.values(),key=lambda e:hashlib.sha256(('blind:'+iid+':'+e['example_id']).encode()).hexdigest())
                blind.append(dict(input_id=iid,system=s,mention=old['base']['payload']['mention'],
                    context=old['base']['payload']['context'],options=options))
        if n%100==0:print(f'Semantic retrieval prepared {n}/500',flush=True)
    memory.close()
    pilot_stats={s:Counter() for s in SYSTEMS}
    for iid in plan['pilot_input_ids']:
        e=read_json(EXPERIMENT/'evidence'/(iid+'.json'))
        for s in SYSTEMS:
            a,b=e['examples']['lexical'][s],e['examples']['semantic'][s];t=pilot_stats[s]
            t['inputs']+=1;t['different_order']+=a!=b;t['different_set']+={x['example_id'] for x in a}!={x['example_id'] for x in b}
            t['lexical_examples']+=len(a);t['semantic_examples']+=len(b)
    write(EXPERIMENT/'retrieval_audit.json',dict(training_rows=3423,offline_inputs=500,
        all500={s:dict(v) for s,v in stats.items()},pilot={s:dict(v) for s,v in pilot_stats.items()},
        numerical_checks='All embeddings finite and unit normalized; exact masked case retrieval; shared display/budget controls. No labels or predictions loaded for preparation.',
        model=embedding['model'],revision=embedding['revision']))
    write(EXPERIMENT/'blind_relevance_review.json',blind)
    protected=[EXPERIMENT/'plan.json',EXPERIMENT/'embeddings.json',EXPERIMENT/'retrieval_audit.json',EXPERIMENT/'blind_relevance_review.json']
    protected+=list((EXPERIMENT/'evidence').glob('*.json'))
    protected+=list((MEMORY/'evidence').glob('*.json'))
    protected+=list((MEMORY/'responses').glob('*/*.json'))
    write(EXPERIMENT/'prepared.json',dict(hashes={**embedding['hashes'],**{str(p):sha(p) for p in protected}}))


def audit_gate():
    plan=read_json(EXPERIMENT/'plan.json');audit=read_json(EXPERIMENT/'retrieval_audit.json')
    reviews=read_json(EXPERIMENT/'relevance_ratings.json');items=read_json(EXPERIMENT/'blind_relevance_review.json')
    assert len(items)==24
    summary=[];total=0;improved=0;worse=0
    for item in items:
        key=item['input_id']+':'+item['system'];scores=reviews['ratings'][key]
        assert set(scores)=={e['example_id'] for e in item['options']}
        assert all(type(v['score']) is int and 0<=v['score']<=2 and v['note'] for v in scores.values())
        e=read_json(EXPERIMENT/'evidence'/(item['input_id']+'.json'));a=e['examples']['lexical'][item['system']];b=e['examples']['semantic'][item['system']]
        old=scores[a[0]['example_id']]['score'] if a else 0;new=scores[b[0]['example_id']]['score'] if b else 0
        total+=new-old;improved+=new>old;worse+=new<old
        summary.append(dict(key=key,lexical_score=old,semantic_score=new))
    changed=any(audit['pilot'][s]['different_set']>=30 for s in SYSTEMS)
    passed=changed and total>=4 and improved>worse
    result=dict(passed=passed,changed_sets_gate=changed,relevance_total_delta=total,improved=improved,worsened=worse,
        comparisons=summary,limitation='Small qualitative concept-relevance audit by the coding assistant, not blinded clinical adjudication or gold-code accuracy. Retrieval names hidden during rating; context/evidence and some cases were already known.')
    write(EXPERIMENT/'audit_gate.json',result);print(json.dumps(result,indent=2))


def freeze():
    if (EXPERIMENT/'config.json').exists():verify();return
    assert read_json(EXPERIMENT/'audit_gate.json')['passed'],'Offline relevance gate did not pass'
    prepared=read_json(EXPERIMENT/'prepared.json')
    for path,h in prepared['hashes'].items():assert sha(Path(path))==h,path
    endpoint=next(e for e in read_json(EXPERIMENT/'provider_prices.json')['data']['endpoints'] if e['provider_name']==PROVIDER)
    assert float(endpoint['pricing']['prompt'])<=.00000013 and float(endpoint['pricing']['completion'])<=.00000028
    paths=list(Path(__file__).parent.glob('*.py'))+[EXPERIMENT/'prepared.json',EXPERIMENT/'provider_prices.json',EXPERIMENT/'relevance_ratings.json',EXPERIMENT/'audit_gate.json',DATA/'labels.json']
    write(EXPERIMENT/'config.json',dict(n=100,modes=MODES,model=MODEL,provider=PROVIDER,reasoning=False,max_tokens=2048,
        seed=13,temperature=0,cap_usd=CAP,workers=WORKERS,plan=read_json(EXPERIMENT/'plan.json'),
        scope='Matched semantic versus lexical-context case retrieval, fixed candidates/guide/display. Canonical cached controls and exact-request response sharing; only changed semantic requests call API.',
        hashes={**prepared['hashes'],**{str(p):sha(p) for p in paths}}))
    verify();print('FROZEN: audited semantic memory, cached lexical control, exact-request deduplication, 100 inputs, $1 new-usage cap',flush=True)


def verify():
    for path,h in read_json(EXPERIMENT/'config.json')['hashes'].items():assert sha(Path(path))==h,path


def run_one(iid,mode,budget):
    prepared=read_json(EXPERIMENT/'evidence'/(iid+'.json'));request=body(prepared,mode);key=fingerprint(request)
    cache_path,saved=cached_response(iid,key);origin='cached' if saved else 'fresh'
    paid_path=EXPERIMENT/'responses'/mode/(iid+'.json')
    if saved is None and paid_path.exists():
        saved=read_json(paid_path);assert saved['request_sha256']==key
    if saved is None:
        client=OpenRouterClient(MODEL,PROVIDER,budget)
        upper=(len(json.dumps(request,ensure_ascii=False).encode())+10000)*.00000013+2048*.00000028
        failure_path=EXPERIMENT/'transport_failures'/mode/(iid+'.json');failures=read_json(failure_path) if failure_path.exists() else []
        if len(failures)>=2:raise RuntimeError('Two transport failures; diagnose before another request')
        budget.reserve(upper);started=time.monotonic()
        try:
            r=requests.post(client.url,headers={'Authorization':'Bearer '+client.key,'Content-Type':'application/json'},json=request,timeout=(15,120))
            if r.status_code!=200:raise RuntimeError('HTTP '+str(r.status_code)+': '+r.text.replace(client.key,'[REDACTED]')[:500])
            raw=r.json()
            if raw.get('error') or not raw.get('id'):raise RuntimeError('Provider error: '+str(raw.get('error')).replace(client.key,'[REDACTED]')[:500])
            saved=dict(input_id=iid,mode=mode,request_sha256=key,data=raw,latency_seconds=time.monotonic()-started);write(paid_path,saved)
            cost=raw.get('usage',{}).get('cost');budget.add(float(cost or 0));budget.release(upper,upper if cost is None else 0)
            if cost is None:raise RuntimeError('Missing usage cost; preserve paid response')
        except Exception as exc:
            if not paid_path.exists():
                budget.release(upper,upper);failures.append(dict(error=str(exc).replace(client.key,'[REDACTED]')[:600],uncertain_cost_upper=upper));write(failure_path,failures)
            raise
    raw=saved['data'];assert raw['model']==MODEL and raw['provider']==PROVIDER and raw['usage'].get('cost') is not None
    choice=raw['choices'][0]
    try:prediction=json.loads(choice['message'].get('content') or '');error=validate(prediction,prepared['base'])
    except (ValueError,TypeError):error='Invalid JSON output'
    if error:prediction=dict(icd10cm_code=prepared['base']['fallback']['ICD10CM'],snomed_code=prepared['base']['fallback']['SNOMEDCT_US'],confidence=0.,evidence='Visible-code fallback after invalid output.')
    result=dict(input_id=iid,mode=mode,prediction=prediction,status='fallback_invalid' if error else 'model',error=error,
        generation_id=raw['id'],provider=raw['provider'],model=raw['model'],usage=raw['usage'],finish_reason=choice.get('finish_reason'),
        origin=origin,cache_path=str(cache_path) if cache_path else None,request_sha256=key,latency_seconds=saved['latency_seconds'],new_cost_usd=0 if origin=='cached' else float(raw['usage']['cost']))
    write(EXPERIMENT/'runs'/mode/(iid+'.json'),result);return result


def run():
    verify();assert read_json(EXPERIMENT/'audit_gate.json')['passed'];budget=ReservedBudget(CAP)
    for path in (EXPERIMENT/'responses').glob('*/*.json'):
        value=read_json(path)['data']['usage'].get('cost')
        if value is None:raise RuntimeError('Missing prior cost')
        budget.add(float(value))
    for path in (EXPERIMENT/'transport_failures').glob('*/*.json'):budget.uncertain_cost_bound+=sum(x['uncertain_cost_upper'] for x in read_json(path))
    ids=read_json(EXPERIMENT/'plan.json')['pilot_input_ids'];tasks=[(iid,m) for iid in ids for m in MODES if not (EXPERIMENT/'runs'/m/(iid+'.json')).exists()];errors=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs={pool.submit(run_one,iid,m,budget):(iid,m) for iid,m in tasks}
        for n,future in enumerate(as_completed(jobs),1):
            try:future.result()
            except Exception as exc:
                error=dict(input_id=jobs[future][0],mode=jobs[future][1],error=str(exc)[:800]);errors.append(error);print('REQUEST ERROR '+json.dumps(error),flush=True)
            if n%25==0 or n==len(jobs):print(f'Completed {n}/{len(jobs)}; new cost ${budget.cost:.6f}; errors {len(errors)}',flush=True)
    write(EXPERIMENT/'execution.json',dict(errors=errors,new_cost_usd=budget.cost,uncertain_cost_bound=budget.uncertain_cost_bound))
    if errors:raise RuntimeError('Unresolved infrastructure failures; retain successful requests and diagnose')


def analyze():
    verify();ids=read_json(EXPERIMENT/'plan.json')['pilot_input_ids'];lookup={x['input_id']:x for x in read_json(DATA/'labels.json')};labels=[lookup[iid] for iid in ids]
    vectors={};summary={};all_records={}
    for mode in MODES:
        rr=[read_json(EXPERIMENT/'runs'/mode/(iid+'.json')) for iid in ids];all_records[mode]=rr
        vectors[mode]={s:[r['prediction'][f]==x['gold'][s] for r,x in zip(rr,labels)] for s,f in FIELDS.items()};systems={}
        for s in SYSTEMS:
            systems[s]={}
            for bucket in ['all','unseen','rare','common']:
                ii=[i for i,x in enumerate(labels) if bucket=='all' or x['bucket'][s]==bucket]
                systems[s][bucket]=dict(n=len(ii),correct=sum(vectors[mode][s][i] for i in ii),accuracy=sum(vectors[mode][s][i] for i in ii)/len(ii))
        summary[mode]=dict(per_system=systems,headline=min(systems[s]['all']['accuracy'] for s in SYSTEMS),both_correct=sum(all(vectors[mode][s][i] for s in SYSTEMS) for i in range(100)),
            statuses=dict(Counter(r['status'] for r in rr)),origins=dict(Counter(r['origin'] for r in rr)),new_cost_usd=sum(r['new_cost_usd'] for r in rr),
            historical_or_fresh_response_cost=sum(float(r['usage']['cost']) for r in rr),latency_median=statistics.median(r['latency_seconds'] for r in rr),latency_p95=sorted(r['latency_seconds'] for r in rr)[94])
    paired={}
    for s in SYSTEMS:
        paired[s]={}
        for bucket in ['all','unseen','rare','common']:
            ii=[i for i,x in enumerate(labels) if bucket=='all' or x['bucket'][s]==bucket];a=[vectors['lexical'][s][i] for i in ii];b=[vectors['semantic'][s][i] for i in ii]
            gains=sum(y and not x for x,y in zip(a,b));losses=sum(x and not y for x,y in zip(a,b))
            paired[s][bucket]=dict(n=len(ii),corrections=gains,regressions=losses,delta=(gains-losses)/len(ii),cluster_95_interval=cluster_interval(a,b,[labels[i]['group_id'] for i in ii]))
    import random
    rng=random.Random(13);boot=[]
    for _ in range(5000):
        ii=[rng.randrange(100) for _ in labels];score=lambda mode:min(sum(vectors[mode][s][i] for i in ii)/100 for s in SYSTEMS);boot.append(score('semantic')-score('lexical'))
    boot.sort();shared=sum(a['request_sha256']==b['request_sha256'] for a,b in zip(all_records['lexical'],all_records['semantic']))
    for a,b in zip(all_records['lexical'],all_records['semantic']):
        if a['request_sha256']==b['request_sha256']:assert a['generation_id']==b['generation_id'] and a['prediction']==b['prediction']
    cost=sum(summary[m]['new_cost_usd'] for m in MODES)
    write(EXPERIMENT/'results.json',dict(arms=summary,paired=paired,headline_delta=summary['semantic']['headline']-summary['lexical']['headline'],headline_cluster_95_interval=[boot[125],boot[4875]],
        new_cost_usd=cost,fresh_requests=len(list((EXPERIMENT/'responses').glob('*/*.json'))),shared_identical_control_requests=shared,
        limitation='Previously exposed 100-input development pilot; canonical cached lexical controls, fresh semantic calls for changed bodies, fixed candidates and guide. No new-input generalization, fresh full-agent run or automatic500 expansion.'))
    print(json.dumps(read_json(EXPERIMENT/'results.json'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','audit_gate','freeze','verify','run','analyze']);args=parser.parse_args();globals()[args.command]()
