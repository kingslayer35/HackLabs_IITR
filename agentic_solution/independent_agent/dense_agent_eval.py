"""Paired dense-retrieval/paging agent experiment, reusing the original80 control."""
import argparse
import copy
import csv
import json
import re
import shutil
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed

import requests

from .agent import OpenRouterClient,validate_finish
from .common import ARTIFACTS,SEED,SYSTEMS,digest,read_json,write_json
from .comparisons import ReservedBudget,TASK_PROMPT,assert_flat,paired_delta
from .recovery_eval import BASE_POLICY
from .selection_prompt_eval import file_hash
from .dense_agent_tools import DenseAgentTools,agent_schemas

SOURCE=ARTIFACTS/'recovery_v1'
EXPERIMENT=ARTIFACTS/'dense_agent_v1'
MODEL='deepseek/deepseek-v4-flash-0731'
PROVIDER='DeepInfra'
MAX_TURNS=5
URL='http://127.0.0.1:18473'


def prepare():
    EXPERIMENT.mkdir(exist_ok=True)
    if (EXPERIMENT/'prepared.json').exists():return read_json(EXPERIMENT/'prepared.json')
    protected=[SOURCE/n for n in ('inputs.json','labels.json','examples.sqlite','leakage_audit.json')]
    protected+=sorted((SOURCE/'runs/baseline').glob('*.json'))+sorted((SOURCE/'initial_evidence').glob('*.json'))
    retained={str(p.relative_to(SOURCE)):file_hash(p) for p in protected}
    for p in protected:
        relative=p.relative_to(SOURCE)
        if relative.parts[0]=='initial_evidence':relative=type(relative)('baseline_initial_evidence')/relative.name
        target=EXPERIMENT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    inputs=read_json(EXPERIMENT/'inputs.json');assert len(inputs)==80
    response=requests.get(URL+'/health',timeout=30);response.raise_for_status();service=response.json()
    for i,x in enumerate(inputs):
        baseline=read_json(SOURCE/'initial_evidence'/(x['input_id']+'.json'))
        oldrun=read_json(SOURCE/'runs/baseline'/(x['input_id']+'.json'))
        assert oldrun['messages'][0]['content']==TASK_PROMPT+'\n'+BASE_POLICY
        tools=DenseAgentTools(EXPERIMENT/'examples.sqlite')
        initial={s:tools.search_concepts(s,[x['entity']],12) for s in SYSTEMS}
        payload=copy.deepcopy(baseline['payload']);payload['initial_candidates']=initial
        for s in SYSTEMS:
            for e in payload['training_examples'][s]['examples']:tools.card(s,e['code'])
        fallback={s:next((r['code'] for r in initial[s]['results'] if tools.valid(s,r['code'])),baseline['fallback'][s]) for s in SYSTEMS}
        assert_flat(payload)
        evidence=dict(input_id=x['input_id'],payload=payload,seen={s:sorted(tools.seen[s]) for s in SYSTEMS},fallback=fallback,
            sha256=digest(json.dumps(payload,sort_keys=True,ensure_ascii=False)))
        write_json(EXPERIMENT/'initial_evidence'/(x['input_id']+'.json'),evidence)
        write_json(EXPERIMENT/'initial_pools'/(x['input_id']+'.json'),tools.pools)
        tools.close()
        if (i+1)%20==0:print('PREPARED',i+1,'/80',flush=True)
    write_json(EXPERIMENT/'retained_baseline.json',retained)
    manifest=dict(n=80,baseline_reused=80,model=MODEL,provider=PROVIDER,max_responses=MAX_TURNS,
        intervention='new ICD dense/SNOMED cascade backend, hidden pools100, deeper-page tool; task prompt, model, examples and response budget unchanged',
        sample='same previously inspected80 development inputs;500 retrieval development inputs not sent to LLM in this experiment',
        service=service,preprocessing_enabled=False,graph_enabled=False,
        price_check=dict(source='https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints',prompt_per_token=.00000006,completion_per_token=.00000018))
    write_json(EXPERIMENT/'prepared.json',manifest);return manifest


class GuardedClient(OpenRouterClient):
    def complete(self,messages,tools,force_finish=False):
        prompt_bound=len(json.dumps(messages,ensure_ascii=False).encode())+len(json.dumps(tools).encode())+10000
        request_upper=prompt_bound*.00000012+2048*.00000020
        upper=2*request_upper;self.budget.reserve(upper)
        body=dict(model=self.model,messages=messages,tools=tools,tool_choice={'type':'function','function':{'name':'finish'}} if force_finish else 'auto',
            temperature=0,seed=SEED,max_tokens=2048,reasoning={'enabled':False},usage={'include':True},
            provider={'order':[self.provider],'allow_fallbacks':False,'require_parameters':True})
        attempts=[]
        try:
            for attempt in range(2):
                started=time.monotonic()
                try:
                    response=requests.post(self.url,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},json=body,timeout=(15,150))
                    if response.status_code!=200:
                        error=response.text.replace(self.key,'[REDACTED]')[:800]
                        attempts.append(dict(status=response.status_code,seconds=round(time.monotonic()-started,3),error=error))
                        if attempt==0 and (response.status_code in {408,429} or response.status_code>=500):time.sleep(2);continue
                        raise RuntimeError(f'OpenRouter HTTP {response.status_code}: {error}')
                    data=response.json()
                    if data.get('error'):raise RuntimeError(str(data['error']).replace(self.key,'[REDACTED]')[:800])
                    attempts.append(dict(status=200,seconds=round(time.monotonic()-started,3)))
                    self.budget.add(float(data.get('usage',{}).get('cost') or 0))
                    write_json(EXPERIMENT/'usage_ledger'/(data['id']+'.json'),dict(generation_id=data['id'],usage=data.get('usage',{}),provider=data.get('provider'),attempts=attempts))
                    failed=sum(a.get('status')!=200 for a in attempts);missing=int(data.get('usage',{}).get('cost') is None)
                    self.budget.release(upper,request_upper*(failed+missing));return data,attempts
                except (requests.Timeout,requests.ConnectionError) as exc:
                    attempts.append(dict(error=type(exc).__name__,seconds=round(time.monotonic()-started,3)))
                    if attempt:raise RuntimeError(type(exc).__name__+' after two attempts') from None
                    time.sleep(2)
            raise RuntimeError('Request did not complete')
        except Exception:
            self.budget.release(upper,upper);raise


def run_one(item,budget,client_factory=GuardedClient):
    if set(item)!={'input_id','entity','chunk_text'}:raise ValueError('Only unlabelled input fields permitted')
    started=time.monotonic();evidence=read_json(EXPERIMENT/'initial_evidence'/(item['input_id']+'.json'))
    tools=DenseAgentTools(EXPERIMENT/'examples.sqlite');tools.seen={s:set(evidence['seen'][s]) for s in SYSTEMS}
    tools.pools=read_json(EXPERIMENT/'initial_pools'/(item['input_id']+'.json'))
    messages=[dict(role='system',content=TASK_PROMPT+'\n'+BASE_POLICY),dict(role='user',content=json.dumps(evidence['payload'],ensure_ascii=False))]
    client=client_factory(MODEL,PROVIDER,budget);trace=[];usages=[];prediction=None;error=None;status='model'
    visible={s:set(evidence['seen'][s]) for s in SYSTEMS}
    try:
        for turn in range(MAX_TURNS):
            # Only evidence returned before this response can support its finish.
            visible={s:set(tools.seen[s]) for s in SYSTEMS}
            allowed=agent_schemas(final_turn=turn==MAX_TURNS-1)
            data,attempts=client.complete(messages,allowed,force_finish=turn==MAX_TURNS-1)
            usages.append(dict(usage=data.get('usage',{}),provider=data.get('provider'),model=data.get('model'),generation_id=data.get('id'),attempts=attempts))
            choice=data['choices'][0];message=choice['message']
            messages.append({k:v for k,v in message.items() if k in {'role','content','tool_calls','reasoning_details'} and v is not None})
            record=dict(turn=turn+1,finish_reason=choice.get('finish_reason'),tool_calls=[],assistant_content=message.get('content'));trace.append(record)
            def problem(candidate):
                issue=validate_finish(candidate,tools)
                if issue:return issue
                if any(candidate[f] not in visible[s] for s,f in zip(SYSTEMS,('icd10cm_code','snomed_code'))):return 'Confirm codes in a tool result before the response that finishes; read new results first'
                return None
            calls=message.get('tool_calls',[])
            if not calls:
                try:
                    candidate=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',message.get('content') or '').strip())
                    if problem(candidate) is None:prediction=candidate;break
                except (ValueError,TypeError,AttributeError):pass
                messages.append(dict(role='user',content='Use the permitted tools, then return both codes using finish within the remaining response budget.'));continue
            if len(calls)>12:raise RuntimeError('Per-response tool limit exceeded')
            permitted={s['function']['name'] for s in allowed}
            for call in calls:
                name=call['function']['name'];arguments={}
                try:
                    arguments=json.loads(call['function']['arguments'])
                    if not isinstance(arguments,dict):raise ValueError('Tool arguments must be an object')
                    if name not in permitted:raise ValueError('Tool disabled in this phase')
                    if name=='finish':
                        issue=problem(arguments);result={'error':issue} if issue else {'accepted':True}
                        if not issue:prediction=arguments
                    else:result=tools.dispatch(name,arguments);assert_flat(result)
                except (ValueError,TypeError,sqlite3.Error) as exc:result={'error':str(exc)}
                record['tool_calls'].append(dict(name=name,arguments=arguments,result=result))
                messages.append(dict(role='tool',tool_call_id=call['id'],content=json.dumps(result,ensure_ascii=False)))
                if prediction:break
            if prediction:break
    except Exception as exc:error=str(exc).replace(client.key,'[REDACTED]')[:1200];status='fallback_error'
    if prediction is None:
        if status!='fallback_error':status='fallback_budget_or_invalid'
        prediction=dict(icd10cm_code=evidence['fallback']['ICD10CM'],snomed_code=evidence['fallback']['SNOMEDCT_US'],confidence=0.,evidence='Deterministic initial-search fallback; no accepted model finish.')
    result=dict(input_id=item['input_id'],mode='dense_paging',model=MODEL,provider_requested=PROVIDER,prediction=prediction,status=status,error=error,
        initial_evidence_sha256=evidence['sha256'],seen_candidates={s:sorted(visible[s]) for s in SYSTEMS},
        generated_but_not_necessarily_read_candidates={s:sorted(tools.seen[s]) for s in SYSTEMS},pages_read=tools.pages_read,
        trace=trace,messages=messages,model_calls=len(usages),usage_records=usages,
        cost_usd=sum(float(u['usage'].get('cost') or 0) for u in usages),latency_seconds=round(time.monotonic()-started,3))
    tools.close();return result


def config(cap,workers):
    p=__import__('pathlib').Path(__file__).parent
    protected=read_json(EXPERIMENT/'retained_baseline.json')
    return dict(cap_usd=cap,workers=workers,prepared=read_json(EXPERIMENT/'prepared.json'),
        prompt_sha256=digest(TASK_PROMPT+'\n'+BASE_POLICY),schemas_sha256=digest(json.dumps(agent_schemas(),sort_keys=True)),
        source_sha256={f.name:file_hash(f) for f in sorted(p.glob('*.py'))},
        frozen_inputs_and_baseline={str(f.relative_to(EXPERIMENT)):file_hash(f) for directory in ('initial_evidence','initial_pools','runs/baseline') for f in sorted((EXPERIMENT/directory).glob('*.json'))},
        retained_source_verified=all(file_hash(SOURCE/k)==v for k,v in protected.items()),
        sample_sha256={n:file_hash(EXPERIMENT/n) for n in ('inputs.json','labels.json','examples.sqlite','leakage_audit.json')})


def score():
    labels=read_json(EXPERIMENT/'labels.json');out={};vectors={};detailed=[]
    for arm in ('baseline','dense_paging'):
        rows=[read_json(EXPERIMENT/'runs'/arm/(x['input_id']+'.json')) for x in labels]
        if any(r['status']=='fallback_error' for r in rows):raise RuntimeError('Resolve infrastructure errors before scoring; preserve completed predictions')
        systems={};vectors[arm]={}
        for s,field in zip(SYSTEMS,('icd10cm_code','snomed_code')):
            initial12=initialall=ever=correct=miss=selection=deep=0;hits=[]
            for x,r in zip(labels,rows):
                evidence=read_json(EXPERIMENT/('baseline_initial_evidence' if arm=='baseline' else 'initial_evidence')/(x['input_id']+'.json'))
                g=x['gold'][s];pred=r['prediction'][field];hit=g==pred;seen=g in r['seen_candidates'][s]
                if hit and not seen:raise AssertionError('Prediction correct without pre-decision visibility')
                top=g in [c['code'] for c in evidence['payload']['initial_candidates'][s]['results']]
                initial12+=top;initialall+=g in evidence['seen'][s];ever+=seen;correct+=hit;miss+=not hit and not seen;selection+=not hit and seen;hits.append(int(hit))
                inpool=None
                if arm=='dense_paging':
                    pools=read_json(EXPERIMENT/'initial_pools'/(x['input_id']+'.json'));pool=next(v for v in pools.values() if v['system']==s)
                    inpool=g in [v['code'] for v in pool['results']];deep+=inpool
                detailed.append(dict(arm=arm,input_id=x['input_id'],entity=x['entity'],system=s,gold=g,predicted=pred,correct=hit,initial_dictionary12=top,initial_evidence=g in evidence['seen'][s],ever_seen=seen,initial_hidden100=inpool,status=r['status']))
            systems[s]=dict(n=80,correct=correct,accuracy=correct/80,initial_dictionary12=initial12,initial_evidence_including_examples=initialall,ever_seen=ever,ever_seen_recall=ever/80,
                retrieval_misses=miss,selection_errors=selection,initial_hidden100=deep if arm=='dense_paging' else None)
            vectors[arm][s]=hits
        both=[a*b for a,b in zip(vectors[arm]['ICD10CM'],vectors[arm]['SNOMEDCT_US'])];vectors[arm]['both']=both
        calls=Counter(c['name'] for r in rows for t in r['trace'] for c in t['tool_calls'] if c['name']!='finish')
        out[arm]=dict(per_system=systems,headline_min_accuracy=min(s['accuracy'] for s in systems.values()),both_correct=sum(both),
            statuses=dict(Counter(r['status'] for r in rows)),model_calls=sum(r['model_calls'] for r in rows),tool_calls=dict(calls),cost_usd=sum(r['cost_usd'] for r in rows),
            inputs_using_paging=sum(any(c['name']=='get_more_candidates' for t in r['trace'] for c in t['tool_calls']) for r in rows))
    result=dict(n=80,arms=out,paired={s:paired_delta(vectors['dense_paging'][s],vectors['baseline'][s]) for s in (*SYSTEMS,'both')},
        new_inference_cost_usd=out['dense_paging']['cost_usd'],baseline_API_calls_this_experiment=0,
        leakage_audit=read_json(EXPERIMENT/'leakage_audit.json'),limits=['Reused80 development inputs; cached baseline not simultaneous inference','Backend and paging change together; aggregate effects do not isolate paging','Rank100 pool presence is diagnostic, not model visibility','No preprocessing, graph, model or task-prompt change'])
    write_json(EXPERIMENT/'results.json',result)
    with (EXPERIMENT/'all_predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(detailed[0]));w.writeheader();w.writerows(detailed)
    print(json.dumps(result,indent=2),flush=True);return result


def run(cap=.25,workers=4):
    cfg=config(cap,workers);path=EXPERIMENT/'config.json'
    assert cfg['retained_source_verified']
    if path.exists():assert read_json(path)==cfg,'Frozen configuration changed'
    write_json(path,cfg)
    directory=EXPERIMENT/'runs/dense_paging';directory.mkdir(parents=True,exist_ok=True)
    jobs=[];retained=0
    for x in read_json(EXPERIMENT/'inputs.json'):
        target=directory/(x['input_id']+'.json')
        if target.exists():
            old=read_json(target)
            if old['status']!='fallback_error':retained+=1;continue
            archive=EXPERIMENT/'failed_attempts'/x['input_id'];archive.mkdir(parents=True,exist_ok=True);target.rename(archive/f'attempt_{len(list(archive.glob("*.json")))+1}.json')
        jobs.append((x,target))
    budget=ReservedBudget(cap);budget.add(sum(float(read_json(p)['usage'].get('cost') or 0) for p in (EXPERIMENT/'usage_ledger').glob('*.json')))
    execution=EXPERIMENT/'execution.json'
    if execution.exists():budget.uncertain_cost_bound=read_json(execution)['uncertain_cost_upper_bound_usd']
    print(f'Dense agent: {len(jobs)} pending, {retained} preserved; baseline80 reused; cap ${cap:.2f}',flush=True)
    def work(job):
        x,path=job;result=run_one(x,budget);write_json(path,result);return result
    start=time.monotonic()
    jobs.sort(key=lambda j:digest('dense-agent:'+j[0]['input_id']))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(work,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();print(f'[{i}/{len(jobs)}] {r["input_id"]} {r["status"]} calls={r["model_calls"]} cost=${r["cost_usd"]:.5f} total=${budget.cost:.5f}',flush=True)
    write_json(execution,dict(wall_seconds_this_invocation=round(time.monotonic()-start,3),reported_cost_usd=budget.cost,uncertain_cost_upper_bound_usd=budget.uncertain_cost_bound,cap_usd=cap,baseline_API_calls=0))
    assert config(cap,workers)==cfg,'Frozen data/source changed during evaluation'
    return score()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','score']);parser.add_argument('--max-cost',type=float,default=.25);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.command=='run':run(args.max_cost,args.workers)
    else:score()


if __name__=='__main__':main()
