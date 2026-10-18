"""Matched, resumable 500-input comparison with a shared spending cap."""
import argparse
import copy
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import OpenRouterClient, validate_finish
from .backend_ablation import AblationTools, schemas as icd_schemas
from .common import ARTIFACTS, SYSTEMS, SEED, digest, read_json, write_json
from .comparisons import PolicyTools, ReservedBudget, TASK_PROMPT, assert_flat, schemas_for
from .diverse500_prepare import EXPERIMENT, ARMS
from .recovery_eval import BASE_POLICY
from .selection_prompt_eval import file_hash

MODEL='deepseek/deepseek-v4-flash-0731'
PROVIDER='CoreWeave'
FIELDS=dict(zip(SYSTEMS,('icd10cm_code','snomed_code')))
CAP=2.0
WORKERS=4


def schemas(arm,final=False):
    if arm not in ARMS: raise ValueError('Unknown arm')
    return schemas_for('search_inspect',final) if arm==ARMS[0] else icd_schemas('icd_only',final)


def make_tools(arm):
    if arm==ARMS[0]: return PolicyTools(EXPERIMENT/'examples.sqlite')
    if arm==ARMS[1]: return AblationTools(EXPERIMENT/'examples.sqlite','icd_only')
    raise ValueError('Unknown arm')


def fingerprint(messages):
    return digest(json.dumps(messages,sort_keys=True,ensure_ascii=False))


def freeze():
    if (EXPERIMENT/'config.json').exists(): verify();return
    inputs=read_json(EXPERIMENT/'inputs.json');assert len(inputs)==500
    evidence={}
    for x in inputs:
        name=x['input_id']+'.json'
        a=read_json(EXPERIMENT/ARMS[0]/'initial_evidence'/name)
        b=read_json(EXPERIMENT/ARMS[1]/'initial_evidence'/name)
        assert a['payload']['training_examples']==b['payload']['training_examples']
        assert a['payload']['initial_candidates']['SNOMEDCT_US']==b['payload']['initial_candidates']['SNOMEDCT_US']
        assert a['seen']['SNOMEDCT_US']==b['seen']['SNOMEDCT_US']
        for arm in ARMS:
            path=EXPERIMENT/arm/'initial_evidence'/name
            evidence[str(path.relative_to(EXPERIMENT))]=file_hash(path)
        path=EXPERIMENT/ARMS[1]/'initial_pools'/name
        evidence[str(path.relative_to(EXPERIMENT))]=file_hash(path)
    protected={str(f.relative_to(ARTIFACTS)):file_hash(f) for name in ('backend_ablation_v1','dense_agent_v1','recovery_v1')
               for f in (ARTIFACTS/name).rglob('*') if f.is_file()}
    write_json(EXPERIMENT/'retained_sources.json',protected)
    config=dict(n=500,arms=ARMS,model=MODEL,provider=PROVIDER,cap_usd=CAP,workers=WORKERS,
                temperature=0,seed=SEED,max_responses=5,max_completion_tokens=2048,reasoning=False,
                task_prompt=TASK_PROMPT,policy=BASE_POLICY,final_tool_choice='auto with finish-only menu for BOTH arms',
                model_scope='Both arms freshly evaluated. No added recovery/comparison/graph/selection instructions.',
                source_sha256={str(p):file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                frozen_sha256={n:file_hash(EXPERIMENT/n) for n in ('inputs.json','labels.json','examples.sqlite','selection_manifest.json','leakage_audit.json','training_code_frequencies.json','provider_prices.json','service_execution.json')},
                evidence_sha256=evidence)
    write_json(EXPERIMENT/'config.json',config)
    verify();print('FROZEN: 500 inputs, identical examples and SNOMED evidence, both arms CoreWeave, shared $2 cap',flush=True)


def verify():
    cfg=read_json(EXPERIMENT/'config.json')
    for name,expected in cfg['source_sha256'].items(): assert file_hash(Path(name))==expected,name
    for name,expected in {**cfg['frozen_sha256'],**cfg['evidence_sha256']}.items(): assert file_hash(EXPERIMENT/name)==expected,name
    for name,expected in read_json(EXPERIMENT/'retained_sources.json').items(): assert file_hash(ARTIFACTS/name)==expected,name


class Client(OpenRouterClient):
    def complete(self,messages,tools,force_finish=False):
        assert self.model==MODEL and self.provider==PROVIDER
        size=len(json.dumps(messages,ensure_ascii=False).encode())+len(json.dumps(tools).encode())+10000
        one=size*.00000013+2048*.00000028
        self.budget.reserve(2*one)
        body=dict(model=self.model,messages=messages,tools=tools,tool_choice='auto',temperature=0,seed=SEED,
                  max_tokens=2048,reasoning={'enabled':False},usage={'include':True},
                  provider={'order':[PROVIDER],'allow_fallbacks':False,'require_parameters':True})
        attempts=[]
        try:
            for attempt in range(2):
                try:
                    r=requests.post(self.url,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},json=body,timeout=(15,120))
                    attempts.append(dict(status=r.status_code))
                    if r.status_code!=200:
                        if not attempt and (r.status_code in {408,429} or r.status_code>=500):time.sleep(3);continue
                        raise RuntimeError('OpenRouter HTTP '+str(r.status_code)+': '+r.text.replace(self.key,'[REDACTED]')[:500])
                    data=r.json()
                    if data.get('error'):raise RuntimeError(str(data['error']).replace(self.key,'[REDACTED]')[:500])
                    cost=data.get('usage',{}).get('cost');self.budget.add(float(cost or 0))
                    write_json(EXPERIMENT/'usage_ledger'/(data['id']+'.json'),dict(generation_id=data['id'],provider=data.get('provider'),usage=data.get('usage',{}),attempts=attempts))
                    self.budget.release(2*one,one*(sum(a.get('status')!=200 for a in attempts)+int(cost is None)))
                    return data,attempts
                except (requests.Timeout,requests.ConnectionError) as exc:
                    attempts.append(dict(error=type(exc).__name__))
                    if attempt:raise RuntimeError('Transport failed after two attempts') from None
                    time.sleep(3)
        except Exception:
            self.budget.release(2*one,2*one);raise


def run_one(item,arm,budget,client_factory=Client,tool_factory=make_tools,prior=None):
    if set(item)!={'input_id','entity','chunk_text'}:raise ValueError('Only unlabelled fields permitted')
    started=time.monotonic();name=item['input_id']+'.json';evidence=read_json(EXPERIMENT/arm/'initial_evidence'/name)
    tools=tool_factory(arm);tools.seen={s:set(evidence['seen'][s]) for s in SYSTEMS}
    if arm==ARMS[1]:
        tools.pools=read_json(EXPERIMENT/arm/'initial_pools'/name)
        for s in SYSTEMS:tools.exposed[s]={r['code'] for r in evidence['payload']['initial_candidates'][s]['results']}
    messages=[dict(role='system',content=TASK_PROMPT+'\n'+BASE_POLICY),dict(role='user',content=json.dumps(evidence['payload'],ensure_ascii=False))]
    client=client_factory(MODEL,PROVIDER,budget)
    trace=[];responses=[];usages=[];prediction=None;status='model';error=None
    replay=(prior or {}).get('responses',[]);visible={s:set(tools.seen[s]) for s in SYSTEMS}
    try:
        for turn in range(1,6):
            visible={s:set(tools.seen[s]) for s in SYSTEMS};allowed=schemas(arm,turn==5)
            request_sha=fingerprint(messages)
            if turn<=len(replay):
                saved=replay[turn-1]
                if saved['request_sha256']!=request_sha:raise RuntimeError('Replay conversation differs from saved paid response; stop rather than resend')
                data,attempts=copy.deepcopy(saved['data']),saved['attempts']
            else:data,attempts=client.complete(messages,allowed,force_finish=turn==5)
            responses.append(dict(request_sha256=request_sha,data=data,attempts=attempts))
            usages.append(dict(usage=data.get('usage',{}),provider=data.get('provider'),generation_id=data.get('id'),attempts=attempts))
            message=data['choices'][0]['message']
            messages.append({k:v for k,v in message.items() if k in {'role','content','tool_calls','reasoning_details'} and v is not None})
            record=dict(turn=turn,tool_calls=[],assistant_content=message.get('content'));trace.append(record)
            def finish(candidate):
                issue=validate_finish(candidate,tools)
                if not issue and any(candidate[FIELDS[s]] not in visible[s] for s in SYSTEMS):issue='Read new code evidence in a later response before selecting it'
                return issue
            calls=message.get('tool_calls',[])
            if len(calls)>12:raise RuntimeError('Per-response tool cap exceeded')
            if not calls:
                try:
                    candidate=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',message.get('content') or '').strip())
                    if finish(candidate) is None:prediction=candidate;break
                except (ValueError,TypeError,AttributeError):pass
                messages.append(dict(role='user',content='Use permitted tools and finish with both codes within the remaining budget.'));continue
            permitted={t['function']['name'] for t in allowed}
            for call in calls:
                fname=call['function']['name'];arguments={}
                try:
                    arguments=json.loads(call['function']['arguments'])
                    if not isinstance(arguments,dict):raise ValueError('Tool arguments must be an object')
                    if fname not in permitted:raise ValueError('Tool disabled in this phase')
                    if fname=='finish':
                        issue=finish(arguments);result={'error':issue} if issue else {'accepted':True}
                        if not issue:prediction=arguments
                    else:result=tools.dispatch(fname,arguments);assert_flat(result)
                except (ValueError,TypeError,sqlite3.Error) as exc:result={'error':str(exc)}
                record['tool_calls'].append(dict(name=fname,arguments=arguments,result=result))
                messages.append(dict(role='tool',tool_call_id=call['id'],content=json.dumps(result,ensure_ascii=False)))
                if prediction:break
            if prediction:break
    except Exception as exc:status='fallback_error';error=str(exc).replace(client.key,'[REDACTED]')[:1200]
    if prediction is None:
        if status!='fallback_error':status='fallback_budget_or_invalid'
        prediction=dict(icd10cm_code=evidence['fallback']['ICD10CM'],snomed_code=evidence['fallback']['SNOMEDCT_US'],confidence=0.,evidence='Deterministic initial-search fallback; no accepted model finish.')
    result=dict(input_id=item['input_id'],arm=arm,model=MODEL,provider_requested=PROVIDER,status=status,error=error,
                prediction=prediction,trace=trace,messages=messages,responses=responses,usage_records=usages,
                model_calls=len(responses),replayed_responses=min(len(replay),len(responses)),
                seen_candidates={s:sorted(visible[s]) for s in SYSTEMS},pages_read=getattr(tools,'pages_read',[]),
                cost_usd=sum(float(u['usage'].get('cost') or 0) for u in usages),
                latency_seconds=round(time.monotonic()-started,3)+(prior or {}).get('latency_seconds',0))
    tools.close();return result


def run():
    verify();cfg=read_json(EXPERIMENT/'config.json');budget=ReservedBudget(cfg['cap_usd'])
    budget.add(sum(float(read_json(f)['usage'].get('cost') or 0) for f in (EXPERIMENT/'usage_ledger').glob('*.json')))
    if (EXPERIMENT/'execution.json').exists():budget.uncertain_cost_bound=read_json(EXPERIMENT/'execution.json')['uncertain_cost_upper_bound_usd']
    inputs=sorted(read_json(EXPERIMENT/'inputs.json'),key=lambda x:digest('paired500:'+x['input_id']))
    jobs=[];retained={}
    for x in inputs:
        for arm in ARMS:
            p=EXPERIMENT/arm/'runs'/(x['input_id']+'.json');prior=None
            if p.exists():
                prior=read_json(p)
                if prior['status']!='fallback_error':retained[str(p.relative_to(EXPERIMENT))]=file_hash(p);continue
                archive=EXPERIMENT/'failed_attempts'/arm/x['input_id'];archive.mkdir(parents=True,exist_ok=True)
                write_json(archive/('attempt_'+str(len(list(archive.glob('*.json')))+1)+'.json'),prior)
            jobs.append((x,arm,p,prior))
    print('Pending',len(jobs),'retained',len(retained),'reported cost',budget.cost,flush=True)
    def work(job):
        x,arm,path,prior=job;result=run_one(x,arm,budget,prior=prior);write_json(path,result);return result
    started=time.monotonic();done=0;failure=None
    try:
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            queue=iter(jobs);pending={}
            for _ in range(cfg['workers']):
                job=next(queue,None)
                if job:pending[pool.submit(work,job)]=job
            while pending:
                f=next(as_completed(pending));pending.pop(f);r=f.result();done+=1
                print(f'[{done}/{len(jobs)}] {r["arm"]} {r["input_id"]} {r["status"]} calls={r["model_calls"]} replayed={r["replayed_responses"]} total=${budget.cost:.5f}',flush=True)
                if r['status']=='fallback_error':failure=r['error']
                if failure is None:
                    job=next(queue,None)
                    if job:pending[pool.submit(work,job)]=job
    finally:
        write_json(EXPERIMENT/'execution.json',dict(reported_cost_usd=budget.cost,
                   uncertain_cost_upper_bound_usd=budget.uncertain_cost_bound,cap_usd=cfg['cap_usd'],
                   seconds_this_invocation=round(time.monotonic()-started,3),completed_this_invocation=done,
                   pending_this_invocation=len(jobs)-done,failure=failure))
        for name,sha in retained.items():assert file_hash(EXPERIMENT/name)==sha,name
        verify()
    if failure:raise RuntimeError('Infrastructure pause; completed runs and paid-response prefixes preserved: '+failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.command=='freeze' else run()
