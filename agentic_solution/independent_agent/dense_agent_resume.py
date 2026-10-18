"""User-authorized provider switch for unfinished dense-agent runs only.

Original scripts, configuration and all completed responses remain unchanged.
Reconcile failed-request bounds at verified DeepInfra prices rather than assuming
429s were free; keep the experiment's original total $0.25 cap.
"""
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from . import dense_agent_eval as e
from .common import digest,read_json,write_json
from .comparisons import ReservedBudget
from .dense_agent_tools import agent_schemas

PROVIDER='CoreWeave'
CAP=.25


class ScaledReservations:
    # The original request guard bounds prices at .12/.20 per million tokens.
    # 1.5x gives .18/.30, above CoreWeave's verified .13/.28.
    def __init__(self,budget):self.budget=budget
    def reserve(self,value):self.budget.reserve(value*1.5)
    def release(self,value,uncertain):self.budget.release(value*1.5,uncertain*1.5)
    def add(self,cost):self.budget.add(cost)


class AlternateClient(e.GuardedClient):
    def __init__(self,model,provider,budget):
        assert provider==PROVIDER
        super().__init__(model,provider,ScaledReservations(budget))


def verified_old_failure_bound(rows):
    bound=0.;details=[]
    for r in rows:
        # Full final conversation is at least as long as any preceding request.
        prompt_bytes=len(json.dumps(r['messages'],ensure_ascii=False).encode())+len(json.dumps(agent_schemas()).encode())+10000
        one_request=prompt_bytes*.00000006+2048*.00000018
        failed=0
        if r['status']=='fallback_error':
            if 'OpenRouter HTTP 429' in (r['error'] or ''):failed+=2
            elif 'spending cap reached' not in (r['error'] or '').lower():
                raise RuntimeError('Unrecognized failed request; do not discard its uncertainty')
        for usage in r['usage_records']:
            failed+=sum(a.get('status')!=200 for a in usage['attempts'])
            failed+=int(usage['usage'].get('cost') is None)
        bound+=failed*one_request
        if failed:details.append(dict(input_id=r['input_id'],failed_request_upper_count=failed,conservative_bound_usd=failed*one_request))
    return bound,details


def main():
    root=e.EXPERIMENT;phase=root/'provider_switch_v1';phase.mkdir(exist_ok=True)
    original_config=read_json(root/'config.json')
    source_root=Path(__file__).parent
    for name,expected in original_config['source_sha256'].items():assert e.file_hash(source_root/name)==expected,'Original source changed'
    for relative,expected in read_json(root/'retained_baseline.json').items():assert e.file_hash(e.SOURCE/relative)==expected
    directory=root/'runs/dense_paging'
    inputs=read_json(root/'inputs.json');rows=[read_json(directory/(x['input_id']+'.json')) for x in inputs if (directory/(x['input_id']+'.json')).exists()]
    if not (phase/'manifest.json').exists():
        assert len(rows)==80
        assert all(r['provider_requested']=='DeepInfra' for r in rows)
        retained={r['input_id']:e.file_hash(directory/(r['input_id']+'.json')) for r in rows if r['status']!='fallback_error'}
        old_execution=read_json(root/'execution.json')
        uncertainty,details=verified_old_failure_bound(rows)
        assert uncertainty<=old_execution['uncertain_cost_upper_bound_usd']
        shutil.copyfile(root/'execution.json',phase/'execution_before_switch.json')
        manifest=dict(provider=PROVIDER,endpoint='coreweave/fp8',model=e.MODEL,prices=dict(prompt=.00000013,completion=.00000028),
            retained_success_sha256=retained,retained_success_count=len(retained),remaining_count=80-len(retained),
            failed_DeepInfra_request_bound_usd=uncertainty,failure_bound_details=details,
            bound_note='Every explicit429 is still budgeted at up to two full max-token requests, using verified original DeepInfra prices and UTF8-byte prompt bound; preflight spending refusals make no request. No assumption failed requests were free.',
            source_sha256=e.file_hash(Path(__file__)),original_config_sha256=e.file_hash(root/'config.json'),
            user_instruction='Change provider for remaining runs; retain completed runs',workers=2,total_cap_usd=CAP,
            sampling='same seed13, temperature0, reasoning disabled, max2048, max5 model responses')
        write_json(phase/'manifest.json',manifest)
    manifest=read_json(phase/'manifest.json')
    assert manifest['source_sha256']==e.file_hash(Path(__file__))
    for key,expected in manifest['retained_success_sha256'].items():assert e.file_hash(directory/(key+'.json'))==expected
    ledger=[read_json(p) for p in (root/'usage_ledger').glob('*.json')]
    budget=ReservedBudget(CAP);budget.add(sum(float(r['usage'].get('cost') or 0) for r in ledger))
    budget.uncertain_cost_bound=manifest['failed_DeepInfra_request_bound_usd']
    if (phase/'execution.json').exists():budget.uncertain_cost_bound=read_json(phase/'execution.json')['uncertain_cost_upper_bound_usd']
    jobs=[]
    for x in inputs:
        p=directory/(x['input_id']+'.json')
        if p.exists() and read_json(p)['status']!='fallback_error':continue
        if p.exists():
            archive=root/'failed_attempts'/x['input_id'];archive.mkdir(parents=True,exist_ok=True);p.rename(archive/f'attempt_{len(list(archive.glob("*.json")))+1}.json')
        jobs.append((x,p))
    jobs.sort(key=lambda j:digest('dense-agent:'+j[0]['input_id']))
    e.PROVIDER=PROVIDER
    print(f'Switching remaining {len(jobs)} inputs to {PROVIDER}; retained original {len(manifest["retained_success_sha256"])} successes; total cap ${CAP:.2f}; retained failure bound ${budget.uncertain_cost_bound:.5f}',flush=True)
    def work(job):
        x,p=job;result=e.run_one(x,budget,client_factory=AlternateClient);write_json(p,result);return result
    started=time.monotonic();successes=0
    try:
        # Verify one real input before launching the rest. It is part of the eval,
        # never a throwaway model call and never a rerun of a completed input.
        if jobs:
            first=work(jobs.pop(0));print('PROVIDER_CHECK',first['input_id'],first['status'],first['error'],flush=True)
            if first['status']=='fallback_error':raise RuntimeError('Alternate provider preflight failed; preserve responses and diagnose before continuing')
            successes+=1
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(work,j) for j in jobs]
            for f in as_completed(futures):
                r=f.result();successes+=1
                print(f'[{successes}/{len(jobs)+1}] {r["input_id"]} {r["status"]} calls={r["model_calls"]} cost=${r["cost_usd"]:.5f} total=${budget.cost:.5f}',flush=True)
    finally:
        write_json(phase/'execution.json',dict(wall_seconds_this_invocation=round(time.monotonic()-started,3),reported_cost_all_phases_usd=budget.cost,uncertain_cost_upper_bound_usd=budget.uncertain_cost_bound,total_cap_usd=CAP,provider=PROVIDER))
    for key,expected in manifest['retained_success_sha256'].items():assert e.file_hash(directory/(key+'.json'))==expected
    for relative,expected in read_json(root/'retained_baseline.json').items():assert e.file_hash(e.SOURCE/relative)==expected
    result=e.score()
    result['limits'].append('User-requested remaining-runs provider switch:34 successful DeepInfra runs retained; remaining46 routed to CoreWeave FP8, same model/seed; provider differences confound aggregate attribution')
    result['reported_cost_including_abandoned_attempts_usd']=budget.cost
    result['uncertain_cost_upper_bound_usd']=budget.uncertain_cost_bound
    result['dense_agent_provider_counts']={p:sum(read_json(directory/(x['input_id']+'.json'))['provider_requested']==p for x in inputs) for p in ('DeepInfra',PROVIDER)}
    write_json(root/'results.json',result)


if __name__=='__main__':main()
