"""Complete only unfinished CoreWeave conversations, replaying paid responses.

No completed prediction is rerun. CoreWeave's unsupported named tool_choice is
replaced at response5 by auto with finish as the sole permitted function.
"""
import json
import shutil
import time
from pathlib import Path

import requests

from . import dense_agent_eval as e
from .agent import OpenRouterClient
from .common import SEED,read_json,write_json
from .comparisons import ReservedBudget
from .dense_agent_tools import agent_schemas


def main():
    root=e.EXPERIMENT;directory=root/'runs/dense_paging';phase=root/'provider_finish_v1';phase.mkdir(exist_ok=True)
    rows=[read_json(p) for p in directory.glob('*.json')];assert len(rows)==80
    switching=read_json(root/'provider_switch_v1/manifest.json')
    old_bound=switching['failed_DeepInfra_request_bound_usd']
    extra=0.;details=[]
    for r in rows:
        if r['provider_requested']!='CoreWeave':continue
        size=len(json.dumps(r['messages'],ensure_ascii=False).encode())+len(json.dumps(agent_schemas()).encode())+10000
        request_bound=size*.00000013+2048*.00000028
        failed=sum(sum(a.get('status')!=200 for a in u['attempts'])+int(u['usage'].get('cost') is None) for u in r['usage_records'])
        if r['status']=='fallback_error':
            if 'HTTP 404' in r['error']:
                # The unsupported named-tool routing request returns404 without
                # retry in the original client. No model response was generated.
                assert r['model_calls']==4;failed+=1
            elif 'spending cap reached' not in r['error'].lower():raise RuntimeError('Unrecognized error: retain its original cost uncertainty')
        extra+=failed*request_bound
        if failed:details.append(dict(input_id=r['input_id'],failed_request_count=failed,conservative_bound_usd=failed*request_bound))
    retained={r['input_id']:e.file_hash(directory/(r['input_id']+'.json')) for r in rows if r['status']!='fallback_error'}
    pending=[r for r in rows if r['status']=='fallback_error']
    ledger=[read_json(p) for p in (root/'usage_ledger').glob('*.json')]
    budget=ReservedBudget(.25);budget.add(sum(float(u['usage'].get('cost') or 0) for u in ledger));budget.uncertain_cost_bound=old_bound+extra
    manifest=dict(retained_success_sha256=retained,remaining=len(pending),paid_responses_to_replay=sum(r['model_calls'] for r in pending),
        provider='CoreWeave',total_cap_usd=.25,source_sha256=e.file_hash(Path(__file__)),
        previous_failed_request_bound_usd=budget.uncertain_cost_bound,CoreWeave_bound_details=details,
        compatibility='Only response5 changes tool_choice from named finish to auto with sole finish schema; no extra model responses',
        resume='Replay each recorded response and assert identical reconstructed conversation before first new API request; do not charge or resend replayed responses')
    write_json(phase/'manifest.json',manifest)
    e.PROVIDER='CoreWeave'
    print('FINISH_REMAINING',len(pending),'replay_paid_responses',manifest['paid_responses_to_replay'],'known_cost',budget.cost,'retained_failure_bound',budget.uncertain_cost_bound,flush=True)
    for old in sorted(pending,key=lambda r:r['input_id']):
        source=directory/(old['input_id']+'.json');shutil.copyfile(source,phase/(old['input_id']+'.before.json'))
        assistants=[m for m in old['messages'] if m['role']=='assistant']
        assert len(assistants)==old['model_calls']==len(old['usage_records'])
        class ContinuingClient(OpenRouterClient):
            def __init__(self,model,provider,budget):
                assert provider=='CoreWeave';super().__init__(model,provider,budget);self.index=0
            def complete(self,messages,tools,force_finish=False):
                if self.index<len(assistants):
                    j=self.index;self.index+=1;u=old['usage_records'][j]
                    return dict(id=u['generation_id'],provider=u['provider'],model=u['model'],usage=u['usage'],choices=[dict(finish_reason=old['trace'][j]['finish_reason'],message=assistants[j])]),u['attempts']
                if self.index==len(assistants):assert messages==old['messages'],'Replay must reconstruct exact failed-request conversation'
                self.index+=1
                if force_finish:assert [s['function']['name'] for s in tools]==['finish']
                prompt_bound=len(json.dumps(messages,ensure_ascii=False).encode())+len(json.dumps(tools).encode())+10000
                upper=prompt_bound*.00000015+2048*.00000030
                self.budget.reserve(upper)
                body=dict(model=self.model,messages=messages,tools=tools,tool_choice='auto',temperature=0,seed=SEED,max_tokens=2048,
                    reasoning={'enabled':False},usage={'include':True},provider={'order':['CoreWeave'],'allow_fallbacks':False,'require_parameters':True})
                start=time.monotonic()
                try:
                    response=requests.post(self.url,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},json=body,timeout=(15,150))
                    if response.status_code!=200:raise RuntimeError('OpenRouter HTTP '+str(response.status_code)+': '+response.text.replace(self.key,'[REDACTED]')[:800])
                    data=response.json()
                    if data.get('error'):raise RuntimeError(str(data['error']).replace(self.key,'[REDACTED]')[:800])
                    self.budget.add(float(data.get('usage',{}).get('cost') or 0))
                    attempts=[dict(status=200,seconds=round(time.monotonic()-start,3))]
                    write_json(root/'usage_ledger'/(data['id']+'.json'),dict(generation_id=data['id'],usage=data.get('usage',{}),provider=data.get('provider'),attempts=attempts))
                    self.budget.release(upper,upper if data.get('usage',{}).get('cost') is None else 0)
                    return data,attempts
                except Exception:self.budget.release(upper,upper);raise
        item=next(x for x in read_json(root/'inputs.json') if x['input_id']==old['input_id'])
        result=e.run_one(item,budget,client_factory=ContinuingClient)
        assert result['model_calls']<=5
        write_json(source,result)
        print('COMPLETED',result['input_id'],result['status'],'responses',result['model_calls'],'reported_total',round(budget.cost,6),flush=True)
        write_json(phase/'execution.json',dict(reported_cost_all_phases_usd=budget.cost,uncertain_cost_upper_bound_usd=budget.uncertain_cost_bound,cap_usd=.25))
        if result['status']=='fallback_error':raise RuntimeError('Unfinished conversation still blocked: '+str(result['error']))
    for key,expected in retained.items():assert e.file_hash(directory/(key+'.json'))==expected
    for key,expected in switching['retained_success_sha256'].items():assert e.file_hash(directory/(key+'.json'))==expected
    for relative,expected in read_json(root/'retained_baseline.json').items():assert e.file_hash(e.SOURCE/relative)==expected
    result=e.score()
    result['reported_cost_including_abandoned_attempts_usd']=budget.cost;result['uncertain_cost_upper_bound_usd']=budget.uncertain_cost_bound
    result['dense_agent_provider_counts']={p:sum(read_json(f)['provider_requested']==p for f in directory.glob('*.json')) for p in ('DeepInfra','CoreWeave')}
    result['limits']+=['User-approved remaining-runs provider switch:34 retained DeepInfra FP8;46 CoreWeave FP8; aggregate provider effects not isolated','CoreWeave fifth response uses auto with finish-only schema because named forced tool was unsupported; maximum5 successful responses unchanged','Previously paid partial responses replayed locally and not resent to the API']
    write_json(root/'results.json',result)


if __name__=='__main__':main()
