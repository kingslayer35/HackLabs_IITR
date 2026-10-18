"""Offline long-tail evaluation with paired context-cluster uncertainty."""
import csv
import json
from collections import Counter,defaultdict

import numpy as np

from .backend_ablation_analysis import visibility
from .common import ARTIFACTS,SYSTEMS,read_json,write_json
from .diverse500_eval import ARMS,EXPERIMENT,FIELDS,verify
from .tools import TerminologyTools


def analyze():
    verify();labels=read_json(EXPERIMENT/'labels.json');group_names=sorted({x['group_id'] for x in labels})
    group_index={g:i for i,g in enumerate(group_names)};ids=np.array([group_index[x['group_id']] for x in labels])
    draws=np.random.default_rng(13).integers(0,len(group_names),size=(5000,len(group_names)))
    def interval(values):
        return [float(v) for v in np.nanquantile(values,[.025,.975])]
    def boot_ratio(numerator,denominator):
        hits=np.bincount(ids,weights=numerator,minlength=len(group_names))
        count=np.bincount(ids,weights=denominator,minlength=len(group_names))
        top=hits[draws].sum(axis=1);bottom=count[draws].sum(axis=1)
        return np.divide(top,bottom,out=np.full_like(top,np.nan),where=bottom>0)
    previous=(read_json(ARTIFACTS/'smoke20.json')+read_json(ARTIFACTS/'comparisons_v1/labels.json')+read_json(ARTIFACTS/'recovery_v1/labels.json'))
    previous_groups={x['group_id'] for x in previous}
    masks={'all':np.ones(500,dtype=bool),'no_prior_agent_context':np.array([x['group_id'] not in previous_groups for x in labels])}
    for b in ('unseen','rare','common'):
        for s in SYSTEMS:masks[s+':'+b]=np.array([x['bucket'][s]==b for x in labels])
    vectors={};boot={};arms={};cases=[];runs_by_arm={};validator=TerminologyTools()
    for arm in ARMS:
        rows=[];systems={};vectors[arm]={};boot[arm]={}
        for x in labels:
            name=x['input_id']+'.json';r=read_json(EXPERIMENT/arm/'runs'/name)
            if r['status']=='fallback_error':raise RuntimeError('Resolve infrastructure failure: '+arm+':'+x['input_id'])
            assert r['model_calls']<=5
            evidence=read_json(EXPERIMENT/arm/'initial_evidence'/name);seen,first=visibility(r,evidence)
            rows.append(r)
            for s in SYSTEMS:
                pred=r['prediction'][FIELDS[s]];assert validator.valid(s,pred)
                gold=x['gold'][s];initial=gold in {v['code'] for v in evidence['payload']['initial_candidates'][s]['results']}
                cases.append(dict(arm=arm,input_id=x['input_id'],group_id=x['group_id'],entity=x['entity'],
                    system=s,gold=gold,predicted=pred,correct=pred==gold,frequency=x['frequency'][s],bucket=x['bucket'][s],
                    domain=x['features']['domain'],initial12=initial,initial_all=gold in evidence['seen'][s],
                    ever_seen=gold in seen[s],first_source=first[s].get(gold),status=r['status'],
                    prior_context=x['group_id'] in previous_groups,evidence=r['prediction']['evidence']))
        runs_by_arm[arm]=rows
        for s in SYSTEMS:
            ss=[c for c in cases if c['arm']==arm and c['system']==s]
            hits=np.array([c['correct'] for c in ss],dtype=float);vectors[arm][s]=hits
            boot[arm][s]=boot_ratio(hits,np.ones(500))
            by_code=defaultdict(list)
            for c in ss:by_code[c['gold']].append(c['correct'])
            systems[s]=dict(n=500,correct=int(hits.sum()),accuracy=float(hits.mean()),
                cluster_bootstrap_95_ci=interval(boot[arm][s]),macro_code_accuracy=float(np.mean([np.mean(v) for v in by_code.values()])),
                unique_codes=len(by_code),initial12=sum(c['initial12'] for c in ss),initial_evidence=sum(c['initial_all'] for c in ss),
                ever_seen=sum(c['ever_seen'] for c in ss),selection_errors=sum(c['ever_seen'] and not c['correct'] for c in ss),
                retrieval_misses=sum(not c['ever_seen'] and not c['correct'] for c in ss),valid=500,buckets={})
            for b in ('unseen','rare','common'):
                mask=masks[s+':'+b];subset=[c for c in ss if c['bucket']==b]
                boot[arm][s+':'+b]=boot_ratio(hits*mask,mask)
                codes={c['gold'] for c in subset}
                systems[s]['buckets'][b]=dict(n=len(subset),correct=sum(c['correct'] for c in subset),unique_codes=len(codes),
                    accuracy=float(hits[mask].mean()) if mask.any() else None,
                    macro_code_accuracy=float(np.mean([np.mean(by_code[code]) for code in codes])) if codes else None,
                    cluster_bootstrap_95_ci=interval(boot[arm][s+':'+b]),
                    initial12=sum(c['initial12'] for c in subset),ever_seen=sum(c['ever_seen'] for c in subset),
                    selection_errors=sum(c['ever_seen'] and not c['correct'] for c in subset),
                    retrieval_misses=sum(not c['ever_seen'] and not c['correct'] for c in subset))
            mask=masks['no_prior_agent_context']
            systems[s]['no_prior_agent_context']=dict(n=int(mask.sum()),correct=int(hits[mask].sum()),accuracy=float(hits[mask].mean()))
        joint=np.logical_and(vectors[arm][SYSTEMS[0]],vectors[arm][SYSTEMS[1]])
        vectors[arm]['both']=joint;boot[arm]['both']=boot_ratio(joint,np.ones(500))
        arms[arm]=dict(per_system=systems,headline=min(v['accuracy'] for v in systems.values()),
            headline_cluster_95_ci=interval(np.minimum(boot[arm][SYSTEMS[0]],boot[arm][SYSTEMS[1]])),
            both_correct=int(joint.sum()),both_accuracy=float(joint.mean()),statuses=dict(Counter(r['status'] for r in rows)),
            responses=sum(r['model_calls'] for r in rows),cost_usd=sum(r['cost_usd'] for r in rows),
            median_latency_seconds=float(np.median([r['latency_seconds'] for r in rows])),
            p95_latency_seconds=float(np.quantile([r['latency_seconds'] for r in rows],.95)),
            paged_inputs=sum(bool(r['pages_read']) for r in rows))
    validator.close();a,b=ARMS;paired={}
    for s in SYSTEMS:
        paired[s]={}
        for bucket_name in ('all','unseen','rare','common'):
            mask=masks['all'] if bucket_name=='all' else masks[s+':'+bucket_name]
            old,new=vectors[a][s][mask],vectors[b][s][mask]
            key=s if bucket_name=='all' else s+':'+bucket_name
            paired[s][bucket_name]=dict(n=int(mask.sum()),delta_accuracy=float(new.mean()-old.mean()),
                corrections=int(np.logical_and(new,~old.astype(bool)).sum()),
                regressions=int(np.logical_and(old,~new.astype(bool)).sum()),
                cluster_bootstrap_95_ci=interval(boot[b][key]-boot[a][key]))
    paired['headline']=dict(delta_accuracy=arms[b]['headline']-arms[a]['headline'],cluster_bootstrap_95_ci=interval(
        np.minimum(boot[b][SYSTEMS[0]],boot[b][SYSTEMS[1]])-np.minimum(boot[a][SYSTEMS[0]],boot[a][SYSTEMS[1]])))
    domain_metrics={arm:{d:{s:dict(n=len(subset:=[c for c in cases if c['arm']==arm and c['system']==s and c['domain']==d]),
        correct=sum(c['correct'] for c in subset)) for s in SYSTEMS} for d in sorted({x['features']['domain'] for x in labels})} for arm in ARMS}
    features={arm:{f:{s:dict(n=len(subset:=[c for c,x in zip([r for r in cases if r['arm']==arm and r['system']==s],labels) if x['features'][f]]),
        correct=sum(c['correct'] for c in subset)) for s in SYSTEMS} for f in labels[0]['features'] if f!='domain'} for arm in ARMS}
    ledger=[read_json(p) for p in (EXPERIMENT/'usage_ledger').glob('*.json')]
    cost=sum(float(r['usage'].get('cost') or 0) for r in ledger)
    assert abs(cost-sum(r['cost_usd'] for rows in runs_by_arm.values() for r in rows))<1e-8
    generation_ids={u['generation_id'] for rows in runs_by_arm.values() for r in rows for u in r['usage_records']}
    assert generation_ids=={r['generation_id'] for r in ledger}
    result=dict(n=500,context_groups=len(group_names),selection=read_json(EXPERIMENT/'selection_manifest.json'),
        arms=arms,paired=paired,domain_metrics=domain_metrics,challenge_metrics=features,reported_cost_usd=cost,
        billed_responses=len(ledger),execution=read_json(EXPERIMENT/'execution.json'),
        uncertainty='5000 paired bootstrap resamples of entire normalized context groups, including all inputs in a selected group. Descriptive development intervals; no multiplicity adjustment.',
        limitations=['Unseen means absent from the permitted example store, not absent from model pretraining or dictionaries.',
                     'The sample contains 382 context groups; 55 inputs share contexts with earlier agent experiments.',
                     'Broader development evaluation; overlap with earlier retrieval-benchmark development is recorded.',
                     'Common-code SNOMED subset is small; report its denominator and interval.',
                     'CPU search arithmetic and hosted inference need not be bit-identical to previous experiments. Both arms freshly evaluated on the same provider.',
                     'The ICD variant retains its previously tested paging interface; this evaluates that configuration rather than a pure embedding-only intervention.'])
    write_json(EXPERIMENT/'results.json',result);write_json(EXPERIMENT/'case_diagnostics.json',cases)
    with (EXPERIMENT/'all_predictions.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    index={(c['arm'],c['input_id'],c['system']):c for c in cases};changes=[]
    for c in cases:
        if c['arm']==b:
            old=index[(a,c['input_id'],c['system'])]
            if c['correct']!=old['correct']:changes.append(dict(c,original_prediction=old['predicted'],change='correction' if c['correct'] else 'regression'))
    write_json(EXPERIMENT/'changed_predictions.json',changes)
    verify();print(json.dumps(result,indent=2))


if __name__=='__main__':analyze()
