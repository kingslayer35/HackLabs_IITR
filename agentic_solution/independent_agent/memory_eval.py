"""Matched training-memory ablations on fixed candidate pools; no new retrieval IDs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import OpenRouterClient
from .common import ARTIFACTS, SYSTEMS, read_json
from .comparisons import ReservedBudget
from .context_memory import (ContextMemory, build_index, MODES, MAX_EXAMPLES,
                              MAX_SYSTEM_BYTES, local_window, literal_cues)
from .selector500_eval import (MODEL, PROVIDER, FIELDS, build_messages, response_format,
                               validate, sha, write, cluster_interval, verify as verify_selector)
from .tools import query_terms, fts_query

SOURCE = ARTIFACTS / 'selector500_v1'
DATA = ARTIFACTS / 'diverse500_v1'
EXPERIMENT = ARTIFACTS / 'memory_context_v1'
MAX_TOKENS = 2048
CAP = 2.0
WORKERS = 4
MEMORY_NOTE = '''Training-memory annotations are literal occurrences, not formal code requirements or
proof that a modifier belongs to the target. A description cue absent literally from an example
does not disprove its annotation or exclude synonyms/defaults. The example's code is an observed
training label. Similar and different-code examples can express competing interpretations;
compare their target meaning and context rather than voting by example frequency.'''


def body(prepared, mode):
    if mode not in MODES: raise ValueError('Unknown memory mode')
    value = copy.deepcopy(prepared['base'])
    value['payload']['training_examples'] = prepared['examples'][mode]
    messages = build_messages(value, 'coding_guide')
    messages[0]['content'] += '\n\n' + MEMORY_NOTE
    return dict(model=MODEL, messages=messages, response_format=response_format(value), temperature=0, seed=13,
        max_tokens=MAX_TOKENS, reasoning={'enabled': False}, usage={'include': True},
        provider={'order': [PROVIDER], 'allow_fallbacks': False, 'require_parameters': True})


def prepare():
    if (EXPERIMENT / 'prepared.json').exists():
        for path, expected in read_json(EXPERIMENT / 'prepared.json')['hashes'].items(): assert sha(Path(path)) == expected
        return
    verify_selector(); EXPERIMENT.mkdir(exist_ok=True)
    index = EXPERIMENT / 'memory.sqlite'
    count = build_index(DATA / 'examples.sqlite', index, ARTIFACTS / 'terminology.sqlite')
    assert count == 3423
    memory = ContextMemory(index)
    source_db = sqlite3.connect(f'file:{DATA}/examples.sqlite?mode=ro', uri=True)
    source_db.row_factory = sqlite3.Row
    training = {(r['system'], r['code'], r['input_id']): dict(r) for r in source_db.execute('select * from examples')}
    ids = [x['input_id'] for x in read_json(DATA / 'inputs.json')]
    pilot_ids = read_json(ARTIFACTS / 'reasoning100_v1/selection_manifest.json')['input_ids']
    assert len(ids) == 500 and len(pilot_ids) == 100 and set(pilot_ids) <= set(ids)
    stats = {m: {s: Counter() for s in SYSTEMS} for m in MODES}; examples_for_review = []
    baseline_checks = 0
    for n, iid in enumerate(ids, 1):
        original = read_json(SOURCE / 'evidence' / (iid + '.json'))
        base = dict(input_id=iid, payload=copy.deepcopy(original['payload']), fallback=original['fallback'], distinctions={})
        base['payload']['training_examples'] = {s: [] for s in SYSTEMS}
        examples = {m: {} for m in MODES}; diagnostics = {m: {} for m in MODES}
        for system in SYSTEMS:
            codes = {c['code'] for c in base['payload']['candidates'][system]}
            retrieved = {m: memory.retrieve(base['payload']['mention'], base['payload']['context'], system, codes, m) for m in MODES}
            # The control uses exactly the old full-context FTS scoring and mention query,
            # with the same visible-code restriction and common new display/budget rules.
            terms = query_terms(base['payload']['mention'])
            if terms:
                placeholders = ','.join('?' for _ in codes)
                rows = source_db.execute(f'''select e.id,bm25(example_fts,5.0,1.0) as relevance from example_fts
                  join examples e on e.id=example_fts.rowid where example_fts match ? and e.system=?
                  and e.code in ({placeholders}) order by relevance,e.id limit 100''', [fts_query(terms), system, *sorted(codes)]).fetchall()
                assert retrieved['mention']['ranked_row_ids'] == [r['id'] for r in rows]
            else: assert not retrieved['mention']['ranked_row_ids']
            baseline_checks += 1
            for mode in MODES:
                info = retrieved[mode]; selected = info['examples']
                examples[mode][system] = selected
                diagnostics[mode][system] = {k: v for k, v in info.items() if k != 'examples'}
                stat = stats[mode][system]; stat['inputs'] += 1; stat['examples'] += len(selected)
                stat['empty_inputs'] += not selected
                stat['multiple_labels'] += len({e['code'] for e in selected}) > 1
                stat['serialized_bytes'] += info['serialized_bytes']
                stat['differs_from_mention_control'] += [e['example_id'] for e in selected] != [e['example_id'] for e in retrieved['mention']['examples']]
                assert len(selected) <= MAX_EXAMPLES and info['serialized_bytes'] <= MAX_SYSTEM_BYTES
                for e in selected:
                    assert e['code'] in codes
                    row = training[(system, e['code'], e['example_id'])]
                    window = local_window(row['entity'], row['context'])
                    assert e['mention'] == row['entity'] and e['context'] == window['text']
                    for cue in e['literal_cues']:
                        assert e[cue['field']][cue['start']:cue['end']] == cue['text']
        prepared = dict(input_id=iid, base=base, examples=examples, diagnostics=diagnostics)
        write(EXPERIMENT / 'evidence' / (iid + '.json'), prepared)
        # Hash-selected review set plus a previously discussed specificity example;
        # no evaluation gold or selector prediction is read for the retrieval audit.
        if iid in sorted(pilot_ids, key=lambda x: hashlib.sha256(('memory-audit:'+x).encode()).hexdigest())[:8] or iid == 'a0f4da0e49d1903f35f8':
            examples_for_review.append(dict(input_id=iid, mention=base['payload']['mention'],
                target_local_context=local_window(base['payload']['mention'], base['payload']['context']), examples=examples))
        if n % 100 == 0: print(f'Prepared and audited memory for {n}/500 inputs', flush=True)
    source_db.close(); memory.close()
    write(EXPERIMENT / 'retrieval_audit.json', dict(training_rows=count, inputs=500, baseline_rank_checks=baseline_checks,
        stats={m: {s: dict(v) for s, v in systems.items()} for m, systems in stats.items()},
        verified='All training provenance, source snippets/cue offsets, candidate-label restrictions, example/byte caps and mention-baseline FTS rank identity passed.',
        caveat='Mechanical retrieval audit, not clinical adjudication. Context cues are lexical occurrences. No gold or predictions read.'))
    write(EXPERIMENT / 'retrieval_review.json', examples_for_review)
    paths = [DATA / 'inputs.json', DATA / 'examples.sqlite', ARTIFACTS / 'terminology.sqlite', index,
             ARTIFACTS / 'reasoning100_v1/selection_manifest.json', EXPERIMENT / 'retrieval_audit.json', EXPERIMENT / 'retrieval_review.json']
    paths += list((EXPERIMENT / 'evidence').glob('*.json'))
    paths += list((SOURCE / 'evidence').glob('*.json'))
    write(EXPERIMENT / 'prepared.json', dict(input_ids=ids, pilot_input_ids=pilot_ids, hashes={str(p): sha(p) for p in paths}))


def freeze():
    if (EXPERIMENT / 'config.json').exists(): verify(); return
    preparation = read_json(EXPERIMENT / 'prepared.json')
    for path, expected in preparation['hashes'].items(): assert sha(Path(path)) == expected
    endpoint = next(e for e in read_json(EXPERIMENT / 'provider_prices.json')['data']['endpoints'] if e['provider_name'] == PROVIDER)
    assert float(endpoint['pricing']['prompt']) <= .00000013 and float(endpoint['pricing']['completion']) <= .00000028
    paths = list(Path(__file__).parent.glob('*.py')) + [EXPERIMENT / 'prepared.json', EXPERIMENT / 'provider_prices.json', DATA / 'labels.json']
    if (EXPERIMENT / 'retrieval_review_notes.json').exists(): paths.append(EXPERIMENT / 'retrieval_review_notes.json')
    write(EXPERIMENT / 'config.json', dict(n_pilot=100, n_available=500, modes=MODES, model=MODEL, provider=PROVIDER,
        max_tokens=MAX_TOKENS, reasoning=False, seed=13, temperature=0, cap_usd=CAP, workers=WORKERS,
        examples_per_system_max=MAX_EXAMPLES, bytes_per_system_max=MAX_SYSTEM_BYTES, memory_note=MEMORY_NOTE,
        extension_rule='A variant is promising if pilot headline gains at least 2 pp over mention control, neither system loses overall accuracy, neither unseen bucket loses more than one correct input, and invalid outputs do not increase. If any variant passes, extend all three frozen conditions to all500, retaining pilot responses. This remains adaptive development evidence.',
        scope='Fixed-evidence candidate selection. All arms have same new example display and bounds; only example selection differs. New matched control is not previous B or the full rescue agent.',
        hashes={**preparation['hashes'], **{str(p): sha(p) for p in paths}}))
    verify(); print('FROZEN: three memory conditions, 100-input pilot, optional frozen all500 expansion, shared $2 cap', flush=True)


def verify():
    for path, expected in read_json(EXPERIMENT / 'config.json')['hashes'].items(): assert sha(Path(path)) == expected, path


def run_one(iid, mode, budget):
    prepared = read_json(EXPERIMENT / 'evidence' / (iid + '.json')); request = body(prepared, mode)
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    paid_path = EXPERIMENT / 'responses' / mode / (iid + '.json')
    if paid_path.exists():
        saved = read_json(paid_path); assert saved['request_sha256'] == fingerprint
    else:
        client = OpenRouterClient(MODEL, PROVIDER, budget)
        upper = (len(json.dumps(request, ensure_ascii=False).encode())+10000)*.00000013 + MAX_TOKENS*.00000028
        failures_path = EXPERIMENT / 'transport_failures' / mode / (iid + '.json')
        failures = read_json(failures_path) if failures_path.exists() else []
        if len(failures) >= 2: raise RuntimeError('Two failed attempts; diagnose before another request')
        budget.reserve(upper); started = time.monotonic()
        try:
            response = requests.post(client.url, headers={'Authorization':'Bearer '+client.key,'Content-Type':'application/json'}, json=request, timeout=(15,120))
            if response.status_code != 200:
                raise RuntimeError('HTTP '+str(response.status_code)+': '+response.text.replace(client.key,'[REDACTED]')[:500])
            raw = response.json()
            if raw.get('error') or not raw.get('id'): raise RuntimeError('Provider error: '+str(raw.get('error')).replace(client.key,'[REDACTED]')[:500])
            saved = dict(input_id=iid, mode=mode, request_sha256=fingerprint, data=raw, latency_seconds=time.monotonic()-started)
            write(paid_path, saved)
            cost = raw.get('usage',{}).get('cost'); budget.add(float(cost or 0)); budget.release(upper,upper if cost is None else 0)
            if cost is None: raise RuntimeError('Missing usage cost; retain paid response for diagnosis')
        except Exception as exc:
            if not paid_path.exists():
                budget.release(upper,upper); failures.append(dict(error=str(exc).replace(client.key,'[REDACTED]')[:600],uncertain_cost_upper=upper)); write(failures_path,failures)
            raise
    raw = saved['data']; assert raw['provider']==PROVIDER and raw['model']==MODEL
    assert raw['usage'].get('cost') is not None
    choice = raw['choices'][0]
    try:
        prediction = json.loads(choice['message'].get('content') or ''); error = validate(prediction,prepared['base'])
    except (ValueError,TypeError): error='Invalid JSON output'
    if error:
        prediction=dict(icd10cm_code=prepared['base']['fallback']['ICD10CM'],snomed_code=prepared['base']['fallback']['SNOMEDCT_US'],confidence=0.,evidence='Visible-code fallback after invalid terminal output.')
    record=dict(input_id=iid,mode=mode,prediction=prediction,status='fallback_invalid' if error else 'model',error=error,
        generation_id=raw['id'],provider=raw['provider'],model=raw['model'],usage=raw['usage'],finish_reason=choice.get('finish_reason'),
        request_sha256=fingerprint,latency_seconds=saved['latency_seconds'])
    write(EXPERIMENT/'runs'/mode/(iid+'.json'),record);return record


def run(phase):
    verify(); plan=read_json(EXPERIMENT/'prepared.json'); budget=ReservedBudget(CAP)
    for path in (EXPERIMENT/'responses').glob('*/*.json'):
        cost=read_json(path)['data']['usage'].get('cost')
        if cost is None:raise RuntimeError('Resolve missing usage before continuing')
        budget.add(float(cost))
    for path in (EXPERIMENT/'transport_failures').glob('*/*.json'):
        budget.uncertain_cost_bound+=sum(x['uncertain_cost_upper'] for x in read_json(path))
    ids=plan['pilot_input_ids'] if phase=='pilot' else plan['input_ids']
    if phase=='all500':assert read_json(EXPERIMENT/'results_pilot.json')['promising_variants'],'Pilot expansion rule not met'
    tasks=[]
    for iid in ids:
        order=sorted(MODES,key=lambda m:hashlib.sha256((iid+':'+m).encode()).hexdigest())
        tasks.extend((iid,m) for m in order if not (EXPERIMENT/'runs'/m/(iid+'.json')).exists())
    errors=[];started=time.monotonic()
    print(f'Pending {len(tasks)} {phase} selections; retained paid cost ${budget.cost:.6f}',flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs={pool.submit(run_one,iid,mode,budget):(iid,mode) for iid,mode in tasks}
        for n,future in enumerate(as_completed(jobs),1):
            try:future.result()
            except Exception as exc:
                error=dict(input_id=jobs[future][0],mode=jobs[future][1],error=str(exc)[:800]);errors.append(error);print('REQUEST ERROR '+json.dumps(error),flush=True)
            if n%25==0 or n==len(jobs):print(f'Completed {n}/{len(jobs)}; cost ${budget.cost:.6f}; errors {len(errors)}',flush=True)
    write(EXPERIMENT/('execution_'+phase+'.json'),dict(errors=errors,reported_cost_usd=budget.cost,uncertain_cost_bound=budget.uncertain_cost_bound,wall_seconds=time.monotonic()-started))
    if errors:raise RuntimeError('Unresolved request failures; retain successful responses and diagnose')


def analyze(phase):
    verify(); plan=read_json(EXPERIMENT/'prepared.json');ids=plan['pilot_input_ids'] if phase=='pilot' else plan['input_ids']
    by_id={x['input_id']:x for x in read_json(DATA/'labels.json')};labels=[by_id[iid] for iid in ids]; n=len(ids)
    vectors={};summary={};generations=[]
    for mode in MODES:
        rr=[read_json(EXPERIMENT/'runs'/mode/(iid+'.json')) for iid in ids]
        vectors[mode]={s:[r['prediction'][f]==x['gold'][s] for r,x in zip(rr,labels)] for s,f in FIELDS.items()}
        systems={}
        for s in SYSTEMS:
            systems[s]={}
            for bucket in ['all','unseen','rare','common']:
                ii=[i for i,x in enumerate(labels) if bucket=='all' or x['bucket'][s]==bucket]
                systems[s][bucket]=dict(n=len(ii),correct=sum(vectors[mode][s][i] for i in ii),accuracy=sum(vectors[mode][s][i] for i in ii)/len(ii))
        summary[mode]=dict(per_system=systems,headline=min(systems[s]['all']['accuracy'] for s in SYSTEMS),
            both_correct=sum(all(vectors[mode][s][i] for s in SYSTEMS) for i in range(n)),statuses=dict(Counter(r['status'] for r in rr)),
            cost_usd=sum(float(r['usage']['cost']) for r in rr),latency_median=statistics.median(r['latency_seconds'] for r in rr),
            latency_p95=sorted(r['latency_seconds'] for r in rr)[math.ceil(.95*n)-1])
        assert all(r['usage']['completion_tokens_details']['reasoning_tokens']==0 for r in rr)
        generations += [r['generation_id'] for r in rr]
    assert len(generations)==len(set(generations))==n*3
    paired={}
    for old,new in [('mention','context'),('context','contrast'),('mention','contrast')]:
        key=new+'_vs_'+old;paired[key]={}
        for s in SYSTEMS:
            paired[key][s]={}
            for bucket in ['all','unseen','rare','common']:
                ii=[i for i,x in enumerate(labels) if bucket=='all' or x['bucket'][s]==bucket]
                a=[vectors[old][s][i] for i in ii];b=[vectors[new][s][i] for i in ii]
                gains=sum(y and not x for x,y in zip(a,b));losses=sum(x and not y for x,y in zip(a,b))
                paired[key][s][bucket]=dict(n=len(ii),corrections=gains,regressions=losses,delta=(gains-losses)/len(ii),cluster_95_interval=cluster_interval(a,b,[labels[i]['group_id'] for i in ii]))
        grouped=defaultdict(list)
        for i,x in enumerate(labels):grouped[x['group_id']].append(i)
        groups=list(grouped.values());rng=random.Random(13);boot=[]
        for _ in range(5000):
            ii=[i for _ in groups for i in groups[rng.randrange(len(groups))]]
            score=lambda mode:min(sum(vectors[mode][s][i] for i in ii)/len(ii) for s in SYSTEMS)
            boot.append(score(new)-score(old))
        boot.sort();paired[key]['headline']=dict(delta=summary[new]['headline']-summary[old]['headline'],cluster_95_interval=[boot[125],boot[4875]])
    promising=[]
    if phase=='pilot':
        for mode in ['context','contrast']:
            if summary[mode]['headline']-summary['mention']['headline']>=.02-1e-12 and all(summary[mode]['per_system'][s]['all']['correct']>=summary['mention']['per_system'][s]['all']['correct'] for s in SYSTEMS) and all(summary[mode]['per_system'][s]['unseen']['correct']>=summary['mention']['per_system'][s]['unseen']['correct']-1 for s in SYSTEMS) and summary[mode]['statuses'].get('model',0)>=summary['mention']['statuses'].get('model',0):promising.append(mode)
    result=dict(n=n,phase=phase,arms=summary,paired=paired,promising_variants=promising,unique_paid_responses=n*3,
        reported_cost_usd=sum(x['cost_usd'] for x in summary.values()),
        interpretation='Matched fixed-pool development selection experiment. All memory labels restricted to candidates. New example format/budgets in all arms; not a comparison with earlier B or complete rescue agent. Multiple descriptive paired context intervals, no multiplicity adjustment.')
    write(EXPERIMENT/('results_'+phase+'.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','freeze','verify','run','analyze']);parser.add_argument('--phase',choices=['pilot','all500'],default='pilot')
    args=parser.parse_args()
    if args.command in {'run','analyze'}:globals()[args.command](args.phase)
    else:globals()[args.command]()
