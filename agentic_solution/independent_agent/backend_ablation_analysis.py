"""Offline pre-decision exposure audit and paired scoring for backend ablations."""
import csv
import statistics
import random
from collections import Counter

from .backend_ablation import ARMS, EXPERIMENT, SOURCE, FIELDS, verify_sources
from .common import SYSTEMS, read_json, write_json
from .comparisons import paired_delta
from .tools import TerminologyTools


def visibility(run, evidence):
    visible = {s: set() for s in SYSTEMS}
    for s in SYSTEMS:
        visible[s].update(row['code'] for row in evidence['payload']['initial_candidates'][s]['results'])
        visible[s].update(row['code'] for row in evidence['payload']['training_examples'][s]['examples'])
        assert visible[s] == set(evidence['seen'][s])
    first = {s: {code: 'initial' for code in visible[s]} for s in SYSTEMS}
    decision = None
    for turn in run['trace']:
        before = {s: set(v) for s, v in visible.items()}
        if any(c['name'] == 'finish' and c['result'].get('accepted') for c in turn['tool_calls']):
            decision = before
            break
        if not turn['tool_calls'] and run['status'] == 'model' and turn is run['trace'][-1]:
            decision = before
            break
        for call in turn['tool_calls']:
            out = call['result']
            system = out.get('system')
            if system in SYSTEMS:
                for row in out.get('results', out.get('concepts', [])):
                    if 'code' in row and 'error' not in row:
                        first[system].setdefault(row['code'], 'review_page' if out.get('review_required') else call['name'])
                        visible[system].add(row['code'])
            for row in out.get('examples', []):
                first[row['system']].setdefault(row['code'], 'training_example')
                visible[row['system']].add(row['code'])
    if decision is None:
        decision = before
    assert all(decision[s] == set(run['seen_candidates'][s]) for s in SYSTEMS), run['input_id']
    return decision, first


def score():
    verify_sources()
    labels = read_json(SOURCE / 'labels.json')
    results, cases, vectors, runs = {}, [], {}, {}
    validator = TerminologyTools()
    for arm in ('baseline', 'dense_paging') + ARMS:
        counts = {s: Counter() for s in SYSTEMS}
        vectors[arm] = {s: [] for s in SYSTEMS}
        vectors[arm]['both'] = []
        runs[arm] = []
        statuses, providers, discovery = Counter(), Counter(), {s: Counter() for s in SYSTEMS}
        for x in labels:
            name = x['input_id'] + '.json'
            if arm in ARMS:
                run = read_json(EXPERIMENT / arm / 'runs' / name)
                evidence = read_json(EXPERIMENT / arm / 'initial_evidence' / name)
            else:
                run = read_json(SOURCE / 'runs' / arm / name)
                evidence = read_json(SOURCE / ('baseline_initial_evidence' if arm == 'baseline' else 'initial_evidence') / name)
            if run['status'] == 'fallback_error':
                raise RuntimeError('Unresolved infrastructure failure: ' + arm + ':' + x['input_id'])
            visible, first = visibility(run, evidence)
            statuses[run['status']] += 1
            providers[run['provider_requested']] += 1
            hits = []
            for s in SYSTEMS:
                gold, predicted = x['gold'][s], run['prediction'][FIELDS[s]]
                assert validator.valid(s, predicted)
                hit = predicted == gold
                hits.append(hit)
                vectors[arm][s].append(hit)
                counts[s]['correct'] += hit
                counts[s]['initial12'] += gold in {r['code'] for r in evidence['payload']['initial_candidates'][s]['results']}
                counts[s]['initial_evidence'] += gold in evidence['seen'][s]
                counts[s]['ever_seen'] += gold in visible[s]
                counts[s]['selection_errors'] += not hit and gold in visible[s]
                counts[s]['retrieval_misses'] += not hit and gold not in visible[s]
                counts[s]['valid'] += 1
                discovery[s][first[s].get(gold, 'never_seen')] += 1
                cases.append(dict(arm=arm, input_id=x['input_id'], entity=x['entity'], system=s,
                                  gold=gold, predicted=predicted, correct=hit, ever_seen=gold in visible[s],
                                  first_gold_source=first[s].get(gold), provider=run['provider_requested'],
                                  status=run['status'], evidence=run['prediction']['evidence']))
            vectors[arm]['both'].append(all(hits))
            runs[arm].append(run)
        results[arm] = dict(per_system={s: dict(v, accuracy=v['correct'] / 80) for s, v in counts.items()},
                            headline=min(v['correct'] / 80 for v in counts.values()),
                            both_correct=sum(vectors[arm]['both']), statuses=dict(statuses), providers=dict(providers),
                            first_gold_source={s: dict(v) for s, v in discovery.items()},
                            model_responses=sum(r['model_calls'] for r in runs[arm]),
                            recorded_cost=sum(r['cost_usd'] for r in runs[arm]),
                            median_latency_seconds=statistics.median(r['latency_seconds'] for r in runs[arm]))
    validator.close()
    paired = {}
    for arm in ARMS:
        paired[arm] = {}
        for control in ('baseline', 'dense_paging'):
            paired[arm][control] = {s: paired_delta(vectors[arm][s], vectors[control][s]) for s in SYSTEMS + ('both',)}
            for s in SYSTEMS + ('both',):
                delta = (sum(vectors[arm][s]) - sum(vectors[control][s])) / len(labels)
                assert abs(paired[arm][control][s]['delta_accuracy'] - delta) < 1e-12
            rng = random.Random(13)
            samples = []
            for _ in range(10000):
                ids = [rng.randrange(len(labels)) for _ in labels]
                newer = min(sum(vectors[arm][s][i] for i in ids) / len(ids) for s in SYSTEMS)
                older = min(sum(vectors[control][s][i] for i in ids) / len(ids) for s in SYSTEMS)
                samples.append(newer - older)
            samples.sort()
            paired[arm][control]['headline'] = dict(delta_accuracy=results[arm]['headline'] - results[control]['headline'],
                paired_bootstrap_95_ci=[samples[250], samples[9749]])
        strata = {}
        for provider in ('DeepInfra', 'CoreWeave'):
            indices = [i for i, r in enumerate(runs['dense_paging']) if r['provider_requested'] == provider]
            assert all(runs[arm][i]['provider_requested'] == provider for i in indices)
            strata[provider] = dict(n=len(indices), **{s: dict(
                cached_dense_correct=sum(vectors['dense_paging'][s][i] for i in indices),
                new_correct=sum(vectors[arm][s][i] for i in indices)) for s in SYSTEMS})
        paired[arm]['provider_strata_vs_dense'] = strata
    review = []
    for i, (x, run) in enumerate(zip(labels, runs['selective_snomed'])):
        triggered = [event for event in run['review_events'] if event['triggered']]
        for event in triggered:
            g = x['gold']['SNOMEDCT_US']
            before = set(read_json(EXPERIMENT / 'selective_snomed/initial_evidence' / (x['input_id'] + '.json'))['seen']['SNOMEDCT_US'])
            # Use the first-source audit, rather than merely initial exposure, to establish new recovery.
            diagnostic = next(c for c in cases if c['arm'] == 'selective_snomed' and c['input_id'] == x['input_id'] and c['system'] == 'SNOMEDCT_US')
            review.append(dict(input_id=x['input_id'], entity=x['entity'], turn=event['turn'],
                               confidence=event['proposed']['confidence'], offset=event['offset'],
                               proposed_code=event['proposed']['snomed_code'], final_code=run['prediction']['snomed_code'],
                               gold_in_page=g in event['codes'], gold_newly_exposed=diagnostic['first_gold_source'] == 'review_page',
                               proposed_correct=event['proposed']['snomed_code'] == g,
                               final_correct=vectors['selective_snomed']['SNOMEDCT_US'][i],
                               cached_dense_correct=vectors['dense_paging']['SNOMEDCT_US'][i]))
    ledger = [read_json(p) for p in (EXPERIMENT / 'usage_ledger').glob('*.json')]
    result = dict(n=80, arms=results, paired=paired, review=dict(activated_inputs=len(review),
                  newly_exposed_gold=sum(r['gold_newly_exposed'] for r in review),
                  newly_exposed_gold_selected=sum(r['gold_newly_exposed'] and r['final_correct'] for r in review),
                  proposed_correct=sum(r['proposed_correct'] for r in review),
                  final_correct=sum(r['final_correct'] for r in review),
                  proposed_to_final_corrections=sum(not r['proposed_correct'] and r['final_correct'] for r in review),
                  proposed_to_final_regressions=sum(r['proposed_correct'] and not r['final_correct'] for r in review),
                  late_review_skips=sum(sum(not e['triggered'] for e in r['review_events']) for r in runs['selective_snomed'])),
                  reported_new_cost_usd=sum(float(r['usage'].get('cost') or 0) for r in ledger),
                  unique_billed_responses=len(ledger), execution=read_json(EXPERIMENT / 'execution.json'),
                  limits=['Repeatedly inspected development sample; no generalization claim.',
                          'Provider matched per input to dense control; cached lexical baseline all DeepInfra.',
                          'Cached controls were run earlier; hosted inference varies despite seed and temperature0.',
                          'ICD-only arm retains dense ICD paging, changes SNOMED backend and tool description.',
                          'Selective policy adds prompt plus deterministic review gate based on uncalibrated joint confidence.',
                          'Proposed-to-final review changes are descriptive, not randomized causal effects.'])
    assert abs(result['reported_new_cost_usd'] - sum(results[a]['recorded_cost'] for a in ARMS)) < 1e-8
    write_json(EXPERIMENT / 'results.json', result)
    write_json(EXPERIMENT / 'review_cases.json', review)
    write_json(EXPERIMENT / 'case_diagnostics.json', cases)
    with (EXPERIMENT / 'all_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    changes = []
    index = {(c['arm'], c['input_id'], c['system']): c for c in cases}
    for c in cases:
        if c['arm'] in ARMS:
            old = index[('dense_paging', c['input_id'], c['system'])]
            if c['correct'] != old['correct']:
                changes.append(dict(c, cached_prediction=old['predicted'],
                                    change='correction' if c['correct'] else 'regression'))
    write_json(EXPERIMENT / 'changed_predictions.json', changes)
    verify_sources()
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    score()
