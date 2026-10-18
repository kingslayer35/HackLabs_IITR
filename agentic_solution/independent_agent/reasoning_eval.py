"""Matched off/high reasoning pilot on frozen condition-B evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import OpenRouterClient
from .common import ARTIFACTS, SYSTEMS, read_json
from .comparisons import ReservedBudget
from .selector500_eval import (MODEL, PROVIDER, FIELDS, build_messages, response_format,
                               validate, sha, write, cluster_interval, verify as verify_selector)

SOURCE = ARTIFACTS / 'selector500_v1'
DATA = ARTIFACTS / 'diverse500_v1'
EXPERIMENT = ARTIFACTS / 'reasoning100_v1'
ARMS = ('off', 'high')
MAX_TOKENS = 8192
CAP = 1.0
WORKERS = 4


def select(metadata, n=100):
    """Outcome-blind selection: domains, distinct contexts/codes, frequency balance."""
    rows = sorted(metadata, key=lambda x: hashlib.sha256(('reasoning-pilot:13:' + x['input_id']).encode()).hexdigest())
    chosen = []; groups = set(); codes = {s: set() for s in SYSTEMS}
    bucket_counts = {s: Counter() for s in SYSTEMS}; domains = Counter()
    totals = {s: Counter(x['bucket'][s] for x in rows) for s in SYSTEMS}
    def add(x):
        chosen.append(x); groups.add(x['group_id']); domains[x['features']['domain']] += 1
        for s in SYSTEMS:
            codes[s].add(x['gold'][s]); bucket_counts[s][x['bucket'][s]] += 1
    # Cover every available coarse domain before filling proportional bucket deficits.
    for domain in sorted({x['features']['domain'] for x in rows}):
        eligible = [x for x in rows if x['features']['domain'] == domain and x['group_id'] not in groups]
        if eligible and len(chosen) < n:
            add(eligible[0])
    while len(chosen) < n:
        eligible = [x for x in rows if x['group_id'] not in groups]
        if not eligible:
            raise ValueError('Insufficient distinct context groups')
        def score(x):
            value = 1 / (1 + domains[x['features']['domain']])
            for s in SYSTEMS:
                b = x['bucket'][s]
                target = n * totals[s][b] / len(rows)
                value += 3 * max(0, target - bucket_counts[s][b]) / max(1, target)
                value += .75 * (x['gold'][s] not in codes[s])
            return value
        add(max(eligible, key=score))
    return chosen


def body(prepared, arm):
    if arm not in ARMS:
        raise ValueError('Unknown reasoning condition')
    return dict(model=MODEL, messages=build_messages(prepared, 'coding_guide'), response_format=response_format(prepared),
        temperature=0, seed=13, max_tokens=MAX_TOKENS,
        reasoning={'enabled': False} if arm == 'off' else {'effort': 'high'},
        usage={'include': True}, provider={'order': [PROVIDER], 'allow_fallbacks': False, 'require_parameters': True})


def reasoning_tokens(data):
    return data.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')


def prepare():
    if (EXPERIMENT / 'config.json').exists():
        verify(); return
    verify_selector()
    # Gold identity/frequency/domain may select diverse inputs; correctness and prior
    # predictions are never read. No label metadata enters prepared model evidence.
    selected = select(read_json(DATA / 'labels.json'))
    ids = [x['input_id'] for x in selected]
    manifest = dict(input_ids=ids, n=len(ids), context_groups=len({x['group_id'] for x in selected}),
        domains=dict(Counter(x['features']['domain'] for x in selected)),
        buckets={s: dict(Counter(x['bucket'][s] for x in selected)) for s in SYSTEMS},
        unique_codes={s: len({x['gold'][s] for x in selected}) for s in SYSTEMS},
        selection='Deterministic domain coverage, distinct contexts/code diversity and frequency balance. No prior predictions or correctness read.',
        status='Development pilot drawn from previously evaluated 500 inputs, not untouched confirmation.')
    write(EXPERIMENT / 'selection_manifest.json', manifest)
    prices = read_json(EXPERIMENT / 'provider_prices.json')
    endpoint = next(x for x in prices['data']['endpoints'] if x['provider_name'] == PROVIDER)
    assert float(endpoint['pricing']['prompt']) <= .00000013
    assert float(endpoint['pricing']['completion']) <= .00000028
    assert 'reasoning_effort' in endpoint['supported_parameters']
    assert 'high' in read_json(EXPERIMENT / 'model_metadata.json')['reasoning']['supported_efforts']
    protected = [SOURCE / 'evidence' / (iid + '.json') for iid in ids]
    protected += [DATA / 'labels.json', SOURCE / 'config.json', EXPERIMENT / 'provider_prices.json', EXPERIMENT / 'model_metadata.json', EXPERIMENT / 'selection_manifest.json']
    protected += list(Path(__file__).parent.glob('*.py'))
    config = dict(n=100, arms=ARMS, model=MODEL, provider=PROVIDER, max_output_tokens=MAX_TOKENS,
        temperature=0, seed=13, workers=WORKERS, cap_usd=CAP, responses_per_input_arm=1,
        condition='Identical frozen B evidence/guide/schema and 8192 output cap; only reasoning object differs.',
        extension_rule='Promising if headline improves by at least 2 percentage points, neither system loses overall accuracy, and invalid/truncated outputs do not increase. Any larger evaluation is separately frozen, reuses pilot responses, and remains development evidence.',
        hashes={str(p): sha(p) for p in protected})
    write(EXPERIMENT / 'config.json', config)
    verify(); print(json.dumps(manifest, indent=2), flush=True)


def verify():
    for name, expected in read_json(EXPERIMENT / 'config.json')['hashes'].items():
        assert sha(Path(name)) == expected, name


def run_one(iid, arm, budget):
    prepared = read_json(SOURCE / 'evidence' / (iid + '.json'))
    request = body(prepared, arm)
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    paid_path = EXPERIMENT / 'responses' / arm / (iid + '.json')
    if paid_path.exists():
        saved = read_json(paid_path)
        assert saved['request_sha256'] == fingerprint
    else:
        client = OpenRouterClient(MODEL, PROVIDER, budget)
        upper = (len(json.dumps(request, ensure_ascii=False).encode()) + 10000) * .00000013 + MAX_TOKENS * .00000028
        failure_path = EXPERIMENT / 'transport_failures' / arm / (iid + '.json')
        failures = read_json(failure_path) if failure_path.exists() else []
        if len(failures) >= 2:
            raise RuntimeError('Two transport failures; diagnosis needed')
        budget.reserve(upper)
        started = time.monotonic()
        try:
            r = requests.post(client.url, headers={'Authorization': 'Bearer ' + client.key, 'Content-Type': 'application/json'}, json=request, timeout=(15, 180))
            if r.status_code != 200:
                raise RuntimeError('HTTP ' + str(r.status_code) + ': ' + r.text.replace(client.key, '[REDACTED]')[:500])
            data = r.json()
            if data.get('error') or not data.get('id'):
                raise RuntimeError('Provider response error: ' + str(data.get('error')).replace(client.key, '[REDACTED]')[:500])
            saved = dict(input_id=iid, arm=arm, request_sha256=fingerprint, data=data, latency_seconds=time.monotonic() - started)
            write(paid_path, saved)
            cost = data.get('usage', {}).get('cost')
            budget.add(float(cost or 0)); budget.release(upper, upper if cost is None else 0)
        except Exception as exc:
            if not paid_path.exists():
                budget.release(upper, upper)
                failures.append(dict(error=str(exc).replace(client.key, '[REDACTED]')[:600], uncertain_cost_upper=upper))
                write(failure_path, failures)
            raise
    data = saved['data']
    assert data['provider'] == PROVIDER and data['model'] == MODEL
    assert data['usage'].get('cost') is not None
    choice = data['choices'][0]; content = choice['message'].get('content') or ''
    try:
        prediction = json.loads(content); error = validate(prediction, prepared)
    except (TypeError, ValueError):
        error = 'Invalid JSON output'
    if error:
        prediction = dict(icd10cm_code=prepared['fallback']['ICD10CM'], snomed_code=prepared['fallback']['SNOMEDCT_US'], confidence=0., evidence='Deterministic visible-code fallback after invalid terminal output.')
    record = dict(input_id=iid, arm=arm, prediction=prediction, status='fallback_invalid' if error else 'model', error=error,
        generation_id=data['id'], provider=data['provider'], model=data['model'], usage=data['usage'],
        reasoning_tokens=reasoning_tokens(data), finish_reason=choice.get('finish_reason'),
        request_sha256=fingerprint, latency_seconds=saved['latency_seconds'])
    write(EXPERIMENT / 'runs' / arm / (iid + '.json'), record)
    return record


def run(limit=None):
    verify(); budget = ReservedBudget(CAP)
    for f in (EXPERIMENT / 'responses').glob('*/*.json'):
        value = read_json(f)['data']['usage'].get('cost')
        if value is None:
            raise RuntimeError('Resolve missing paid usage before more requests')
        budget.add(float(value))
    for f in (EXPERIMENT / 'transport_failures').glob('*/*.json'):
        budget.uncertain_cost_bound += sum(x['uncertain_cost_upper'] for x in read_json(f))
    tasks = []
    for iid in read_json(EXPERIMENT / 'selection_manifest.json')['input_ids']:
        order = ARMS if int(iid[-1], 16) % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            if not (EXPERIMENT / 'runs' / arm / (iid + '.json')).exists(): tasks.append((iid, arm))
    if limit: tasks = tasks[:limit]
    print(f'Pending {len(tasks)}; prior cost ${budget.cost:.6f}; uncertain ${budget.uncertain_cost_bound:.6f}', flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(run_one, iid, arm, budget): (iid, arm) for iid, arm in tasks}
        for n, future in enumerate(as_completed(jobs), 1):
            try: future.result()
            except Exception as exc:
                error = dict(input_id=jobs[future][0], arm=jobs[future][1], error=str(exc)[:800]); errors.append(error)
                print('REQUEST ERROR ' + json.dumps(error), flush=True)
            if n % 10 == 0 or n == len(jobs):
                print(f'Completed {n}/{len(jobs)}; cost ${budget.cost:.6f}; errors {len(errors)}', flush=True)
    write(EXPERIMENT / 'execution.json', dict(errors=errors, reported_cost_usd=budget.cost, uncertain_cost_bound=budget.uncertain_cost_bound))
    if errors: raise RuntimeError('Unresolved request failures; retain paid successes and diagnose')


def summarize(results, labels):
    systems = {}; vectors = {}
    for s, field in FIELDS.items():
        correct = [r['prediction'][field] == x['gold'][s] for r, x in zip(results, labels)]
        vectors[s] = correct; buckets = {}
        for bucket in ['all', 'unseen', 'rare', 'common']:
            ids = [i for i, x in enumerate(labels) if bucket == 'all' or x['bucket'][s] == bucket]
            visible = [i for i in ids if labels[i]['gold'][s] in {c['code'] for c in read_json(SOURCE / 'evidence' / (labels[i]['input_id'] + '.json'))['payload']['candidates'][s]}]
            hits = sum(correct[i] for i in ids)
            buckets[bucket] = dict(n=len(ids), correct=hits, accuracy=hits / len(ids) if ids else None,
                gold_visible=len(visible), selection_accuracy_given_visible_gold=sum(correct[i] for i in visible) / len(visible) if visible else None)
        systems[s] = buckets
    token = [r['reasoning_tokens'] for r in results if r['reasoning_tokens'] is not None]
    return dict(per_system=systems, headline=min(systems[s]['all']['accuracy'] for s in SYSTEMS),
        both_correct=sum(all(vectors[s][i] for s in SYSTEMS) for i in range(len(labels))),
        statuses=dict(Counter(r['status'] for r in results)), finish_reasons=dict(Counter(r['finish_reason'] for r in results)),
        cost_usd=sum(float(r['usage']['cost']) for r in results),
        latency_median=statistics.median(r['latency_seconds'] for r in results), latency_p95=sorted(r['latency_seconds'] for r in results)[math.ceil(.95 * len(results))-1],
        completion_tokens=sum(r['usage']['completion_tokens'] for r in results),
        reasoning_tokens_reported=len(token), positive_reasoning_responses=sum(v > 0 for v in token),
        reasoning_tokens_total=sum(token), reasoning_tokens_median=statistics.median(token) if token else None), vectors


def analyze():
    verify(); ids = read_json(EXPERIMENT / 'selection_manifest.json')['input_ids']
    lookup = {x['input_id']: x for x in read_json(DATA / 'labels.json')}; labels = [lookup[iid] for iid in ids]
    summaries = {}; vectors = {}; generations = []
    for arm in ARMS:
        results = [read_json(EXPERIMENT / 'runs' / arm / (iid + '.json')) for iid in ids]
        assert len(results) == 100
        summaries[arm], vectors[arm] = summarize(results, labels)
        generations += [r['generation_id'] for r in results]
    assert len(generations) == len(set(generations)) == 200
    assert summaries['high']['positive_reasoning_responses'] == 100, 'Reasoning manipulation not verified for all high requests'
    assert summaries['off']['positive_reasoning_responses'] == 0
    paired = {}
    for s in SYSTEMS:
        paired[s] = {}
        for bucket in ['all', 'unseen', 'rare', 'common']:
            ii = [i for i, x in enumerate(labels) if bucket == 'all' or x['bucket'][s] == bucket]
            a = [vectors['off'][s][i] for i in ii]; b = [vectors['high'][s][i] for i in ii]
            gains = sum(y and not x for x, y in zip(a, b)); losses = sum(x and not y for x, y in zip(a, b))
            paired[s][bucket] = dict(n=len(ii), corrections=gains, regressions=losses, delta=(gains-losses)/len(ii) if ii else None,
                cluster_95_interval=cluster_interval(a, b, [labels[i]['group_id'] for i in ii]) if ii else None)
    import random
    rng = random.Random(13); samples = []
    assert len({x['group_id'] for x in labels}) == len(labels)
    for _ in range(5000):
        ii = [rng.randrange(100) for _ in labels]
        score = lambda arm: min(sum(vectors[arm][s][i] for i in ii)/100 for s in SYSTEMS)
        samples.append(score('high')-score('off'))
    samples.sort(); delta = summaries['high']['headline']-summaries['off']['headline']
    invalid = lambda a: 100-summaries[a]['statuses'].get('model', 0)
    truncated = lambda a: summaries[a]['finish_reasons'].get('length', 0)
    promising = delta >= .02-1e-12 and all(paired[s]['all']['delta'] >= 0 for s in SYSTEMS) and invalid('high') <= invalid('off') and truncated('high') <= truncated('off')
    result = dict(arms=summaries, paired=paired, headline_delta=delta, headline_cluster_95_interval=[samples[125], samples[4875]],
        total_cost_usd=sum(x['cost_usd'] for x in summaries.values()), unique_paid_responses=200, promising_under_frozen_rule=promising,
        status='100-input development pilot; paired complete-context resampling, 5000 draws, no multiplicity correction; no unseen-generalization claim from small strata')
    write(EXPERIMENT / 'results.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'verify', 'run', 'analyze']); parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'run': run(args.limit)
    else: globals()[args.command]()
