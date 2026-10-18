"""Frozen 500-prefix controlled replay with fresh fallback-only continuations."""
from __future__ import annotations

import argparse
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
from .diverse500_eval import verify as verify_source
from .selector500_eval import sha, write, FIELDS, MODEL, PROVIDER, cluster_interval
from .terminal_rescue import (MAX_TOKENS, needs_rescue, verify_prefix, prepare_terminal,
                              terminal_request, finalize_after_budget)

SOURCE = ARTIFACTS / 'diverse500_v1'
SOURCE_ARM = 'sapbert_icd_text_snomed'
EXPERIMENT = ARTIFACTS / 'terminal_rescue500_v1'
CAP = .25
WORKERS = 4


def source_run(iid):
    return read_json(SOURCE / SOURCE_ARM / 'runs' / (iid + '.json'))


def prepare():
    if (EXPERIMENT / 'config.json').exists():
        verify(); return
    verify_source()
    inputs = read_json(SOURCE / 'inputs.json'); assert len(inputs) == 500
    db = sqlite3.connect(f'file:{ARTIFACTS}/terminology.sqlite?mode=ro', uri=True); db.row_factory = sqlite3.Row
    paths = [SOURCE / 'inputs.json', SOURCE / 'labels.json', SOURCE / 'config.json', ARTIFACTS / 'terminology.sqlite', SOURCE / 'examples.sqlite']
    eligible = []; calls = 0
    for item in inputs:
        iid = item['input_id']; assert set(item) == {'input_id', 'entity', 'chunk_text'}
        run = source_run(iid); calls += verify_prefix(run)
        rp = SOURCE / SOURCE_ARM / 'runs' / (iid + '.json')
        ep = SOURCE / SOURCE_ARM / 'initial_evidence' / (iid + '.json')
        paths.extend([rp, ep])
        if needs_rescue(run):
            prepared = prepare_terminal(run, read_json(ep), db)
            # Confirm reconstruction, without reading previous selector predictions.
            original = read_json(ARTIFACTS / 'selector500_v1/evidence' / (iid + '.json'))
            assert prepared['payload'] == original['payload']
            write(EXPERIMENT / 'evidence' / (iid + '.json'), prepared)
            paths.append(EXPERIMENT / 'evidence' / (iid + '.json'))
            eligible.append(iid)
    db.close(); assert len(eligible) == 35
    endpoint = next(e for e in read_json(EXPERIMENT / 'provider_prices.json')['data']['endpoints'] if e['provider_name'] == PROVIDER)
    assert float(endpoint['pricing']['prompt']) <= .00000013 and float(endpoint['pricing']['completion']) <= .00000028
    assert {'response_format', 'structured_outputs'} <= set(endpoint['supported_parameters'])
    paths += [EXPERIMENT / 'provider_prices.json'] + list(Path(__file__).parent.glob('*.py'))
    write(EXPERIMENT / 'config.json', dict(n=500, eligible_input_ids=eligible, eligible_count=35,
        retained_accepted=465, model=MODEL, provider=PROVIDER, reasoning=False, seed=13, temperature=0,
        max_terminal_output_tokens=MAX_TOKENS, extra_responses_per_failed_input=1,
        max_source_responses=5, max_combined_responses=6, cap_usd=CAP, workers=WORKERS,
        source_arm=SOURCE_ARM, source_paid_prefixes_replayed=calls,
        intervention='After observable five-response budget/invalid fallback only, make one fresh condition-B terminal request; preserve every accepted source decision. No new retrieval.',
        interpretation='Controlled replay of previously evaluated development prefixes, with fresh terminal calls. Not fresh full-agent inference or independent confirmation.',
        labels_runtime_policy='Only input IDs, mentions, source traces/evidence, dictionary and training examples used before scoring. Labels hashed for audit but never loaded in prepare/run.',
        hashes={str(p): sha(p) for p in paths}))
    verify(); print('FROZEN: 500 verified prefixes, 465 retained finishes, 35 fresh terminal continuations, $0.25 cap', flush=True)


def verify():
    for path, expected in read_json(EXPERIMENT / 'config.json')['hashes'].items():
        assert sha(Path(path)) == expected, path


def request_fingerprint(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_one(iid, budget):
    original = source_run(iid)
    if not needs_rescue(original):
        result = finalize_after_budget(original, None, lambda _: (_ for _ in ()).throw(AssertionError('Accepted finish must not call model')))
        result['terminal_latency_seconds'] = 0.
    else:
        prepared = read_json(EXPERIMENT / 'evidence' / (iid + '.json'))
        paid_path = EXPERIMENT / 'responses' / (iid + '.json')
        def complete(body):
            fingerprint = request_fingerprint(body)
            if paid_path.exists():
                saved = read_json(paid_path); assert saved['request_sha256'] == fingerprint
                return saved['data']
            client = OpenRouterClient(MODEL, PROVIDER, budget)
            upper = (len(json.dumps(body, ensure_ascii=False).encode()) + 10000) * .00000013 + MAX_TOKENS * .00000028
            failure_path = EXPERIMENT / 'transport_failures' / (iid + '.json')
            failures = read_json(failure_path) if failure_path.exists() else []
            if len(failures) >= 2: raise RuntimeError('Two failed requests; diagnose before another attempt')
            budget.reserve(upper); started = time.monotonic()
            try:
                r = requests.post(client.url, headers={'Authorization': 'Bearer ' + client.key, 'Content-Type': 'application/json'}, json=body, timeout=(15, 120))
                if r.status_code != 200:
                    raise RuntimeError('HTTP ' + str(r.status_code) + ': ' + r.text.replace(client.key, '[REDACTED]')[:500])
                data = r.json()
                if data.get('error') or not data.get('id'):
                    raise RuntimeError('Provider error: ' + str(data.get('error')).replace(client.key, '[REDACTED]')[:500])
                write(paid_path, dict(input_id=iid, request_sha256=fingerprint, data=data, latency_seconds=time.monotonic()-started))
                cost = data.get('usage', {}).get('cost'); budget.add(float(cost or 0)); budget.release(upper, upper if cost is None else 0)
                if cost is None: raise RuntimeError('Missing billed cost; retain paid response for diagnosis')
                return data
            except Exception as exc:
                if not paid_path.exists():
                    budget.release(upper, upper)
                    failures.append(dict(error=str(exc).replace(client.key, '[REDACTED]')[:600], uncertain_cost_upper=upper))
                    write(failure_path, failures)
                raise
        result = finalize_after_budget(original, prepared, complete)
        saved = read_json(paid_path)
        assert saved['data']['usage'].get('cost') is not None
        result['terminal_latency_seconds'] = saved['latency_seconds']
        result['request_sha256'] = saved['request_sha256']
    result['prefix_replayed'] = True
    result['source_prediction_sha256'] = hashlib.sha256(json.dumps(original['prediction'], sort_keys=True).encode()).hexdigest()
    write(EXPERIMENT / 'runs' / (iid + '.json'), result)
    return result


def run():
    verify(); budget = ReservedBudget(CAP)
    for f in (EXPERIMENT / 'responses').glob('*.json'):
        value = read_json(f)['data']['usage'].get('cost')
        if value is None: raise RuntimeError('Resolve missing cost before continuing')
        budget.add(float(value))
    for f in (EXPERIMENT / 'transport_failures').glob('*.json'):
        budget.uncertain_cost_bound += sum(x['uncertain_cost_upper'] for x in read_json(f))
    todo = [x['input_id'] for x in read_json(SOURCE / 'inputs.json') if not (EXPERIMENT / 'runs' / (x['input_id']+'.json')).exists()]
    errors = []; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(run_one, iid, budget): iid for iid in todo}
        for n, future in enumerate(as_completed(jobs), 1):
            try: future.result()
            except Exception as exc:
                error = dict(input_id=jobs[future], error=str(exc)[:800]); errors.append(error)
                print('REQUEST ERROR ' + json.dumps(error), flush=True)
            if n % 50 == 0 or n == len(jobs):
                print(f'Controlled prefixes {n}/{len(jobs)}; new usage ${budget.cost:.6f}; errors {len(errors)}', flush=True)
    write(EXPERIMENT / 'execution.json', dict(errors=errors, new_reported_cost_usd=budget.cost,
        uncertain_cost_bound=budget.uncertain_cost_bound, replay_and_terminal_wall_seconds=time.monotonic()-started))
    if errors: raise RuntimeError('Unresolved failures; preserve completed requests and diagnose')


def analyze():
    verify(); labels = read_json(SOURCE / 'labels.json')
    original = [source_run(x['input_id']) for x in labels]
    rescued = [read_json(EXPERIMENT / 'runs' / (x['input_id']+'.json')) for x in labels]
    assert len(rescued) == 500
    vectors = {}; summaries = {}
    for arm, rr in [('source', original), ('rescue', rescued)]:
        vectors[arm] = {s: [r['prediction'][field] == x['gold'][s] for r, x in zip(rr, labels)] for s, field in FIELDS.items()}
        systems = {}
        for s in SYSTEMS:
            systems[s] = {}
            for bucket in ['all', 'unseen', 'rare', 'common']:
                ii = [i for i, x in enumerate(labels) if bucket == 'all' or x['bucket'][s] == bucket]
                systems[s][bucket] = dict(n=len(ii), correct=sum(vectors[arm][s][i] for i in ii),
                    accuracy=sum(vectors[arm][s][i] for i in ii)/len(ii))
        summaries[arm] = dict(per_system=systems, headline=min(systems[s]['all']['accuracy'] for s in SYSTEMS),
            both_correct=sum(all(vectors[arm][s][i] for s in SYSTEMS) for i in range(500)), statuses=dict(Counter(r['status'] for r in rr)))
    paired = {}
    for s in SYSTEMS:
        paired[s] = {}
        for bucket in ['all', 'unseen', 'rare', 'common']:
            ii = [i for i, x in enumerate(labels) if bucket == 'all' or x['bucket'][s] == bucket]
            a, b = [vectors['source'][s][i] for i in ii], [vectors['rescue'][s][i] for i in ii]
            gains = sum(y and not x for x, y in zip(a, b)); losses = sum(x and not y for x, y in zip(a, b))
            paired[s][bucket] = dict(n=len(ii), corrections=gains, regressions=losses, delta=(gains-losses)/len(ii),
                cluster_95_interval=cluster_interval(a, b, [labels[i]['group_id'] for i in ii]))
    grouped = defaultdict(list)
    for i, x in enumerate(labels): grouped[x['group_id']].append(i)
    groups = list(grouped.values()); rng = random.Random(13); samples = []
    for _ in range(5000):
        ii = [i for _ in groups for i in groups[rng.randrange(len(groups))]]
        score = lambda arm: min(sum(vectors[arm][s][i] for i in ii)/len(ii) for s in SYSTEMS)
        samples.append(score('rescue')-score('source'))
    samples.sort()
    attempted = [r for r in rescued if r['rescue_attempted']]
    assert len(attempted) == 35 and len({r['terminal']['generation_id'] for r in attempted}) == 35
    for old, new in zip(original, rescued):
        if old['status'] == 'model': assert old['prediction'] == new['prediction'] and not new['rescue_attempted']
    subgroup = {}
    for s in SYSTEMS:
        ii = [i for i, r in enumerate(rescued) if r['rescue_attempted']]
        subgroup[s] = dict(n=35, source_correct=sum(vectors['source'][s][i] for i in ii), rescue_correct=sum(vectors['rescue'][s][i] for i in ii))
    cost = sum(float(r['terminal']['usage']['cost']) for r in attempted)
    assert cost <= CAP
    latencies = sorted(r['terminal_latency_seconds'] for r in attempted)
    historical = sum(r['cost_usd'] for r in original)
    result = dict(arms=summaries, paired=paired,
        headline_delta=summaries['rescue']['headline']-summaries['source']['headline'], headline_cluster_95_interval=[samples[125], samples[4875]],
        routed_subgroup=subgroup, new_requests=35, new_cost_usd=cost, historical_prefix_cost_usd=historical,
        estimated_combined_reported_model_cost_usd=historical+cost,
        terminal_latency_median=statistics.median(latencies), terminal_latency_p95=latencies[math.ceil(.95*len(latencies))-1],
        mean_incremental_latency_across_all_inputs=sum(latencies)/500,
        limitation='Controlled continuation replay on known development prefixes, with fresh terminal requests; not fresh end-to-end inference. Historic prefix cost is not a new charge; latency is measured only for the new terminal calls.')
    write(EXPERIMENT / 'results.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'verify', 'run', 'analyze'])
    args = parser.parse_args(); globals()[args.command]()
