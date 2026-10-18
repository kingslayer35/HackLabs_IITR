"""ICD-only backend and label-blind selective SNOMED paging development ablations.

Previous sources and traces are read-only. All new artifacts live in a new run directory.
"""
import argparse
import copy
import csv
import json
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import OpenRouterClient, validate_finish
from .common import ARTIFACTS, SYSTEMS, SEED, digest, read_json, write_json
from .comparisons import PolicyTools, ReservedBudget, TASK_PROMPT, assert_flat, paired_delta
from .dense_agent_tools import DenseAgentTools, agent_schemas
from .recovery_eval import BASE_POLICY
from .selection_prompt_eval import file_hash

SOURCE = ARTIFACTS / 'dense_agent_v1'
EXPERIMENT = ARTIFACTS / 'backend_ablation_v1'
MODEL = 'deepseek/deepseek-v4-flash-0731'
ARMS = ('icd_only', 'selective_snomed')
FIELDS = dict(zip(SYSTEMS, ('icd10cm_code', 'snomed_code')))
THRESHOLD = .85
REVIEW_POLICY = """Finish normally when ready. If your proposed finish has confidence below 0.85,
the controller may return one unread SNOMED page for review, once per input, during responses 1-3.
Read that evidence and reconsider both codes; you need not change either. This heuristic uses overall
confidence, not calibrated SNOMED confidence. Responses 4-5 will never be delayed for this review.
Keep your confidence honest; do not raise it to avoid review. The total budget remains 5 responses."""


class AblationTools(DenseAgentTools):
    def __init__(self, path, arm, **kwargs):
        super().__init__(path, **kwargs)
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm = arm
        self.reviewed = False
        self.review_turn = None
        self.review_events = []

    def search_concepts(self, system, queries, limit=12):
        if self.arm == 'icd_only' and system == 'SNOMEDCT_US':
            return PolicyTools.search_concepts(self, system, queries, limit)
        return super().search_concepts(system, queries, limit)

    def maybe_review(self, candidate, turn):
        """No labels or baseline predictions are available to this runtime policy."""
        if self.arm != 'selective_snomed' or self.reviewed:
            return None
        if not isinstance(candidate.get('confidence'), (float, int)) or candidate['confidence'] >= THRESHOLD:
            return None
        pools = [(k, v) for k, v in self.pools.items() if v['system'] == 'SNOMEDCT_US']
        # Prefer the latest query supporting the proposed code; otherwise latest SNOMED query.
        matching = [(k, v) for k, v in pools if candidate['snomed_code'] in {r['code'] for r in v['results']}]
        ordered = list(reversed(matching or pools))
        chosen = next(((k, v, i) for k, v in ordered for i, r in enumerate(v['results'])
                       if r['code'] not in self.seen['SNOMEDCT_US']), None)
        if chosen is None:
            return None
        search_id, _, offset = chosen
        if turn > 3:
            self.review_events.append(dict(triggered=False, reason='too_late', turn=turn,
                                           proposed_confidence=candidate['confidence']))
            return None
        self.reviewed = True
        self.review_turn = turn
        page = self.get_more_candidates(search_id, offset, limit=12)
        self.review_events.append(dict(triggered=True, turn=turn, proposed=copy.deepcopy(candidate),
                                      search_id=search_id, offset=offset,
                                      codes=[r['code'] for r in page['results']]))
        return dict(page, accepted=False, review_required=True,
                    note='Read this additional SNOMED evidence, then finish within the remaining response budget.')


def schemas(arm, final_turn=False):
    result = agent_schemas(final_turn)
    if arm == 'icd_only' and not final_turn:
        for row in result:
            if row['function']['name'] == 'search_concepts':
                row['function']['description'] = ('Search supplied dictionaries using SapBERT for ICD and the original lexical '
                    'retriever for SNOMED. ICD results include a search_id for deeper pages. Reformulate when needed. '
                    'SNOMED has the original search limit, up to30; no hidden SNOMED pool. Rank is not clinical confidence.')
    return result


def prepare():
    EXPERIMENT.mkdir(exist_ok=True)
    if (EXPERIMENT / 'config.json').exists():
        verify_sources()
        return
    protected = {str(p.relative_to(ARTIFACTS)): file_hash(p)
                 for directory in ('dense_agent_v1', 'recovery_v1')
                 for p in sorted((ARTIFACTS / directory).rglob('*')) if p.is_file()}
    inputs = read_json(SOURCE / 'inputs.json')
    providers = {}
    for x in inputs:
        name = x['input_id'] + '.json'
        old = read_json(SOURCE / 'runs/dense_paging' / name)
        providers[x['input_id']] = old['provider_requested']
        dense = read_json(SOURCE / 'initial_evidence' / name)
        lexical = read_json(SOURCE / 'baseline_initial_evidence' / name)
        for arm in ARMS:
            evidence = copy.deepcopy(dense)
            pools = read_json(SOURCE / 'initial_pools' / name)
            if arm == 'icd_only':
                s = 'SNOMEDCT_US'
                evidence['payload']['initial_candidates'][s] = lexical['payload']['initial_candidates'][s]
                evidence['seen'][s] = lexical['seen'][s]
                evidence['fallback'][s] = lexical['fallback'][s]
                pools = {k: v for k, v in pools.items() if v['system'] == 'ICD10CM'}
            assert_flat(evidence['payload'])
            evidence['sha256'] = digest(json.dumps(evidence['payload'], sort_keys=True, ensure_ascii=False))
            write_json(EXPERIMENT / arm / 'initial_evidence' / name, evidence)
            write_json(EXPERIMENT / arm / 'initial_pools' / name, pools)
    write_json(EXPERIMENT / 'inputs.json', inputs)
    write_json(EXPERIMENT / 'retained_sources.json', protected)
    config = dict(model=MODEL, providers=providers, max_responses=5, seed=SEED, temperature=0,
                  max_completion_tokens=2048, reasoning=False, cap_usd=.30, workers=2,
                  development_sample='same inspected80; no fresh confirmation claim',
                  threshold=THRESHOLD, policy=REVIEW_POLICY, review_deadline_response=3,
                  review_max_pages_per_input=1, review_page_size=12,
                  provider_control='same provider per input as cached dense_paging; original lexical baseline entirely DeepInfra',
                  final_tool_choice='DeepInfra named finish, CoreWeave auto with finish-only menu, matching compatible prior continuation',
                  source_sha256={str(p): file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                  evidence_sha256={str(p.relative_to(EXPERIMENT)): file_hash(p)
                                   for arm in ARMS for sub in ('initial_evidence', 'initial_pools')
                                   for p in sorted((EXPERIMENT / arm / sub).glob('*.json'))})
    write_json(EXPERIMENT / 'config.json', config)


def verify_sources():
    for relative, sha in read_json(EXPERIMENT / 'retained_sources.json').items():
        assert file_hash(ARTIFACTS / relative) == sha, relative
    cfg = read_json(EXPERIMENT / 'config.json')
    for path, sha in cfg['source_sha256'].items():
        assert file_hash(Path(path)) == sha, path
    for relative, sha in cfg['evidence_sha256'].items():
        assert file_hash(EXPERIMENT / relative) == sha, relative


class Client(OpenRouterClient):
    def complete(self, messages, tools, force_finish=False):
        prices = {'DeepInfra': (.00000006, .00000018), 'CoreWeave': (.00000013, .00000028)}
        inp, out = prices[self.provider]
        # UTF-8 byte length exceeds ordinary tokenizer size; reserve both potential attempts.
        size = len(json.dumps(messages, ensure_ascii=False).encode()) + len(json.dumps(tools).encode()) + 10000
        one = size * inp + 2048 * out
        self.budget.reserve(2 * one)
        body = dict(model=self.model, messages=messages, tools=tools,
                    tool_choice={'type': 'function', 'function': {'name': 'finish'}}
                    if force_finish and self.provider == 'DeepInfra' else 'auto',
                    temperature=0, seed=SEED, max_tokens=2048, reasoning={'enabled': False},
                    usage={'include': True}, provider={'order': [self.provider], 'allow_fallbacks': False,
                                                     'require_parameters': True})
        attempts = []
        try:
            for attempt in range(2):
                try:
                    r = requests.post(self.url, headers={'Authorization': 'Bearer ' + self.key,
                                      'Content-Type': 'application/json'}, json=body, timeout=(15, 120))
                    attempts.append(dict(status=r.status_code))
                    if r.status_code != 200:
                        if not attempt and (r.status_code in {408, 429} or r.status_code >= 500):
                            time.sleep(3)
                            continue
                        raise RuntimeError('OpenRouter HTTP ' + str(r.status_code) + ': ' + r.text.replace(self.key, '[REDACTED]')[:500])
                    data = r.json()
                    if data.get('error'):
                        raise RuntimeError(str(data['error']).replace(self.key, '[REDACTED]')[:500])
                    cost = data.get('usage', {}).get('cost')
                    self.budget.add(float(cost or 0))
                    write_json(EXPERIMENT / 'usage_ledger' / (data['id'] + '.json'),
                               dict(generation_id=data['id'], usage=data.get('usage', {}),
                                    provider=data.get('provider'), attempts=attempts))
                    self.budget.release(2 * one, one * (sum(a.get('status') != 200 for a in attempts) + int(cost is None)))
                    return data, attempts
                except (requests.Timeout, requests.ConnectionError) as exc:
                    attempts.append(dict(error=type(exc).__name__))
                    if attempt:
                        raise RuntimeError('Transport failed after two attempts') from None
                    time.sleep(3)
        except Exception:
            self.budget.release(2 * one, 2 * one)
            raise


def run_one(item, arm, budget, client_factory=Client, fetch=None):
    if set(item) != {'input_id', 'entity', 'chunk_text'}:
        raise ValueError('Only unlabelled input fields permitted')
    start = time.monotonic()
    name = item['input_id'] + '.json'
    evidence = read_json(EXPERIMENT / arm / 'initial_evidence' / name)
    provider = read_json(EXPERIMENT / 'config.json')['providers'][item['input_id']]
    tools = AblationTools(SOURCE / 'examples.sqlite', arm, fetch=fetch)
    tools.seen = {s: set(evidence['seen'][s]) for s in SYSTEMS}
    tools.pools = read_json(EXPERIMENT / arm / 'initial_pools' / name)
    for s in SYSTEMS:
        tools.exposed[s] = {r['code'] for r in evidence['payload']['initial_candidates'][s]['results']}
    policy = BASE_POLICY + ('\n' + REVIEW_POLICY if arm == 'selective_snomed' else '')
    messages = [dict(role='system', content=TASK_PROMPT + '\n' + policy),
                dict(role='user', content=json.dumps(evidence['payload'], ensure_ascii=False))]
    client = client_factory(MODEL, provider, budget)
    trace, usages = [], []
    prediction, error, status = None, None, 'model'
    visible = {s: set(tools.seen[s]) for s in SYSTEMS}
    try:
        for turn in range(1, 6):
            visible = {s: set(tools.seen[s]) for s in SYSTEMS}
            allowed = schemas(arm, final_turn=turn == 5)
            data, attempts = client.complete(messages, allowed, force_finish=turn == 5)
            usages.append(dict(usage=data.get('usage', {}), provider=data.get('provider'),
                               generation_id=data.get('id'), attempts=attempts))
            message = data['choices'][0]['message']
            messages.append({k: v for k, v in message.items()
                             if k in {'role', 'content', 'tool_calls', 'reasoning_details'} and v is not None})
            record = dict(turn=turn, tool_calls=[], assistant_content=message.get('content'))
            trace.append(record)

            def finish(candidate):
                if tools.review_turn == turn:
                    return {'error': 'Read the review page in a subsequent response before finishing'}, None
                issue = validate_finish(candidate, tools)
                if not issue and any(candidate[FIELDS[s]] not in visible[s] for s in SYSTEMS):
                    issue = 'Read codes in tool results before selecting them'
                if issue:
                    return {'error': issue}, None
                review = tools.maybe_review(candidate, turn)
                if review:
                    return review, None
                return {'accepted': True}, candidate

            calls = message.get('tool_calls', [])
            if len(calls) > 12:
                raise RuntimeError('Per-response tool cap exceeded')
            if not calls:
                try:
                    candidate = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', message.get('content') or '').strip())
                    result, prediction = finish(candidate)
                    record['tool_calls'].append(dict(name='finish', arguments=candidate, result=result, text_submission=True))
                    if prediction:
                        break
                    messages.append(dict(role='user', content=json.dumps(result, ensure_ascii=False)))
                except (ValueError, TypeError, AttributeError):
                    messages.append(dict(role='user', content='Use permitted tools and finish with both codes within the remaining budget.'))
                continue
            permitted = {t['function']['name'] for t in allowed}
            for call in calls:
                fname, arguments = call['function']['name'], {}
                try:
                    arguments = json.loads(call['function']['arguments'])
                    if not isinstance(arguments, dict):
                        raise ValueError('Tool arguments must be an object')
                    if fname not in permitted:
                        raise ValueError('Tool disabled in this phase')
                    if fname == 'finish':
                        result, prediction = finish(arguments)
                    else:
                        result = tools.dispatch(fname, arguments)
                        assert_flat(result)
                except (ValueError, TypeError, sqlite3.Error) as exc:
                    result = {'error': str(exc)}
                record['tool_calls'].append(dict(name=fname, arguments=arguments, result=result))
                messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(result, ensure_ascii=False)))
                if prediction:
                    break
            if prediction:
                break
    except Exception as exc:
        error = str(exc).replace(client.key, '[REDACTED]')[:1000]
        status = 'fallback_error'
    if prediction is None:
        if status != 'fallback_error':
            status = 'fallback_budget_or_invalid'
        prediction = dict(icd10cm_code=evidence['fallback']['ICD10CM'], snomed_code=evidence['fallback']['SNOMEDCT_US'],
                          confidence=0., evidence='Deterministic initial-search fallback; no accepted model finish.')
    result = dict(input_id=item['input_id'], arm=arm, model=MODEL, provider_requested=provider,
                  prediction=prediction, status=status, error=error, trace=trace, messages=messages,
                  seen_candidates={s: sorted(visible[s]) for s in SYSTEMS}, pages_read=tools.pages_read,
                  review_events=tools.review_events, model_calls=len(usages), usage_records=usages,
                  cost_usd=sum(float(r['usage'].get('cost') or 0) for r in usages),
                  latency_seconds=round(time.monotonic() - start, 3))
    tools.close()
    return result


def run(arm):
    verify_sources()
    cap = read_json(EXPERIMENT / 'config.json')['cap_usd']
    budget = ReservedBudget(cap)
    ledger = [read_json(p) for p in (EXPERIMENT / 'usage_ledger').glob('*.json')]
    budget.add(sum(float(r['usage'].get('cost') or 0) for r in ledger))
    if (EXPERIMENT / 'execution.json').exists():
        budget.uncertain_cost_bound = read_json(EXPERIMENT / 'execution.json')['uncertain_cost_upper_bound_usd']
    jobs = []
    for item in read_json(EXPERIMENT / 'inputs.json'):
        p = EXPERIMENT / arm / 'runs' / (item['input_id'] + '.json')
        if p.exists():
            if read_json(p)['status'] == 'fallback_error':
                raise RuntimeError('Saved infrastructure failure: resolve and preserve partial responses before resuming')
            continue
        jobs.append((item, p))
    # Interleave provider assignments without changing which provider serves an input.
    jobs.sort(key=lambda pair: digest('backend-ablation:' + pair[0]['input_id']))
    def work(pair):
        item, path = pair
        value = run_one(item, arm, budget)
        write_json(path, value)
        return value
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(work, j): j for j in jobs[:2]}
            queue = iter(jobs[2:])
            done = 0
            while pending:
                f = next(as_completed(pending))
                pending.pop(f)
                r = f.result()
                done += 1
                print(f'{arm} [{done}/{len(jobs)}] {r["input_id"]} {r["status"]} calls={r["model_calls"]} '
                      f'reviews={sum(v["triggered"] for v in r["review_events"])} total=${budget.cost:.5f}', flush=True)
                if r['status'] == 'fallback_error':
                    raise RuntimeError('Infrastructure failure; stopping submissions and retaining completed work: ' + str(r['error']))
                job = next(queue, None)
                if job:
                    pending[pool.submit(work, job)] = job
    finally:
        write_json(EXPERIMENT / 'execution.json', dict(reported_cost_usd=budget.cost,
                   uncertain_cost_upper_bound_usd=budget.uncertain_cost_bound, cap_usd=cap))
        verify_sources()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        if not args.arm:
            parser.error('--arm is required')
        run(args.arm)


if __name__ == '__main__':
    main()
