"""One bounded guided terminal selection, only after model-budget exhaustion."""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3

from .common import SEED, SYSTEMS
from .selector500_eval import (FIELDS, build_messages, reconstruct, response_format,
                               validate, MODEL, PROVIDER)

MAX_TOKENS = 2048


def needs_rescue(run):
    if run['status'] == 'model':
        return False
    if run['status'] != 'fallback_budget_or_invalid':
        raise ValueError('Infrastructure/unknown failures require diagnosis, not terminal rescue')
    if run['model_calls'] != 5 or len(run['trace']) != 5:
        raise ValueError('Only the exhausted five-response workflow is eligible')
    return True


def verify_prefix(run):
    """Verify every source assistant request/response against the saved conversation."""
    positions = [i for i, m in enumerate(run['messages']) if m['role'] == 'assistant']
    assert len(positions) == len(run['responses']) == run['model_calls']
    for i, response in zip(positions, run['responses']):
        serialized = json.dumps(run['messages'][:i], sort_keys=True, ensure_ascii=False)
        assert hashlib.sha256(serialized.encode()).hexdigest() == response['request_sha256']
        raw = response['data']['choices'][0]['message']
        expected = {k: v for k, v in raw.items() if k in {'role', 'content', 'tool_calls', 'reasoning_details'} and v is not None}
        assert run['messages'][i] == expected
    return len(positions)


def prepare_terminal(run, initial, db):
    if not needs_rescue(run):
        raise ValueError('Accepted finishes do not need terminal evidence')
    seen, examples = reconstruct(initial, run)
    cards = {}
    for system in SYSTEMS:
        cards[system] = []
        ordered = sorted(seen[system], key=lambda c: hashlib.sha256(f'{SEED}:{run["input_id"]}:{system}:{c}'.encode()).hexdigest())
        for code in ordered:
            row = db.execute('select code,preferred,tag,fsn from concepts where system=? and code=?', (system, code)).fetchone()
            if row is None:
                raise ValueError('Visible candidate absent from supplied dictionary')
            card = dict(row)
            card['synonyms'] = [x[0] for x in db.execute('select distinct text from aliases where system=? and code=? order by description_order,text limit 4', (system, code))]
            cards[system].append(card)
        if not cards[system]:
            raise ValueError('No visible candidates for terminal selection')
    payload = dict(mention=initial['payload']['mention'], context=initial['payload']['context'],
                   candidates=cards, training_examples=examples)
    # This is the original fallback, not a generated prediction or evaluation label.
    return dict(input_id=run['input_id'], payload=payload,
                fallback=copy.deepcopy(initial['fallback']), distinctions={})


def terminal_request(prepared):
    return dict(model=MODEL, messages=build_messages(prepared, 'coding_guide'),
        response_format=response_format(prepared), temperature=0, seed=SEED,
        max_tokens=MAX_TOKENS, reasoning={'enabled': False}, usage={'include': True},
        provider={'order': [PROVIDER], 'allow_fallbacks': False, 'require_parameters': True})


def finalize_after_budget(run, prepared, complete):
    """Accepted source decisions are immutable; no correctness/confidence routing."""
    if not needs_rescue(run):
        return dict(input_id=run['input_id'], prediction=copy.deepcopy(run['prediction']),
            status='retained_model', rescue_attempted=False, accepted_terminal=False,
            source_status=run['status'], source_model_calls=run['model_calls'],
            total_model_calls=run['model_calls'], terminal=None)
    if prepared['input_id'] != run['input_id']:
        raise ValueError('Terminal evidence belongs to a different input')
    raw = complete(terminal_request(prepared))
    if raw.get('provider') != PROVIDER or raw.get('model') != MODEL:
        raise ValueError('Unexpected terminal provider/model')
    message = raw['choices'][0]['message']
    try:
        selected = json.loads(message.get('content') or '')
        error = validate(selected, prepared)
    except (ValueError, TypeError):
        error = 'Invalid JSON terminal selection'
    accepted = error is None
    return dict(input_id=run['input_id'], prediction=selected if accepted else copy.deepcopy(run['prediction']),
        status='rescued_model' if accepted else 'retained_fallback', rescue_attempted=True,
        accepted_terminal=accepted, error=error, source_status=run['status'],
        source_model_calls=run['model_calls'], total_model_calls=run['model_calls'] + 1,
        terminal=dict(generation_id=raw['id'], usage=raw['usage'], provider=raw['provider'], model=raw['model'],
            finish_reason=raw['choices'][0].get('finish_reason')))
