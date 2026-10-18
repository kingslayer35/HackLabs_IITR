"""Read-only audit of the completed 500-input traces; no model requests."""
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .analyze_selection import codes_in_result, visible_examples

SOURCE = Path(__file__).parent / 'artifacts/diverse500_v1'
OUTPUT = SOURCE.parent / 'agent_focus_audit_v1'
FIELDS = {'ICD10CM': 'icd10cm_code', 'SNOMEDCT_US': 'snomed_code'}


def main():
    read = lambda p: json.loads(p.read_text())
    paths = [SOURCE / 'labels.json', SOURCE / 'examples.sqlite']
    paths += sorted((SOURCE / 'sapbert_icd_text_snomed/runs').glob('*.json'))
    paths += sorted((SOURCE / 'sapbert_icd_text_snomed/initial_evidence').glob('*.json'))
    digest = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = digest()
    db = sqlite3.connect(f'file:{SOURCE.parent}/terminology.sqlite?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    concept = lambda s, c: dict(db.execute('select code,preferred,tag,fsn from concepts where system=? and code=?', (s, c)).fetchone())
    counts = {s: Counter() for s in FIELDS}
    cases = []
    for item in read(SOURCE / 'labels.json'):
        iid = item['input_id']
        run = read(SOURCE / 'sapbert_icd_text_snomed/runs' / (iid + '.json'))
        evidence = read(SOURCE / 'sapbert_icd_text_snomed/initial_evidence' / (iid + '.json'))
        if run['status'] != 'model':
            continue
        finish = next(t['turn'] for t in run['trace'] if any(c['name'] == 'finish' and c['result'].get('accepted') for c in t['tool_calls']))
        inspected = {s: set() for s in FIELDS}
        seen = {s: set(evidence['seen'][s]) for s in FIELDS}
        for turn in run['trace']:
            if turn['turn'] >= finish:
                break
            for call in turn['tool_calls']:
                returned = codes_in_result(call['result'])
                for s in FIELDS:
                    seen[s].update(returned[s])
                    if call['name'] == 'inspect_concepts':
                        inspected[s].update(returned[s])
        for s, field in FIELDS.items():
            gold, pred = item['gold'][s], run['prediction'][field]
            if gold == pred or gold not in seen[s]:
                continue
            examples = visible_examples(evidence, run['trace'], s, finish)
            example_codes = {e['code'] for e in examples}
            g, p = concept(s, gold), concept(s, pred)
            flags = dict(gold_in_initial_evidence=gold in evidence['seen'][s],
                         gold_explicitly_inspected=gold in inspected[s],
                         prediction_supported_by_example=pred in example_codes,
                         gold_supported_by_example=gold in example_codes,
                         both_supported_by_examples=gold in example_codes and pred in example_codes,
                         different_semantic_tag=s == 'SNOMEDCT_US' and g['tag'] != p['tag'])
            counts[s]['accepted_visible_gold_errors'] += 1
            counts[s].update({k: int(v) for k, v in flags.items()})
            cases.append(dict(input_id=iid, system=s, mention=item['entity'], context=item['chunk_text'],
                              bucket=item['bucket'][s], gold=g, predicted=p, **flags,
                              final_explanation=run['prediction']['evidence'], visible_examples=examples))
    assert counts['ICD10CM']['accepted_visible_gold_errors'] == 69
    assert counts['SNOMEDCT_US']['accepted_visible_gold_errors'] == 67
    train = sqlite3.connect(f'file:{SOURCE}/examples.sqlite?mode=ro', uri=True)
    memory = {}
    for s in FIELDS:
        rows = train.execute('select entity,code,input_id from examples where system=?', (s,)).fetchall()
        normalized = defaultdict(set)
        for mention, code, _ in rows:
            normalized[re.sub(r'\s+', ' ', mention.casefold()).strip()].add(code)
        ambiguous = {m: sorted(v) for m, v in normalized.items() if len(v) > 1}
        memory[s] = dict(rows=len(rows), unique_codes=len({r[1] for r in rows}),
                         normalized_mentions=len(normalized), mentions_with_multiple_codes=len(ambiguous),
                         rows_with_ambiguous_mention=sum(len(normalized[re.sub(r'\s+', ' ', r[0].casefold()).strip()]) > 1 for r in rows),
                         ambiguous_mentions=ambiguous)
    assert sum(v['rows'] for v in memory.values()) == 3423
    assert before == digest(), 'Completed experiment changed during read-only analysis'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(dict(selection={s: dict(v) for s, v in counts.items()},
        training_memory=memory, note='Flags overlap. Example support is association, not evidence of copying. Multiple labels can reflect legitimate context distinctions or annotation differences. No clinical adjudication.'), indent=2) + '\n')
    (OUTPUT / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (OUTPUT / 'source_hashes.json').write_text(json.dumps(before, indent=2) + '\n')
    print(json.dumps(dict(selection={s: dict(v) for s, v in counts.items()},
                         training_memory={s: {k: v for k, v in x.items() if k != 'ambiguous_mentions'} for s, x in memory.items()}), indent=2))


if __name__ == '__main__':
    main()
