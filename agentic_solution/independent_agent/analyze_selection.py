"""Offline baseline selection-error audit. No inference or network calls.

Review annotations are stored with ignored private experiment artifacts, separately
from the inference code and its prompts. They are never agent training examples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .common import ARTIFACTS, SYSTEMS, read_json, write_json

SOURCE = ARTIFACTS / 'recovery_v1'
OUTPUT = ARTIFACTS / 'selection_audit_v1'
FIELDS = {'ICD10CM': 'icd10cm_code', 'SNOMEDCT_US': 'snomed_code'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def codes_in_result(value, system=None):
    result = {s: set() for s in SYSTEMS}
    if isinstance(value, dict):
        if value.get('error'):
            return result
        system = value.get('system', system)
        if system in SYSTEMS and isinstance(value.get('code'), str):
            result[system].add(value['code'])
        for child in value.values():
            for s, codes in codes_in_result(child, system).items():
                result[s].update(codes)
    elif isinstance(value, list):
        for child in value:
            for s, codes in codes_in_result(child, system).items():
                result[s].update(codes)
    return result


def visible_examples(evidence, trace, system, finish_turn):
    examples = list(evidence['payload']['training_examples'][system]['examples'])
    for turn in trace:
        if turn['turn'] >= finish_turn:
            break
        for call in turn['tool_calls']:
            if call['name'] == 'find_training_examples':
                examples.extend(v for v in call['result'].get('examples', []) if v['system'] == system)
    unique = {}
    for row in examples:
        unique[(row['example_id'], row['system'], row['code'])] = row
    return list(unique.values())


def analyze():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotations = read_json(OUTPUT / 'review_annotations.json')
    source_paths = [SOURCE / name for name in ('inputs.json', 'labels.json', 'results.json', 'examples.sqlite')]
    source_paths += sorted((SOURCE / 'runs' / 'baseline').glob('*.json'))
    source_paths += sorted((SOURCE / 'initial_evidence').glob('*.json'))
    before = {str(p.relative_to(ARTIFACTS)): sha(p) for p in source_paths}
    labels = read_json(SOURCE / 'labels.json')
    existing = read_json(SOURCE / 'results.json')['arms']['baseline']
    con = sqlite3.connect(f"file:{ARTIFACTS / 'terminology.sqlite'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    def concept(system, code):
        row = con.execute('SELECT code,preferred,fsn,tag FROM concepts WHERE system=? AND code=?', (system, code)).fetchone()
        assert row is not None
        value = dict(row)
        value['aliases'] = [r[0] for r in con.execute('SELECT DISTINCT text FROM aliases WHERE system=? AND code=? ORDER BY description_order,text LIMIT 8',(system,code))]
        return value
    cases, all_rows = [], []
    for item in labels:
        iid = item['input_id']
        run = read_json(SOURCE / 'runs' / 'baseline' / (iid+'.json'))
        evidence = read_json(SOURCE / 'initial_evidence' / (iid+'.json'))
        assert run['status'] == 'model'
        seen = {s: set(evidence['seen'][s]) for s in SYSTEMS}
        finish_turn, before_finish = None, None
        inspected = {s: set() for s in SYSTEMS}
        first_tools = {s: {} for s in SYSTEMS}
        for turn in run['trace']:
            if any(c['name']=='finish' and c['result'].get('accepted') for c in turn['tool_calls']):
                finish_turn = turn['turn']
                before_finish = {s: set(cc) for s,cc in seen.items()}
                break
            for call in turn['tool_calls']:
                returned = codes_in_result(call['result'])
                for s, cc in returned.items():
                    for code in cc - seen[s]:
                        first_tools[s][code] = {'turn': turn['turn'], 'tool': call['name']}
                    seen[s].update(cc)
                    if call['name']=='inspect_concepts':
                        inspected[s].update(cc)
        assert finish_turn is not None
        for system, field in FIELDS.items():
            gold, predicted = item['gold'][system], run['prediction'][field]
            examples = visible_examples(evidence, run['trace'], system, finish_turn)
            example_codes = {ex['code'] for ex in examples}
            wrong = gold != predicted
            ever = gold in run['seen_candidates'][system]
            before_decision = gold in before_finish[system]
            record = {'input_id':iid,'system':system,'correct':not wrong,
                      'gold_ever_retrieved':ever,'gold_visible_before_finish_response':before_decision,
                      'predicted_code_in_visible_examples':predicted in example_codes,
                      'gold_code_in_visible_examples':gold in example_codes}
            all_rows.append(record)
            if not wrong or not ever:
                continue
            # These are actual pre-decision selection errors, not gold that only
            # appeared after finish or in a co-generated, not-yet-read tool result.
            assert before_decision
            key = system+':'+iid
            review = annotations[key]
            candidate_codes = [r['code'] for r in evidence['payload']['initial_candidates'][system]['results']]
            initial_gold = gold in evidence['seen'][system]
            cases.append({**record,'mention':item['entity'],'gold_code':gold,'predicted_code':predicted,
                          'gold':concept(system,gold),'predicted':concept(system,predicted),
                          'gold_in_initial_evidence':initial_gold,
                          'gold_initial_candidate_rank':candidate_codes.index(gold)+1 if gold in candidate_codes else None,
                          'gold_explicitly_inspected':gold in inspected[system],
                          'predicted_explicitly_inspected':predicted in inspected[system],
                          'first_new_gold_tool':first_tools[system].get(gold),
                          'visible_examples':examples,'confidence':run['prediction']['confidence'],
                          'joint_final_explanation':run['prediction']['evidence'],
                          'finish_response_number':finish_turn, 'review':review,
                          'source_trace':str(SOURCE / 'runs' / 'baseline' / (iid+'.json'))})
    con.close()
    assert set(annotations) == {c['system']+':'+c['input_id'] for c in cases}
    per_system = {}
    for system in SYSTEMS:
        rr = [r for r in all_rows if r['system']==system]
        cc = [c for c in cases if c['system']==system]
        expected = existing['per_system'][system]
        assert sum(r['correct'] for r in rr) == expected['correct']
        assert len(cc) == expected['selection_errors'] == 14
        correct = [r for r in rr if r['correct']]
        per_system[system] = {'n':len(rr),'correct':len(correct),'all_errors':sum(not r['correct'] for r in rr),
             'selection_errors':len(cc),'retrieval_misses':sum(not r['correct'] and not r['gold_ever_retrieved'] for r in rr),
             'selection_gold_initially_visible':sum(c['gold_in_initial_evidence'] for c in cc),
             'selection_gold_explicitly_inspected':sum(c['gold_explicitly_inspected'] for c in cc),
             'selection_predicted_code_in_examples':sum(c['predicted_code_in_visible_examples'] for c in cc),
             'selection_gold_code_in_examples':sum(c['gold_code_in_visible_examples'] for c in cc),
             'correct_predictions_with_predicted_code_in_examples':sum(c['predicted_code_in_visible_examples'] for c in correct),
             'fixed_trace_oracle_correct':sum(r['correct'] or r['gold_ever_retrieved'] for r in rr)}
    summary = {'input_count':len(labels),'selection_error_count':len(cases),'distinct_affected_inputs':len({c['input_id'] for c in cases}),
               'per_system':per_system,'review_primary_categories':dict(Counter(c['review']['category'] for c in cases)),
               'gold_initially_visible':sum(c['gold_in_initial_evidence'] for c in cases),
               'gold_explicitly_inspected':sum(c['gold_explicitly_inspected'] for c in cases),
               'predicted_code_in_visible_examples':sum(c['predicted_code_in_visible_examples'] for c in cases),
               'gold_code_in_visible_examples':sum(c['gold_code_in_visible_examples'] for c in cases),
               'new_API_calls':0,'new_API_cost_usd':0,
               'limits':['Manual categories are reviewer interpretations, not causal ground truth.',
                         'Exact-match disagreement does not prove clinical invalidity or a bad annotation.',
                         'Explanations are model outputs, not reliable internal reasoning traces.',
                         'Stored dictionary names are not certified official preferred terms.',
                         'The same 80 reviewed development inputs are being analyzed; no claim of generalization.']}
    assert before == {str(p.relative_to(ARTIFACTS)): sha(p) for p in source_paths}
    write_json(OUTPUT/'source_hashes.json',before)
    write_json(OUTPUT/'cases.json',cases)
    write_json(OUTPUT/'summary.json',summary)
    with (OUTPUT/'cases.csv').open('w',newline='') as handle:
        fields=['input_id','system','mention','gold_code','gold_description','predicted_code','predicted_description',
                'primary_review_category','gold_in_initial_evidence','gold_explicitly_inspected','predicted_code_in_visible_examples',
                'gold_code_in_visible_examples','confidence','context_note','review','caution','joint_final_explanation']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for case in cases:
            writer.writerow({**{k:case[k] for k in ['input_id','system','mention','gold_code','predicted_code','gold_in_initial_evidence',
                             'gold_explicitly_inspected','predicted_code_in_visible_examples','gold_code_in_visible_examples','confidence','joint_final_explanation']},
                  'gold_description':case['gold']['preferred'],'predicted_description':case['predicted']['preferred'],
                  'primary_review_category':case['review']['category'],
                  **{k:case['review'][k] for k in ('context_note','review','caution')}})
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    analyze()
