"""Frozen diverse held-out 500-input benchmark; labels never enter inference."""
import argparse
import copy
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .agent import context_window
from .common import ARTIFACTS, ROOT, SYSTEMS, digest, input_id, jaccard, norm, read_json, shingles, write_json
from .comparisons import PolicyTools, assert_flat
from .backend_ablation import AblationTools
from .selection_prompt_eval import file_hash

EXPERIMENT = ARTIFACTS / 'diverse500_v1'
ARMS = ('text_both', 'sapbert_icd_text_snomed')


def bucket(n):
    return 'unseen' if n == 0 else 'rare' if n <= 5 else 'common'


def domain(code):
    letter = code[0]
    number = int(code[1:3]) if code[1:3].isdigit() else -1
    if letter in 'AB': return 'infectious'
    if letter == 'C' or letter == 'D' and number < 50: return 'neoplasms'
    if letter == 'D': return 'blood_immune'
    if letter == 'H': return 'eye' if number < 60 else 'ear'
    return dict(E='endocrine_metabolic', F='mental_behavioral', G='nervous', I='circulatory',
                J='respiratory', K='digestive', L='skin', M='musculoskeletal', N='genitourinary',
                O='pregnancy', P='perinatal', Q='congenital', R='symptoms_signs', S='injury',
                T='injury_poisoning', V='external_causes', W='external_causes', X='external_causes',
                Y='external_causes', Z='health_status_contact', U='special_purpose').get(letter, 'other')


def features(row):
    mention = norm(row['entity'])
    window = context_window(row['entity'], row['chunk_text'])
    return dict(domain=domain(row['gold']['ICD10CM']),
                short_or_abbreviation_candidate=len(mention.replace(' ', '')) <= 5 or bool(re.search(r'\b[A-Z]{2,6}\b', row['entity'])),
                laterality=bool(re.search(r'\b(left|right|bilateral)\b', mention)),
                acuity=bool(re.search(r'\b(acute|chronic|exacerbation)\b', mention)),
                stage_type_number=bool(re.search(r'\b(stage|type|grade)\b|\d', mention)),
                assertion_cue_in_window=bool(re.search(r'\b(no|not|denies|without|history of)\b', str(window).lower())),
                missing_context=not bool(row['chunk_text'].strip()))


def choose(pool, counts, count=500):
    """Cover chart groups first, then underrepresented codes/domains; no model outcomes."""
    if len(pool) < count: raise ValueError('Insufficient reserved paired inputs')
    remaining = {x['input_id']: x for x in pool}
    chosen = []
    code_counts = {s: Counter() for s in SYSTEMS}
    domains, groups = Counter(), Counter()
    while len(chosen) < count:
        def priority(x):
            new_group = int(groups[x['group_id']] == 0)
            novelty = sum(1 / (1 + code_counts[s][x['gold'][s]]) for s in SYSTEMS)
            tail = sum(counts[s][x['gold'][s]] <= 5 for s in SYSTEMS)
            return (new_group, novelty, tail, 1 / (1 + domains[domain(x['gold']['ICD10CM'])]),
                    -groups[x['group_id']], digest('diverse500:13:' + x['input_id']))
        row = max(remaining.values(), key=priority)
        chosen.append(row); del remaining[row['input_id']]
        groups[row['group_id']] += 1; domains[domain(row['gold']['ICD10CM'])] += 1
        for s in SYSTEMS: code_counts[s][row['gold'][s]] += 1
    return chosen


def select():
    EXPERIMENT.mkdir(exist_ok=True)
    if (EXPERIMENT / 'selection_manifest.json').exists(): return
    assignments = read_json(ARTIFACTS / 'split_manifest.json')['groups']
    previous = (read_json(ARTIFACTS / 'smoke20.json') + read_json(ARTIFACTS / 'comparisons_v1/labels.json')
                + read_json(ARTIFACTS / 'recovery_v1/labels.json'))
    previous_ids = {r['input_id'] for r in previous}
    previous_groups = {r['group_id'] for r in previous}
    grouped = defaultdict(list)
    for r in csv.DictReader((ROOT / 'data/hackathon_train.csv').open()):
        if r['subconcept'] != 'DiagnosisCode': continue
        r['input_id'] = input_id(r['entity'], r['chunk_text'])
        r['group_id'] = digest(norm(r['chunk_text']) or 'empty:' + norm(r['entity']))[:20]
        grouped[r['input_id']].append(r)
    db = sqlite3.connect(ARTIFACTS / 'terminology.sqlite'); db.row_factory = sqlite3.Row
    examples = [dict(r) for r in db.execute('select * from examples order by id')]
    frequencies = {s: Counter(r['code'] for r in examples if r['system'] == s) for s in SYSTEMS}
    pool = []
    for key, rows in grouped.items():
        if assignments[rows[0]['group_id']] == 'train' or key in previous_ids: continue
        if {r['coding_system'] for r in rows} != set(SYSTEMS): continue
        if any(len({r['code'] for r in rows if r['coding_system'] == s}) != 1 for s in SYSTEMS): continue
        pool.append(dict(input_id=key, entity=rows[0]['entity'], chunk_text=rows[0]['chunk_text'],
                         group_id=rows[0]['group_id'], gold={r['coding_system']: r['code'] for r in rows}))
    chosen = choose(pool, frequencies)
    mentions = {norm(r['entity']) for r in chosen}
    eval_groups = {r['group_id'] for r in chosen}
    target_shingles = {r['group_id']: shingles(r['chunk_text']) for r in chosen}
    example_shingles = {r['group_id']: shingles(r['context']) for r in examples}
    excluded = {r['group_id'] for r in examples if norm(r['entity']) in mentions or r['group_id'] in eval_groups}
    near = {}
    for group, words in example_shingles.items():
        match = next((g for g, target in target_shingles.items() if jaccard(words, target) >= .85), None)
        if match is not None: excluded.add(group); near[group] = match
    allowed = [r for r in examples if r['group_id'] not in excluded]
    destination = EXPERIMENT / 'examples.sqlite'
    if destination.exists(): raise RuntimeError('Partial selection exists; inspect before replacing')
    ex = sqlite3.connect(destination)
    ex.executescript("""CREATE TABLE examples(id INTEGER PRIMARY KEY,system TEXT,code TEXT,entity TEXT,context TEXT,input_id TEXT,group_id TEXT);
    CREATE VIRTUAL TABLE example_fts USING fts5(entity,context,content='examples',content_rowid='id',tokenize='porter unicode61 remove_diacritics 2');""")
    ex.executemany('insert into examples values(?,?,?,?,?,?,?)', [tuple(r[k] for k in ('id','system','code','entity','context','input_id','group_id')) for r in allowed])
    ex.execute("insert into example_fts(example_fts) values('rebuild')"); ex.commit(); ex.close()
    final_freq = {s: Counter(r['code'] for r in allowed if r['system'] == s) for s in SYSTEMS}
    for x in chosen:
        x['frequency'] = {s: final_freq[s][x['gold'][s]] for s in SYSTEMS}
        x['bucket'] = {s: bucket(x['frequency'][s]) for s in SYSTEMS}
        x['features'] = features(x)
        x['split'] = assignments[x['group_id']]
        assert all(db.execute('select 1 from concepts where system=? and code=?', (s,x['gold'][s])).fetchone() for s in SYSTEMS)
    db.close()
    max_similarity = max(jaccard(example_shingles[g], words) for g in {r['group_id'] for r in allowed} for words in target_shingles.values())
    audit = dict(shared_input_ids=len({r['input_id'] for r in allowed}&{r['input_id'] for r in chosen}),
                 shared_normalized_mentions=len({norm(r['entity']) for r in allowed}&mentions),
                 shared_context_groups=len({r['group_id'] for r in allowed}&eval_groups),
                 maximum_test_training_shingle_jaccard=max_similarity, near_duplicate_threshold=.85,
                 excluded_training_groups=len(excluded), near_duplicate_training_groups=len(near),
                 permitted_training_examples=len(allowed), previous_agent_input_overlap=len(previous_ids&{r['input_id'] for r in chosen}))
    assert audit['shared_input_ids']==audit['shared_normalized_mentions']==audit['shared_context_groups']==audit['previous_agent_input_overlap']==0
    assert max_similarity < .85
    manifest = dict(count=500, eligible_reserved_paired_inputs=len(pool), distinct_context_groups=len(eval_groups),
                    distinct_normalized_mentions=len(mentions), splits=dict(Counter(x['split'] for x in chosen)),
                    selection='Deterministic context-group coverage, code diversity, long-tail preference and domain coverage; no retrieval/model outcomes used.',
                    bucket_definition='Frequency in the exact filtered training-example store available to BOTH agents: unseen0, rare1-5, common6+.',
                    buckets={s: dict(Counter(x['bucket'][s] for x in chosen)) for s in SYSTEMS},
                    unique_codes={s:len({x['gold'][s] for x in chosen}) for s in SYSTEMS},
                    unique_codes_by_bucket={s:{b:len({x['gold'][s] for x in chosen if x['bucket'][s]==b}) for b in ('unseen','rare','common')} for s in SYSTEMS},
                    domains=dict(Counter(x['features']['domain'] for x in chosen)),
                    challenge_features={f:sum(x['features'][f] for x in chosen) for f in features(chosen[0]) if f!='domain'},
                    max_inputs_per_context=max(Counter(x['group_id'] for x in chosen).values()),
                    prior_agent_context_overlap_inputs=sum(x['group_id'] in previous_groups for x in chosen),
                    prior_agent_context_overlap_groups=len(eval_groups&previous_groups),
                    interpretation='Broader development evaluation. Inputs are new to agent evaluation, but some chart contexts and retrieval-benchmark labels have been inspected previously. Not untouched confirmation.',
                    source_sha256=file_hash(ROOT/'data/hackathon_train.csv'), examples_sha256=file_hash(destination), leakage_audit=audit)
    write_json(EXPERIMENT/'inputs.json',[{k:r[k] for k in ('input_id','entity','chunk_text')} for r in chosen])
    write_json(EXPERIMENT/'labels.json',chosen)
    write_json(EXPERIMENT/'training_code_frequencies.json',{s:dict(v) for s,v in final_freq.items()})
    write_json(EXPERIMENT/'leakage_audit.json',audit)
    write_json(EXPERIMENT/'selection_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


def evidence():
    assert (EXPERIMENT/'selection_manifest.json').exists()
    for n,x in enumerate(read_json(EXPERIMENT/'inputs.json'),1):
        name=x['input_id']+'.json'
        if all((EXPERIMENT/a/'initial_evidence'/name).exists() for a in ARMS): continue
        raw=PolicyTools(EXPERIMENT/'examples.sqlite')
        initial={s:raw.search_concepts(s,[x['entity']],12) for s in SYSTEMS}
        examples={s:raw.find_training_examples(x['entity'],s,2) for s in SYSTEMS}
        payload=dict(mention=x['entity'],context=context_window(x['entity'],x['chunk_text']),initial_candidates=initial,training_examples=examples)
        fallback={s:next((r['code'] for r in initial[s]['results'] if raw.valid(s,r['code'])), 'R69' if s=='ICD10CM' else '404684003') for s in SYSTEMS}
        record=dict(input_id=x['input_id'],payload=payload,seen={s:sorted(raw.seen[s]) for s in SYSTEMS},fallback=fallback)
        record['sha256']=digest(json.dumps(payload,sort_keys=True,ensure_ascii=False))
        assert_flat(payload)
        write_json(EXPERIMENT/ARMS[0]/'initial_evidence'/name,record)
        raw.close()
        dense=AblationTools(EXPERIMENT/'examples.sqlite','icd_only')
        new=copy.deepcopy(record)
        new['payload']['initial_candidates']['ICD10CM']=dense.search_concepts('ICD10CM',[x['entity']],12)
        for row in examples['ICD10CM']['examples']: dense.card('ICD10CM',row['code'])
        new['seen']['ICD10CM']=sorted(dense.seen['ICD10CM'])
        new['fallback']['ICD10CM']=next((r['code'] for r in new['payload']['initial_candidates']['ICD10CM']['results'] if dense.valid('ICD10CM',r['code'])),fallback['ICD10CM'])
        new['sha256']=digest(json.dumps(new['payload'],sort_keys=True,ensure_ascii=False))
        assert_flat(new['payload'])
        write_json(EXPERIMENT/ARMS[1]/'initial_evidence'/name,new)
        write_json(EXPERIMENT/ARMS[1]/'initial_pools'/name,dense.pools)
        dense.close()
        if n%25==0: print('PREPARED',n,'/500',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['select','evidence']);args=parser.parse_args()
    select() if args.command=='select' else evidence()
