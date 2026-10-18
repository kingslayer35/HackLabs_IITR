"""Training-only lexical context memory with explicit provenance, not ontology edges."""
from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

from .common import norm
from .tools import STOPWORDS, fts_query, query_terms

MODES = ('mention', 'context', 'contrast')
MAX_EXAMPLES = 4
MAX_EXAMPLE_BYTES = 1800
MAX_SYSTEM_BYTES = 6000
WINDOW = 320
NOISE = STOPWORDS | set('this that was were are have had been she he her his they their it its is as be an also at note patient patients diagnosis assessment plan history medical past'.split())
ANCHOR_MODIFIERS = set('left right bilateral acute chronic subacute mild moderate severe initial subsequent unspecified stage type grade history possible suspected resolved resolving'.split())
CUES = {
    'laterality': r'\b(?:left|right|bilateral)\b',
    'acuity': r'\b(?:acute|chronic|subacute)\b',
    'stage_type': r'\b(?:stage|type|grade)\s+[0-9ivx]+[a-d]?\b',
    'assertion_status': r'\b(?:no|not|without|denies|denied|suspected|possible|history of|family history|past medical history)\b|\bh/o\b',
    'severity': r'\b(?:mild|moderate|severe)\b',
    'intent': r'\b(?:intentional|accidental|self[ -]harm)\b',
    'description_detail': r'\b(?:simple|congenital|acquired|recurrent|initial|subsequent|unspecified)\b',
}


def local_window(mention, context, size=WINDOW):
    """Exact source substring; absence of an exact target match is recorded, not repaired."""
    position = context.casefold().find(mention.casefold())
    start = max(0, position - 100) if position >= 0 else 0
    return dict(text=context[start:start + max(size, len(mention) + 100)],
                source_start=start, mention_found=position >= 0)


def literal_cues(mention, context):
    result = []
    for field, text in [('mention', mention), ('context', context)]:
        for family, pattern in CUES.items():
            for match in re.finditer(pattern, text, re.I):
                result.append(dict(family=family, value=norm(match.group()), field=field,
                                   start=match.start(), end=match.end(), text=match.group()))
    return result


def content_terms(text, limit=32):
    return [w for w in dict.fromkeys(norm(text).split()) if w not in NOISE and (len(w) > 2 or w.isdigit())][:limit]


def build_index(source_path: Path, target_path: Path, dictionary_path: Path):
    if target_path.exists():
        raise ValueError('Preserve existing frozen memory index')
    source = sqlite3.connect(f'file:{source_path}?mode=ro', uri=True); source.row_factory = sqlite3.Row
    dictionary = sqlite3.connect(f'file:{dictionary_path}?mode=ro', uri=True); dictionary.row_factory = sqlite3.Row
    db = sqlite3.connect(target_path)
    db.executescript('''
      CREATE TABLE cases(id INTEGER PRIMARY KEY,system TEXT,code TEXT,entity TEXT,full_context TEXT,
        context TEXT,source_start INTEGER,mention_found INTEGER,input_id TEXT,group_id TEXT,
        description TEXT,tag TEXT,cues TEXT,description_cues TEXT);
      CREATE INDEX by_system_code ON cases(system,code);
      CREATE VIRTUAL TABLE mention_fts USING fts5(entity,full_context,content='cases',content_rowid='id',tokenize='porter unicode61 remove_diacritics 2');
      CREATE VIRTUAL TABLE context_fts USING fts5(context,content='cases',content_rowid='id',tokenize='porter unicode61 remove_diacritics 2');
    ''')
    count = 0
    for row in source.execute('select * from examples order by id'):
        card = dictionary.execute('select preferred,tag from concepts where system=? and code=?', (row['system'], row['code'])).fetchone()
        if card is None: raise ValueError('Training label absent from dictionary')
        window = local_window(row['entity'], row['context'])
        cues = literal_cues(row['entity'], window['text'])
        display_cues = literal_cues(card['preferred'], '')
        observed = {v['value'] for v in cues}
        not_literal = sorted({v['value'] for v in display_cues if v['value'] not in observed})
        db.execute('insert into cases values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (row['id'], row['system'], row['code'], row['entity'], row['context'], window['text'],
             window['source_start'], int(window['mention_found']), row['input_id'], row['group_id'],
             card['preferred'], card['tag'], json.dumps(cues), json.dumps(not_literal)))
        count += 1
    db.execute("insert into mention_fts(mention_fts) values('rebuild')")
    db.execute("insert into context_fts(context_fts) values('rebuild')")
    db.commit(); db.close(); source.close(); dictionary.close()
    return count


class ContextMemory:
    def __init__(self, path):
        self.db = sqlite3.connect(f'file:{path}?mode=ro', uri=True); self.db.row_factory = sqlite3.Row
        frequencies = {}; self.training_rows = 0
        for row in self.db.execute('select entity from cases'):
            self.training_rows += 1
            for token in set(content_terms(row['entity'])):
                frequencies[token] = frequencies.get(token, 0) + 1
        self.anchor_idf = {token: 1 + math.log((1+self.training_rows)/(1+count)) for token,count in frequencies.items()}

    def close(self): self.db.close()

    def _search(self, index, terms, system, codes, limit=100):
        if not terms or not codes: return []
        if index not in {'mention_fts', 'context_fts'}: raise ValueError('Unknown memory index')
        placeholders = ','.join('?' for _ in codes)
        weight = ',5.0,1.0' if index == 'mention_fts' else ''
        sql = f'''select c.*,bm25({index}{weight}) as relevance from {index}
          join cases c on c.id={index}.rowid where {index} match ? and c.system=?
          and c.code in ({placeholders}) order by relevance,c.id limit ?'''
        return [dict(r) for r in self.db.execute(sql, [fts_query(terms), system, *sorted(codes), limit])]

    @staticmethod
    def _format(row):
        cues = json.loads(row['cues'])
        # Same compact evidence representation in every condition. Cues are literal
        # occurrences only, not asserted target attributes or formal code requirements.
        unique = []; seen = set()
        for cue in cues:
            key = (cue['family'], cue['value'], cue['field'])
            if key not in seen: unique.append(cue); seen.add(key)
        value = dict(system=row['system'], mention=row['entity'], context=row['context'],
            code=row['code'], description=row['description'], semantic_tag=row['tag'],
            example_id=row['input_id'], source='filtered training split only',
            literal_cues=unique[:4], description_cues_not_literal=json.loads(row['description_cues']))
        if len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) > MAX_EXAMPLE_BYTES:
            return None
        return value

    def retrieve(self, mention, context, system, codes, mode):
        if mode not in MODES: raise ValueError('Unknown memory mode')
        codes = set(codes)
        window = local_window(mention, context)['text']
        mention_results = self._search('mention_fts', query_terms(mention), system, codes)
        context_results = self._search('context_fts', content_terms(mention + ' ' + window), system, codes)
        rows = {r['id']: r for r in mention_results + context_results}
        scores = {}; mention_rank = {r['id']: i for i, r in enumerate(mention_results)}
        context_rank = {r['id']: i for i, r in enumerate(context_results)}
        if mode == 'mention':
            order = [r['id'] for r in mention_results]
        else:
            target_cues = {(c['family'], c['value']) for c in literal_cues(mention, window)}
            target_by_family = {}
            for family, value in target_cues: target_by_family.setdefault(family, set()).add(value)
            for key, row in rows.items():
                score = (2 / (20 + mention_rank[key]) if key in mention_rank else 0)
                score += 1 / (20 + context_rank[key]) if key in context_rank else 0
                observed = {(c['family'], c['value']) for c in json.loads(row['cues'])}
                agreement = len(observed & target_cues)
                conflicts = sum(f in target_by_family and value not in target_by_family[f] for f, value in observed)
                score += .01 * min(2, agreement) - .005 * min(2, conflicts)
                # Full note context can match a neighboring condition. Reward overlap
                # with the annotated mention itself, excluding generic modifiers.
                anchors = {w for w in content_terms(mention) if w not in ANCHOR_MODIFIERS and not w.isdigit()}
                if anchors:
                    weight = lambda w: self.anchor_idf.get(w, 1+math.log(1+self.training_rows))
                    matched = anchors & set(content_terms(row['entity']))
                    score += .08 * sum(weight(w) for w in matched)/sum(weight(w) for w in anchors)
                scores[key] = score
            order = sorted(rows, key=lambda key: (-scores[key], key))
            if mode == 'contrast' and order:
                first = order[0]; first_code = rows[first]['code']
                anchors = set(content_terms(mention))
                alternatives = [key for key in order[1:] if rows[key]['code'] != first_code
                    and scores[key] >= .6 * scores[first]
                    and anchors & set(content_terms(rows[key]['entity']))]
                if alternatives:
                    alternative = alternatives[0]
                    order = [first, alternative] + [key for key in order[1:] if key != alternative]
        examples = []; used_bytes = 2; seen_mentions = set(); selected = []
        for key in order:
            row = rows[key]; value = self._format(row)
            if value is None: continue
            normalized = norm(row['entity'])
            if normalized in seen_mentions: continue
            size = len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) + int(bool(examples))
            if used_bytes + size > MAX_SYSTEM_BYTES: continue
            assert row['code'] in codes
            examples.append(value); selected.append(key); used_bytes += size; seen_mentions.add(normalized)
            if len(examples) == MAX_EXAMPLES: break
        return dict(examples=examples, selected_row_ids=selected, serialized_bytes=used_bytes,
            ranked_row_ids=order, scores={str(key): round(score, 8) for key, score in scores.items()},
            available_mention_hits=len(mention_results), available_context_hits=len(context_results))
