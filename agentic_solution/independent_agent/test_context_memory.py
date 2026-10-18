"""Offline retrieval and provenance checks, using a miniature training store."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from .context_memory import (ContextMemory, build_index, literal_cues, local_window,
                             MODES, MAX_EXAMPLES, MAX_SYSTEM_BYTES)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        source = sqlite3.connect(self.root / 'train.sqlite')
        source.execute('create table examples(id integer primary key,system text,code text,entity text,context text,input_id text,group_id text)')
        rows = [
            (1, 'SNOMEDCT_US', 'a', 'renal cyst', 'A renal cyst is present.', 'one', 'g1'),
            (2, 'SNOMEDCT_US', 'b', 'simple renal cyst', 'No simple renal cyst on the left.', 'two', 'g2'),
            (3, 'SNOMEDCT_US', 'a', 'kidney cyst', 'Chronic kidney cyst in the right kidney.', 'three', 'g3'),
            (4, 'SNOMEDCT_US', 'outside', 'renal cyst with fever', 'A renal cyst with fever.', 'four', 'g4'),
            (5, 'ICD10CM', 'N28.1', 'renal cyst', 'A renal cyst is present.', 'five', 'g5')]
        source.executemany('insert into examples values(?,?,?,?,?,?,?)', rows); source.commit(); source.close()
        dictionary = sqlite3.connect(self.root / 'dictionary.sqlite')
        dictionary.execute('create table concepts(system text,code text,preferred text,tag text)')
        dictionary.executemany('insert into concepts values(?,?,?,?)', [
            ('SNOMEDCT_US', 'a', 'Renal cyst', 'disorder'), ('SNOMEDCT_US', 'b', 'Simple renal cyst', 'disorder'),
            ('SNOMEDCT_US', 'outside', 'Renal cyst with fever', 'disorder'), ('ICD10CM', 'N28.1', 'Acquired kidney cyst', None)])
        dictionary.commit(); dictionary.close()
        self.assertEqual(build_index(self.root/'train.sqlite', self.root/'memory.sqlite', self.root/'dictionary.sqlite'), 5)
        self.memory = ContextMemory(self.root/'memory.sqlite')

    def tearDown(self): self.memory.close(); self.tmp.cleanup()

    def test_never_adds_codes_outside_pool_or_cross_system(self):
        for mode in MODES:
            value = self.memory.retrieve('renal cyst', 'No simple renal cyst on the left.', 'SNOMEDCT_US', {'a', 'b'}, mode)
            self.assertTrue(value['examples'])
            self.assertTrue(all(e['code'] in {'a', 'b'} and e['system']=='SNOMEDCT_US' for e in value['examples']))
            self.assertLessEqual(len(value['examples']), MAX_EXAMPLES)
            self.assertLessEqual(value['serialized_bytes'], MAX_SYSTEM_BYTES)

    def test_index_preserves_training_label_without_inventing_support(self):
        row = self.memory.db.execute("select * from cases where input_id='five'").fetchone()
        self.assertEqual(row['code'], 'N28.1')
        self.assertIn('acquired', json.loads(row['description_cues']))
        self.assertFalse(any(c['value']=='acquired' for c in json.loads(row['cues'])))

    def test_literal_cue_offsets_do_not_turn_negation_into_assertion(self):
        mention, context = 'simple renal cyst', 'No simple renal cyst on the left.'
        cues = literal_cues(mention, context)
        self.assertTrue(any(c['value']=='no' for c in cues))
        for cue in cues:
            field = mention if cue['field']=='mention' else context
            self.assertEqual(field[cue['start']:cue['end']], cue['text'])
            self.assertNotIn('asserted', cue)

    def test_exact_context_window_provenance_and_missing_target(self):
        text = 'prefix '*100 + 'renal cyst in left kidney.'
        value = local_window('renal cyst', text)
        self.assertTrue(value['mention_found'])
        self.assertEqual(text[value['source_start']:value['source_start']+len(value['text'])], value['text'])
        missing = local_window('unmatched', text)
        self.assertFalse(missing['mention_found']); self.assertEqual(missing['source_start'], 0)

    def test_deterministic_selection_and_no_fabricated_examples(self):
        for mode in MODES:
            a = self.memory.retrieve('renal cyst', 'No simple renal cyst on the left.', 'SNOMEDCT_US', {'a','b'}, mode)
            b = self.memory.retrieve('renal cyst', 'No simple renal cyst on the left.', 'SNOMEDCT_US', {'b','a'}, mode)
            self.assertEqual(a, b)
            self.assertTrue(all(e['example_id'] in {'one','two','three'} for e in a['examples']))

    def test_empty_pool_and_unknown_mode(self):
        self.assertEqual(self.memory.retrieve('renal cyst', '', 'SNOMEDCT_US', set(), 'context')['examples'], [])
        with self.assertRaises(ValueError): self.memory.retrieve('renal cyst', '', 'SNOMEDCT_US', {'a'}, 'unknown')

    def test_context_retains_the_matching_annotated_mention(self):
        value = self.memory.retrieve('simple renal cyst', 'No simple renal cyst on the left.', 'SNOMEDCT_US', {'a','b'}, 'context')
        self.assertEqual(value['examples'][0]['example_id'], 'two')

    def test_existing_index_not_overwritten(self):
        with self.assertRaises(ValueError): build_index(self.root/'train.sqlite', self.root/'memory.sqlite', self.root/'dictionary.sqlite')


if __name__ == '__main__': unittest.main()
