"""Matched request checks for memory-only ablations; no model requests."""
import copy
import json
import unittest

from .memory_eval import body, MEMORY_NOTE
from .test_selector500 import fixture


class MemoryRequestTests(unittest.TestCase):
    def prepared(self):
        base=fixture()
        return dict(input_id='case',base=base,examples={m:{'ICD10CM':[], 'SNOMEDCT_US':[dict(code='1',mention=m)]} for m in ['mention','context','contrast']},diagnostics={})

    def test_requests_differ_only_in_examples(self):
        prepared=self.prepared();before=copy.deepcopy(prepared);normalized=[]
        for mode in ['mention','context','contrast']:
            request=body(prepared,mode);payload=json.loads(request['messages'][1]['content'])
            self.assertEqual(payload.pop('training_examples'),prepared['examples'][mode])
            request['messages'][1]['content']=json.dumps(payload,sort_keys=True)
            normalized.append(request)
            self.assertIn(MEMORY_NOTE,request['messages'][0]['content'])
        self.assertEqual(normalized[0],normalized[1]);self.assertEqual(normalized[1],normalized[2])
        self.assertEqual(prepared,before)

    def test_same_reasoning_budget_and_candidate_schema(self):
        for mode in ['mention','context','contrast']:
            request=body(self.prepared(),mode)
            self.assertEqual(request['max_tokens'],2048)
            self.assertEqual(request['reasoning'],{'enabled':False})
            self.assertFalse(request['provider']['allow_fallbacks'])
            self.assertEqual(request['response_format']['json_schema']['schema']['properties']['snomed_code']['enum'],['1','2'])

    def test_diagnostics_and_eval_metadata_not_sent(self):
        prepared=self.prepared();prepared['diagnostics']={'gold':'secret-eval-field','scores':[99]}
        serialized=json.dumps(body(prepared,'context'))
        self.assertNotIn('secret-eval-field',serialized)
        self.assertNotIn('"scores"',serialized)

    def test_unknown_condition_rejected(self):
        with self.assertRaises(ValueError):body(self.prepared(),'semantic_graph')


if __name__=='__main__':unittest.main()
