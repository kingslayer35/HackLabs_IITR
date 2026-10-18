"""Semantic vector math, candidate masking, matched requests and canonical caching."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .semantic_memory import average_pool_normalize, encoder_text, SemanticMemory
from .semantic_memory_eval import body, cached_response
from .test_context_memory import MemoryTests
from .test_selector500 import fixture


class SemanticTests(unittest.TestCase):
    def test_masked_mean_and_unit_norm(self):
        hidden=np.array([[[1,0],[0,1],[1000,1000]]],dtype=np.float32)
        value=average_pool_normalize(hidden,np.array([[1,1,0]]))
        self.assertTrue(np.allclose(value,[[2**-.5,2**-.5]],atol=1e-6))

    def test_query_prefix_and_target_first_context(self):
        self.assertEqual(encoder_text('renal cyst'),'query: renal cyst')
        self.assertEqual(encoder_text('renal cyst','left kidney'),'query: Mention: renal cyst\nContext: left kidney')

    def test_only_examples_change_in_selector_requests(self):
        value=dict(base=fixture(),examples={'lexical':{'ICD10CM':[],'SNOMEDCT_US':[{'example_id':'one'}]},
                                          'semantic':{'ICD10CM':[],'SNOMEDCT_US':[{'example_id':'two'}]}},diagnostics={})
        a=body(value,'lexical');b=body(value,'semantic')
        for request in [a,b]:
            payload=json.loads(request['messages'][1]['content']);payload.pop('training_examples')
            request['messages'][1]['content']=json.dumps(payload,sort_keys=True)
        self.assertEqual(a,b)
        self.assertEqual(a['reasoning'],{'enabled':False});self.assertEqual(a['max_tokens'],2048)

    def test_fixed_cache_priority_does_not_choose_by_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode,correct in [('context',False),('mention',True),('contrast',True)]:
                path=root/'responses'/mode/'case.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'request_sha256':'same','prior_correct':correct,'mode':mode}))
            with patch('independent_agent.semantic_memory_eval.MEMORY',root):
                path,value=cached_response('case','same')
                self.assertEqual(value['mode'],'context');self.assertFalse(value['prior_correct'])
                self.assertIsNone(cached_response('case','other')[0])

    def test_semantic_ranking_uses_context_and_masks_codes(self):
        helper=MemoryTests(methodName='test_empty_pool_and_unknown_mode');helper.setUp()
        try:
            root=helper.root
            tm=np.array([[1,0],[.9,.1],[.8,.2],[1,0],[1,0]],dtype=np.float32)
            tc=np.array([[1,0],[1,0],[0,1],[1,0],[1,0]],dtype=np.float32)
            np.savez_compressed(root/'embeddings.npz',train_mention=tm,train_context=tc,
                query_mention=np.array([[1,0],[1,0]],dtype=np.float32),query_context=np.array([[1,0],[0,1]],dtype=np.float32))
            (root/'embedding_inputs.json').write_text(json.dumps({'training_row_ids':[1,2,3,4,5],'query_input_ids':['q1','q2']}))
            memory=SemanticMemory(root,root/'memory.sqlite')
            try:
                a=memory.retrieve('q1','SNOMEDCT_US',{'a','b'});b=memory.retrieve('q2','SNOMEDCT_US',{'a','b'})
                self.assertEqual(a['selected_row_ids'][0],1);self.assertEqual(b['selected_row_ids'][0],3)
                self.assertTrue(all(e['code'] in {'a','b'} and e['system']=='SNOMEDCT_US' for e in b['examples']))
                self.assertEqual(memory.retrieve('q1','SNOMEDCT_US',set())['examples'],[])
            finally:memory.close()
        finally:helper.tearDown()


if __name__=='__main__':unittest.main()
