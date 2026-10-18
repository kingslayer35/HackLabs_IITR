import copy
import json
import unittest
from unittest.mock import patch

from . import diverse500_eval as e
from .common import SYSTEMS,read_json
from .diverse500_prepare import bucket,choose,domain


class Diverse500Tests(unittest.TestCase):
    def test_bucket_boundaries_and_domain_splits(self):
        self.assertEqual([bucket(n) for n in (0,1,5,6)],['unseen','rare','rare','common'])
        self.assertEqual([domain(c) for c in ('D10','D64','H10','H65')],['neoplasms','blood_immune','eye','ear'])

    def test_selector_covers_groups_and_codes_deterministically(self):
        pool=[dict(input_id=str(i),group_id=str(i//2),gold={'ICD10CM':c,'SNOMEDCT_US':str(i)}) for i,c in enumerate(['I10','I10','J10','J11','K10','K11'])]
        from collections import Counter
        frequency={s:Counter() for s in SYSTEMS}
        a=choose(pool,frequency,3);b=choose(list(reversed(pool)),frequency,3)
        self.assertEqual([x['input_id'] for x in a],[x['input_id'] for x in b])
        self.assertEqual(len({x['group_id'] for x in a}),3)

    def test_final_menus_and_original_investigation_tools(self):
        for arm in e.ARMS:self.assertEqual([x['function']['name'] for x in e.schemas(arm,True)],['finish'])
        self.assertEqual({x['function']['name'] for x in e.schemas(e.ARMS[0])},
                         {'search_concepts','inspect_concepts','find_training_examples','finish'})

    def test_runtime_rejects_gold_fields(self):
        with self.assertRaises(ValueError):e.run_one({'input_id':'x','entity':'test','chunk_text':'','gold':{}},e.ARMS[0],None)

    def test_paid_prefix_replay_never_resends_saved_response(self):
        file=next((e.EXPERIMENT/e.ARMS[0]/'initial_evidence').glob('*.json'))
        evidence=read_json(file);item=dict(input_id=file.stem,entity=evidence['payload']['mention'],chunk_text='test')
        prediction=dict(icd10cm_code=evidence['fallback']['ICD10CM'],snomed_code=evidence['fallback']['SNOMEDCT_US'],confidence=.7,evidence='test')
        def response(name,arguments,index):
            return dict(id='test'+str(index),usage={'cost':0},provider=e.PROVIDER,
                choices=[dict(message=dict(role='assistant',tool_calls=[dict(id='t'+str(index),type='function',function=dict(name=name,arguments=json.dumps(arguments)))]))])
        seen=[]
        class FailingClient:
            key='unused-test-key'
            def __init__(self,*args):self.calls=0
            def complete(self,*args,**kwargs):
                self.calls+=1
                if self.calls==2:raise RuntimeError('simulated transport failure')
                return response('inspect_concepts',dict(system='ICD10CM',codes=[prediction['icd10cm_code']]),1),[]
        class ResumeClient:
            key='unused-test-key'
            def __init__(self,*args):pass
            def complete(self,messages,*args,**kwargs):
                seen.append(copy.deepcopy(messages));return response('finish',prediction,2),[]
        partial=e.run_one(item,e.ARMS[0],None,client_factory=FailingClient)
        self.assertEqual(partial['status'],'fallback_error');self.assertEqual(len(partial['responses']),1)
        resumed=e.run_one(item,e.ARMS[0],None,client_factory=ResumeClient,prior=partial)
        self.assertEqual(resumed['status'],'model');self.assertEqual(resumed['model_calls'],2)
        self.assertEqual(resumed['replayed_responses'],1);self.assertEqual(len(seen),1)


if __name__=='__main__':unittest.main()
