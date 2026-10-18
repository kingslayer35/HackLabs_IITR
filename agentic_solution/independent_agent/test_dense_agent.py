import json
import sqlite3
import unittest

from . import dense_agent_eval as evaluation
from .agent import validate_finish
from .common import ARTIFACTS,read_json
from .comparisons import ReservedBudget,TASK_PROMPT
from .recovery_eval import BASE_POLICY
from .dense_agent_tools import DenseAgentTools,agent_schemas


class DenseToolTests(unittest.TestCase):
    def setUp(self):
        db=sqlite3.connect(f'file:{ARTIFACTS / "terminology.sqlite"}?mode=ro',uri=True)
        self.codes=[r[0] for r in db.execute("SELECT code FROM concepts WHERE system='ICD10CM' AND code NOT LIKE '%-%' ORDER BY code LIMIT 30")];db.close()
        self.fetches=0
        def fetch(system,queries):
            self.fetches+=1
            return dict(results=[dict(code=c,matched_alias='fixture',retrieval_score=1/(20+i)) for i,c in enumerate(self.codes)])
        self.tools=DenseAgentTools(evaluation.SOURCE/'examples.sqlite',fetch=fetch)

    def tearDown(self):self.tools.close()

    def test_hidden_pool_not_seen_until_paged_and_paging_reuses_search(self):
        first=self.tools.search_concepts('ICD10CM',['fixture'])
        self.assertEqual(len(first['results']),12);self.assertEqual(first['next_offset'],12)
        self.assertNotIn(self.codes[12],self.tools.seen['ICD10CM'])
        second=self.tools.get_more_candidates(first['search_id'],12)
        self.assertEqual(second['results'][0]['code'],self.codes[12]);self.assertIn(self.codes[12],self.tools.seen['ICD10CM'])
        last=self.tools.get_more_candidates(first['search_id'],24)
        self.assertEqual(len(last['results']),6);self.assertFalse(last['has_more']);self.assertIsNone(last['next_offset'])
        self.assertEqual(self.fetches,1)

    def test_invalid_pages_and_cross_input_search_ids_rejected(self):
        first=self.tools.search_concepts('ICD10CM',['fixture'])
        for offset in (-1,30,True):
            with self.assertRaises(ValueError):self.tools.get_more_candidates(first['search_id'],offset)
        with self.assertRaises(ValueError):self.tools.get_more_candidates('another-input',0)
        with self.assertRaises(ValueError):self.tools.get_more_candidates(first['search_id'],12,13)

    def test_hidden_code_not_valid_final_until_exposed(self):
        self.tools.search_concepts('ICD10CM',['fixture'])
        self.tools.card('SNOMEDCT_US','404684003')
        candidate=dict(icd10cm_code=self.codes[12],snomed_code='404684003',confidence=.5,evidence='fixture')
        self.assertIn('not been retrieved',validate_finish(candidate,self.tools))

    def test_budget_and_disabled_tools(self):
        self.assertEqual(evaluation.MAX_TURNS,5)
        self.assertEqual([s['function']['name'] for s in agent_schemas(True)],['finish'])
        names={s['function']['name'] for s in agent_schemas()}
        self.assertIn('get_more_candidates',names);self.assertNotIn('get_related_concepts',names);self.assertNotIn('recover_queries',names)


class DenseRunnerTests(unittest.TestCase):
    def test_finish_must_read_new_page_before_selecting_from_it(self):
        item=read_json(evaluation.EXPERIMENT/'inputs.json')[0]
        evidence=read_json(evaluation.EXPERIMENT/'initial_evidence'/(item['input_id']+'.json'))
        pools=read_json(evaluation.EXPERIMENT/'initial_pools'/(item['input_id']+'.json'))
        search_id,pool=next((k,v) for k,v in pools.items() if v['system']=='ICD10CM')
        check=DenseAgentTools(evaluation.SOURCE/'examples.sqlite')
        index=next(i for i,r in enumerate(pool['results']) if i>=12 and r['code'] not in evidence['seen']['ICD10CM'] and check.valid('ICD10CM',r['code']))
        sn=next(r['code'] for r in evidence['payload']['initial_candidates']['SNOMEDCT_US']['results'] if check.valid('SNOMEDCT_US',r['code']));check.close()
        final=dict(icd10cm_code=pool['results'][index]['code'],snomed_code=sn,confidence=.5,evidence='Offline fixture')
        testcase=self
        class FakeClient:
            key='fixture'
            def __init__(self,*args):self.n=0
            def complete(self,messages,schemas,**kwargs):
                testcase.assertEqual(messages[0]['content'],TASK_PROMPT+'\n'+BASE_POLICY)
                self.n+=1
                calls=[]
                if self.n==1:calls.append(dict(id='page',type='function',function=dict(name='get_more_candidates',arguments=json.dumps(dict(search_id=search_id,offset=index,limit=1)))))
                calls.append(dict(id='final'+str(self.n),type='function',function=dict(name='finish',arguments=json.dumps(final))))
                return dict(choices=[dict(finish_reason='tool_calls',message=dict(role='assistant',tool_calls=calls))],usage={'cost':0}),[]
        result=evaluation.run_one(item,ReservedBudget(.01),client_factory=FakeClient)
        self.assertEqual(result['status'],'model');self.assertEqual(result['model_calls'],2)
        self.assertIn('read new results first',result['trace'][0]['tool_calls'][1]['result']['error'])
        self.assertEqual(result['prediction'],final)

    def test_labels_are_rejected_and_baseline_retained(self):
        item=read_json(evaluation.EXPERIMENT/'inputs.json')[0]
        with self.assertRaises(ValueError):evaluation.run_one({**item,'gold':{}},ReservedBudget(.01))
        retained=read_json(evaluation.EXPERIMENT/'retained_baseline.json')
        for relative,expected in retained.items():
            self.assertEqual(evaluation.file_hash(evaluation.SOURCE/relative),expected)
            if not relative.startswith('initial_evidence/'):
                self.assertEqual(evaluation.file_hash(evaluation.EXPERIMENT/relative),expected)


if __name__=='__main__':unittest.main()
