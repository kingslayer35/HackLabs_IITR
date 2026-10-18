import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import json

from . import backend_ablation as e
from .backend_ablation import AblationTools, SOURCE, schemas
from .common import SYSTEMS, read_json, write_json
from .comparisons import PolicyTools, assert_flat


class BackendAblationTests(unittest.TestCase):
    def setUp(self):
        self.name = sorted((SOURCE / 'initial_evidence').glob('*.json'))[0].name
        self.evidence = read_json(SOURCE / 'initial_evidence' / self.name)
        self.pools = read_json(SOURCE / 'initial_pools' / self.name)

    def tools(self, arm):
        tools = AblationTools(SOURCE / 'examples.sqlite', arm,
            fetch=lambda *args: self.fail('Unexpected retrieval request'))
        tools.pools = copy.deepcopy(self.pools)
        tools.seen = {s: set(self.evidence['seen'][s]) for s in SYSTEMS}
        self.addCleanup(tools.close)
        return tools

    def candidate(self):
        return dict(icd10cm_code=self.evidence['fallback']['ICD10CM'],
                    snomed_code=self.evidence['fallback']['SNOMEDCT_US'], confidence=.7, evidence='test')

    def test_icd_only_snomed_exactly_matches_original_search(self):
        tools = self.tools('icd_only')
        control = PolicyTools(SOURCE / 'examples.sqlite')
        self.addCleanup(control.close)
        self.assertEqual(tools.search_concepts('SNOMEDCT_US', ['heart failure'], 12),
                         control.search_concepts('SNOMEDCT_US', ['heart failure'], 12))

    def test_icd_dense_page_uses_existing_pool(self):
        tools = self.tools('icd_only')
        key, pool = next((k, p) for k, p in self.pools.items() if p['system'] == 'ICD10CM')
        page = tools.search_concepts('ICD10CM', pool['queries'], 12)
        self.assertEqual(page['search_id'], key)
        self.assertEqual([r['code'] for r in page['results']], [r['code'] for r in pool['results'][:12]])
        assert_flat(page)

    def test_review_is_label_blind_once_and_only_snomed(self):
        tools = self.tools('selective_snomed')
        before = copy.deepcopy(tools.seen)
        review = tools.maybe_review(self.candidate(), 2)
        self.assertTrue(review['review_required'])
        self.assertEqual(review['system'], 'SNOMEDCT_US')
        self.assertLessEqual(len(review['results']), 12)
        self.assertEqual(tools.seen['ICD10CM'], before['ICD10CM'])
        self.assertTrue(any(r['code'] not in before['SNOMEDCT_US'] for r in review['results']))
        self.assertIsNone(tools.maybe_review(self.candidate(), 3))
        self.assertEqual(tools.review_turn, 2)
        assert_flat(review)

    def test_review_does_not_activate_for_high_confidence_or_late_finish(self):
        for confidence, turn in ((.85, 1), (.95, 3), (.7, 4), (.7, 5)):
            with self.subTest(confidence=confidence, turn=turn):
                tools = self.tools('selective_snomed')
                candidate = self.candidate()
                candidate['confidence'] = confidence
                before = copy.deepcopy(tools.seen)
                self.assertIsNone(tools.maybe_review(candidate, turn))
                self.assertEqual(tools.seen, before)

    def test_icd_only_never_runs_review(self):
        tools = self.tools('icd_only')
        self.assertIsNone(tools.maybe_review(self.candidate(), 1))

    def test_both_final_menus_are_finish_only(self):
        for arm in ('icd_only', 'selective_snomed'):
            self.assertEqual([s['function']['name'] for s in schemas(arm, True)], ['finish'])

    def test_runner_requires_response_after_review_and_hides_labels(self):
        evidence, pools = copy.deepcopy(self.evidence), copy.deepcopy(self.pools)
        calls = []
        proposal = self.candidate()
        class FakeClient:
            key = 'unused-test-key'
            def __init__(self, *args):
                pass
            def complete(self, messages, tools, force_finish=False):
                calls.append(copy.deepcopy(messages))
                candidate = copy.deepcopy(proposal)
                # Two finishes in one response cannot bypass reading the new page.
                items = [dict(id='f' + str(i), type='function',
                              function=dict(name='finish', arguments=json.dumps(candidate)))
                         for i in range(2 if len(calls) == 1 else 1)]
                return dict(choices=[dict(message=dict(role='assistant', tool_calls=items))],
                            usage={'cost': 0}, id='test' + str(len(calls))), []
        with tempfile.TemporaryDirectory() as directory, patch.object(e, 'EXPERIMENT', Path(directory)):
            p = Path(directory)
            input_id = evidence['input_id']
            write_json(p / 'selective_snomed/initial_evidence' / (input_id + '.json'), evidence)
            write_json(p / 'selective_snomed/initial_pools' / (input_id + '.json'), pools)
            write_json(p / 'config.json', {'providers': {input_id: 'DeepInfra'}})
            item = dict(input_id=input_id, entity=evidence['payload']['mention'], chunk_text='test context')
            result = e.run_one(item, 'selective_snomed', None, client_factory=FakeClient,
                               fetch=lambda *a: self.fail('Unexpected search'))
            self.assertEqual(result['status'], 'model')
            self.assertEqual(result['model_calls'], 2)
            self.assertTrue(result['trace'][0]['tool_calls'][0]['result']['review_required'])
            self.assertIn('error', result['trace'][0]['tool_calls'][1]['result'])
            self.assertTrue(result['trace'][1]['tool_calls'][0]['result']['accepted'])
            with self.assertRaises(ValueError):
                e.run_one(dict(item, gold={}), 'selective_snomed', None)


if __name__ == '__main__':
    unittest.main()
