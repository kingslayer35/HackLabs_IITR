"""Routing and decision-preservation checks; no API requests."""
import copy
import unittest

from .terminal_rescue import (needs_rescue, terminal_request, finalize_after_budget,
                              MODEL, PROVIDER)
from .test_selector500 import fixture


def run_fixture(status='fallback_budget_or_invalid'):
    return dict(input_id='case', status=status, model_calls=5, trace=[{} for _ in range(5)],
        prediction=dict(icd10cm_code='N28.1', snomed_code='1', confidence=0., evidence='Original decision.'))


class TerminalRescueTests(unittest.TestCase):
    def test_accepted_decision_preserved_without_calling_model(self):
        original = run_fixture('model'); before = copy.deepcopy(original)
        result = finalize_after_budget(original, None, lambda _: self.fail('Must not call model'))
        self.assertEqual(result['prediction'], original['prediction'])
        self.assertFalse(result['rescue_attempted']); self.assertEqual(result['total_model_calls'], 5)
        self.assertEqual(original, before)

    def test_error_is_not_misrouted_as_budget_failure(self):
        with self.assertRaises(ValueError): needs_rescue(run_fixture('fallback_error'))

    def test_unexhausted_budget_is_not_eligible(self):
        original = run_fixture(); original['model_calls'] = 4
        with self.assertRaises(ValueError): needs_rescue(original)

    def test_one_extra_call_only_and_fresh_valid_prediction(self):
        original = run_fixture(); calls = []
        def complete(body):
            calls.append(body)
            return dict(id='new-generation', model=MODEL, provider=PROVIDER, usage={'cost': .001}, choices=[dict(
                finish_reason='stop', message=dict(content='{"icd10cm_code":"N28.1","snomed_code":"2","confidence":0.6,"evidence":"Target evidence."}'))])
        result = finalize_after_budget(original, fixture(), complete)
        self.assertEqual(len(calls), 1); self.assertEqual(result['total_model_calls'], 6)
        self.assertTrue(result['accepted_terminal']); self.assertEqual(result['prediction']['snomed_code'], '2')
        self.assertEqual(original['prediction']['snomed_code'], '1')

    def test_invalid_output_keeps_original_fallback(self):
        original = run_fixture()
        response = dict(id='new-generation', model=MODEL, provider=PROVIDER, usage={'cost': .001}, choices=[dict(finish_reason='length', message={'content': ''})])
        result = finalize_after_budget(original, fixture(), lambda _: response)
        self.assertFalse(result['accepted_terminal']); self.assertEqual(result['status'], 'retained_fallback')
        self.assertEqual(result['prediction'], original['prediction'])

    def test_unseen_identifier_is_not_accepted(self):
        response = dict(id='new-generation', model=MODEL, provider=PROVIDER, usage={'cost': .001}, choices=[dict(
            finish_reason='stop', message={'content': '{"icd10cm_code":"N28.1","snomed_code":"outside","confidence":0.6,"evidence":"Target evidence."}'})])
        result = finalize_after_budget(run_fixture(), fixture(), lambda _: response)
        self.assertEqual(result['status'], 'retained_fallback')

    def test_wrong_evidence_input_is_rejected(self):
        prepared = fixture(); prepared['input_id'] = 'another'
        with self.assertRaises(ValueError): finalize_after_budget(run_fixture(), prepared, lambda _: self.fail('No request'))

    def test_terminal_request_is_guided_nonreasoning_without_tools(self):
        value = terminal_request(fixture())
        self.assertEqual(value['reasoning'], {'enabled': False}); self.assertEqual(value['max_tokens'], 2048)
        self.assertIn('Coding and task guide:', value['messages'][0]['content'])
        self.assertNotIn('tools', value); self.assertFalse(value['provider']['allow_fallbacks'])
        self.assertNotIn('Original decision.', value['messages'][1]['content'])


if __name__ == '__main__': unittest.main()
