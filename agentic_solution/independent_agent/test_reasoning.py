"""Offline checks for the reasoning-only manipulation and pilot selection."""
import copy
import unittest

from .reasoning_eval import body, reasoning_tokens, select
from .test_selector500 import fixture


class ReasoningProtocolTests(unittest.TestCase):
    def test_only_reasoning_differs(self):
        a = body(fixture(), 'off'); b = body(fixture(), 'high')
        self.assertEqual(a.pop('reasoning'), {'enabled': False})
        self.assertEqual(b.pop('reasoning'), {'effort': 'high'})
        self.assertEqual(a, b)
        self.assertEqual(a['max_tokens'], 8192)
        self.assertFalse(a['provider']['allow_fallbacks'])

    def test_same_guide_and_schema(self):
        for arm in ['off', 'high']:
            value = body(fixture(), arm)
            self.assertIn('Coding and task guide:', value['messages'][0]['content'])
            self.assertEqual(value['response_format']['json_schema']['schema']['properties']['snomed_code']['enum'], ['1', '2'])
            self.assertNotIn('tools', value)

    def test_reasoning_usage_missing_is_not_zero(self):
        self.assertIsNone(reasoning_tokens({'usage': {}}))
        self.assertEqual(reasoning_tokens({'usage': {'completion_tokens_details': {'reasoning_tokens': 0}}}), 0)
        self.assertEqual(reasoning_tokens({'usage': {'completion_tokens_details': {'reasoning_tokens': 77}}}), 77)

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError): body(fixture(), 'max')

    def test_selection_deterministic_and_ignores_outcomes(self):
        rows = [dict(input_id=f'{i:020x}', group_id=f'group{i//2}', gold={'ICD10CM': str(i % 13), 'SNOMEDCT_US': str(i % 19)},
                     bucket={'ICD10CM': ['unseen', 'rare', 'common'][i % 3], 'SNOMEDCT_US': ['unseen', 'rare', 'common'][i % 3]},
                     features={'domain': str(i % 4)}) for i in range(60)]
        expected = select(rows, 20)
        altered = copy.deepcopy(rows)
        for x in altered: x['prior_correct'] = False; x['prediction'] = 'irrelevant'
        actual = select(altered, 20)
        self.assertEqual([x['input_id'] for x in expected], [x['input_id'] for x in actual])
        self.assertEqual(len({x['group_id'] for x in expected}), 20)
        self.assertEqual(len({x['features']['domain'] for x in expected}), 4)
        self.assertEqual([x['input_id'] for x in select(list(reversed(rows)), 20)], [x['input_id'] for x in expected])


if __name__ == '__main__': unittest.main()
