"""Selection protocol checks; no paid calls."""
import copy
import json
import unittest

from .selector500_eval import (ARMS, BASE, GUIDE, build_messages, distinctions,
                               reconstruct, request_body, response_format, validate)


def fixture():
    return dict(input_id='case', payload=dict(mention='renal cyst', context='No simple renal cyst was seen.',
        candidates={
            'ICD10CM': [dict(code='N28.1', preferred='Cyst of kidney, acquired', tag=None)],
            'SNOMEDCT_US': [dict(code='1', preferred='Renal cyst', tag='disorder'),
                            dict(code='2', preferred='Simple renal cyst', tag='disorder')]},
        training_examples={'ICD10CM': [], 'SNOMEDCT_US': []}),
        fallback={'ICD10CM': 'N28.1', 'SNOMEDCT_US': '1'}, distinctions={})


class ProtocolTests(unittest.TestCase):
    def test_arms_preserve_same_base_evidence(self):
        source = fixture(); before = copy.deepcopy(source)
        payloads = [json.loads(build_messages(source, a)[1]['content']) for a in ARMS]
        contrast = payloads[2].pop('candidate_distinctions')
        self.assertEqual(contrast, {})
        self.assertEqual(payloads[0], payloads[1]); self.assertEqual(payloads[1], payloads[2])
        self.assertEqual(source, before)

    def test_guidance_is_additive_and_not_in_compact(self):
        self.assertEqual(build_messages(fixture(), ARMS[0])[0]['content'], BASE)
        self.assertIn(GUIDE, build_messages(fixture(), ARMS[1])[0]['content'])
        self.assertIn(GUIDE, build_messages(fixture(), ARMS[2])[0]['content'])

    def test_metadata_cannot_enter_prompt(self):
        value = fixture(); value['gold'] = {'SNOMEDCT_US': '2'}
        with self.assertRaises(AssertionError): build_messages(value, ARMS[0])

    def test_schema_enum_is_candidate_bound(self):
        schema = response_format(fixture())['json_schema']['schema']
        self.assertEqual(schema['properties']['snomed_code']['enum'], ['1', '2'])
        self.assertFalse(schema['additionalProperties'])

    def test_request_is_single_terminal_protocol(self):
        body = request_body(fixture(), ARMS[0])
        self.assertNotIn('tools', body)
        self.assertEqual(body['seed'], 13)
        self.assertFalse(body['reasoning']['enabled'])
        self.assertFalse(body['provider']['allow_fallbacks'])
        self.assertEqual(body['provider']['order'], ['CoreWeave'])

    def test_valid_selection_and_outside_candidate_rejection(self):
        prediction = dict(icd10cm_code='N28.1', snomed_code='1', confidence=.5, evidence='Target says cyst.')
        self.assertIsNone(validate(prediction, fixture()))
        prediction['snomed_code'] = 'unseen'
        self.assertIsNotNone(validate(prediction, fixture()))

    def test_reject_bool_nan_and_extra_fields(self):
        prediction = dict(icd10cm_code='N28.1', snomed_code='1', confidence=.5, evidence='Target says cyst.')
        for confidence in [True, float('nan'), 1.1]:
            self.assertIsNotNone(validate({**prediction, 'confidence': confidence}, fixture()))
        self.assertIsNotNone(validate({**prediction, 'gold': '1'}, fixture()))

    def test_distinctions_have_exact_spans_without_entailment_claim(self):
        source = fixture()['payload']
        result = distinctions(source['candidates']['SNOMEDCT_US'], source['mention'], source['context'])
        simple = next(x for x in result if x['code'] == '2')
        self.assertIn('simple', simple['close_candidate_differences'][0]['words_only_in_this_display'])
        for row in result:
            for span in row['literal_target_occurrences']:
                self.assertEqual(source[span['field']][span['start']:span['end']], span['text'])
        self.assertFalse(any('supported' in k for x in result for k in x))

    def test_predecision_excludes_finish_cogenerated_results(self):
        e = dict(seen={'ICD10CM': ['a'], 'SNOMEDCT_US': ['b']}, payload=dict(training_examples={s: {'examples': []} for s in ['ICD10CM', 'SNOMEDCT_US']}))
        call = dict(name='search_concepts', result=dict(system='ICD10CM', results=[dict(code='unread')]))
        r = dict(trace=[dict(turn=1, tool_calls=[call, dict(name='finish', result={'accepted': True})])], seen_candidates=e['seen'])
        seen, _ = reconstruct(e, r)
        self.assertNotIn('unread', seen['ICD10CM'])

    def test_fallback_excludes_last_unread_results(self):
        e = dict(seen={'ICD10CM': ['a'], 'SNOMEDCT_US': ['b']}, payload=dict(training_examples={s: {'examples': []} for s in ['ICD10CM', 'SNOMEDCT_US']}))
        r = dict(trace=[dict(turn=1, tool_calls=[dict(name='search_concepts', result=dict(system='ICD10CM', results=[dict(code='read')]))]),
                        dict(turn=2, tool_calls=[dict(name='search_concepts', result=dict(system='ICD10CM', results=[dict(code='unread')]))])],
                 seen_candidates={'ICD10CM': ['a', 'read'], 'SNOMEDCT_US': ['b']})
        seen, _ = reconstruct(e, r)
        self.assertEqual(seen['ICD10CM'], {'a', 'read'})


if __name__ == '__main__': unittest.main()
