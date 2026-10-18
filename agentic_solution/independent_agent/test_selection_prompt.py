import json
import unittest

from . import selection_prompt_eval as e
from .common import read_json
from .comparisons import ReservedBudget, TASK_PROMPT, schemas_for
from .recovery_eval import BASE_POLICY


class SelectionPromptTests(unittest.TestCase):
    def test_tool_schemas_exactly_match_baseline_on_every_turn(self):
        for turn in range(5):
            self.assertEqual(e.available_schemas(turn), schemas_for("search_inspect", final_turn=(turn == 4)))
        self.assertEqual(e.MAX_TURNS, 5)

    def test_original_inputs_example_store_and_baseline_retained_byte_for_byte(self):
        retained = read_json(e.EXPERIMENT / "retained_inputs_and_baseline.json")
        self.assertEqual(len(read_json(e.EXPERIMENT / "inputs.json")), 80)
        for relative, expected in retained.items():
            self.assertEqual(e.file_hash(e.EXPERIMENT / relative), expected)
            self.assertEqual(e.file_hash(e.SOURCE / relative), expected)

    def test_only_appended_prompt_changes_and_immediate_finish_remains_valid(self):
        item = read_json(e.EXPERIMENT / "inputs.json")[0]
        evidence = read_json(e.EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
        final = {"icd10cm_code": evidence["fallback"]["ICD10CM"], "snomed_code": evidence["fallback"]["SNOMEDCT_US"], "confidence": .5, "evidence": "Offline test"}
        baseline = read_json(e.SOURCE / "runs/baseline" / (item["input_id"] + ".json"))
        class FakeClient:
            key = "fake-secret"
            def __init__(inner, model, provider, budget):
                self.assertEqual(model, baseline["model"])
                self.assertEqual(provider, baseline["provider_requested"])
            def complete(inner, messages, schemas, **kwargs):
                self.assertEqual(messages[0]["content"], baseline["messages"][0]["content"] + "\n" + e.SELECTION_CHECK)
                self.assertEqual(messages[0]["content"], TASK_PROMPT + "\n" + BASE_POLICY + "\n" + e.SELECTION_CHECK)
                self.assertEqual(messages[1], baseline["messages"][1])
                self.assertEqual(schemas, schemas_for("search_inspect"))
                self.assertFalse(kwargs["force_finish"])
                self.assertIsNone(kwargs["force_tool"])
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
                    {"id": "test", "type": "function", "function": {"name": "finish", "arguments": json.dumps(final)}}]}}], "usage": {"cost": 0}}, []
        result = e.run_one(item, "example_equivalence", ReservedBudget(.1), FakeClient)
        self.assertEqual(result["status"], "model")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["prediction"], final)

    def test_labels_and_baseline_reruns_are_rejected_and_check_has_no_case_mappings(self):
        item = read_json(e.EXPERIMENT / "inputs.json")[0]
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            e.run_one({**item, "gold": {}}, "example_equivalence", ReservedBudget(.1))
        with self.assertRaisesRegex(ValueError, "reuse saved"):
            e.run_one(item, "baseline", ReservedBudget(.1))
        for case in read_json(e.ARTIFACTS / "selection_audit_v1/cases.json"):
            self.assertNotIn(case["input_id"], e.SELECTION_CHECK)
            self.assertNotIn(case["gold_code"], e.SELECTION_CHECK)
            self.assertNotIn(case["predicted_code"], e.SELECTION_CHECK)


if __name__ == "__main__":
    unittest.main()
