import json
import unittest

from . import selective_eval as e
from .common import SYSTEMS, read_json
from .comparisons import ReservedBudget, assert_flat
from .selective_tools import SelectiveTools


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        self.tools = SelectiveTools(e.EXPERIMENT / "examples.sqlite", "CHF", "Assessment: chronic systolic heart failure (CHF).")

    def tearDown(self):
        self.tools.close()

    def args(self, system="ICD10CM"):
        return {"system": system, "reason": "unresolved_abbreviation", "reason_details": "CHF needs its clinical expansion for dictionary retrieval.",
                "standard_name": "chronic systolic heart failure", "alternatives": ["CHF", "chf"],
                "supported_details": [{"detail": "chronic systolic", "quote": "chronic systolic heart failure"}]}

    def test_recovery_is_system_specific_and_preserves_original_candidates(self):
        self.tools.inspect_concepts("ICD10CM", ["I50.9"])
        other_seen = self.tools.seen["SNOMEDCT_US"].copy()
        result = self.tools.recover_queries(**self.args())
        assert_flat(result)
        self.assertEqual(set(result["results"]), {"ICD10CM"})
        self.assertEqual(self.tools.seen["SNOMEDCT_US"], other_seen)
        self.assertIn("I50.9", self.tools.seen["ICD10CM"])
        self.assertEqual(result["queries"], ["CHF", "chronic systolic heart failure"])
        self.assertIn("I50.22", {r["code"] for r in result["results"]["ICD10CM"]["results"]})
        self.assertEqual(self.tools.recovered_systems, {"ICD10CM"})
        searches = len(self.tools.search_log)
        with self.assertRaisesRegex(ValueError, "one successful"):
            self.tools.recover_queries(**self.args())
        self.assertEqual(len(self.tools.search_log), searches)

    def test_trigger_quote_and_reason_validation(self):
        self.tools.search_concepts("ICD10CM", ["heart failure"])
        args = self.args(); args["reason"] = "no_candidates"
        with self.assertRaisesRegex(ValueError, "observed empty"):
            self.tools.recover_queries(**args)
        args = self.args(); args["reason"] = "two_plausible_candidates"
        with self.assertRaisesRegex(ValueError, "allowed inadequacy"):
            self.tools.recover_queries(**args)
        args = self.args(); args["supported_details"][0]["quote"] = "acute left-sided"
        with self.assertRaisesRegex(ValueError, "verbatim"):
            self.tools.recover_queries(**args)
        self.assertEqual(self.tools.recovery_count, 0)
        self.tools.search_concepts("ICD10CM", ["zzzzunfindableterm"])
        args = self.args(); args["reason"] = "no_candidates"
        self.assertIn("ICD10CM", self.tools.recover_queries(**args)["results"])

    def test_recovery_deadline_and_per_system_schema_limits(self):
        names = lambda schemas: {s["function"]["name"] for s in schemas}
        for turn in range(3):
            self.assertIn("recover_queries", names(e.available_schemas(turn, set())))
        self.assertNotIn("recover_queries", names(e.available_schemas(3, set())))
        self.assertEqual(names(e.available_schemas(4, set())), {"finish"})
        self.assertNotIn("recover_queries", names(e.available_schemas(1, set(SYSTEMS))))
        only_sn = next(s for s in e.available_schemas(1, {"ICD10CM"}) if s["function"]["name"] == "recover_queries")
        self.assertEqual(only_sn["function"]["parameters"]["properties"]["system"]["enum"], ["SNOMEDCT_US"])
        self.assertFalse({"compare_candidates", "get_related_concepts"} & names(e.available_schemas(0, set())))

    def test_same_inputs_evidence_examples_and_byte_identical_baseline(self):
        hashes = read_json(e.EXPERIMENT / "retained_inputs_and_baseline.json")
        self.assertEqual(len(read_json(e.EXPERIMENT / "inputs.json")), 80)
        for relative, expected in hashes.items():
            self.assertEqual(e.file_hash(e.EXPERIMENT / relative), expected)
            self.assertEqual(e.file_hash(e.SOURCE / relative), expected)

    def test_direct_finish_requires_no_recovery_and_baseline_cannot_rerun(self):
        item = read_json(e.EXPERIMENT / "inputs.json")[0]
        ev = read_json(e.EXPERIMENT / "initial_evidence" / (item["input_id"]+".json"))
        choice = {"icd10cm_code": ev["fallback"]["ICD10CM"], "snomed_code": ev["fallback"]["SNOMEDCT_US"], "confidence": .5, "evidence": "Offline test"}
        class FakeClient:
            key = "test-secret"
            def __init__(self, *args): pass
            def complete(inner, messages, schemas, **kwargs):
                self.assertIsNone(kwargs["force_tool"])
                self.assertFalse(kwargs["force_finish"])
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
                    {"id": "unit", "type": "function", "function": {"name": "finish", "arguments": json.dumps(choice)}}]}}], "usage": {"cost": 0}}, []
        result = e.run_one(item, "selective_recovery", ReservedBudget(.1), FakeClient)
        self.assertEqual(result["status"], "model")
        self.assertEqual(result["recovery_count"], 0)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["initial_evidence_sha256"], ev["sha256"])
        with self.assertRaisesRegex(ValueError, "reuse saved"):
            e.run_one(item, "baseline", ReservedBudget(.1), FakeClient)
        with self.assertRaisesRegex(ValueError, "unlabelled"):
            e.run_one({**item, "gold": {}}, "selective_recovery", ReservedBudget(.1), FakeClient)

    def test_recovery_result_can_be_reviewed_and_finished_within_budget(self):
        item = read_json(e.EXPERIMENT / "inputs.json")[0]
        ev = read_json(e.EXPERIMENT / "initial_evidence" / (item["input_id"]+".json"))
        choice = {"icd10cm_code": ev["fallback"]["ICD10CM"], "snomed_code": ev["fallback"]["SNOMEDCT_US"], "confidence": .5, "evidence": "Offline test"}
        recovery = {"system": "ICD10CM", "reason": "meaning_mismatch", "reason_details": "The initial wording needs standard clinical terminology.",
                    "standard_name": item["entity"], "alternatives": [], "supported_details": []}
        class FakeClient:
            key = "test-secret"
            def __init__(inner, *args): inner.turn = 0
            def complete(inner, messages, schemas, **kwargs):
                inner.turn += 1
                name, args = ("recover_queries", recovery) if inner.turn == 1 else ("finish", choice)
                if inner.turn == 2:
                    returned = json.loads(messages[-1]["content"])
                    self.assertEqual(set(returned["results"]), {"ICD10CM"})
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
                    {"id": f"unit-{inner.turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}], "usage": {"cost": 0}}, []
        result = e.run_one(item, "selective_recovery", ReservedBudget(.1), FakeClient)
        self.assertEqual(result["status"], "model")
        self.assertEqual(result["recovery_count"], 1)
        self.assertEqual(result["recovered_systems"], ["ICD10CM"])
        self.assertEqual(result["model_calls"], 2)


if __name__ == "__main__":
    unittest.main()
