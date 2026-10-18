import json
import unittest
from unittest.mock import patch

from . import recovery_eval as e
from .common import ARTIFACTS, SYSTEMS, jaccard, norm, read_json, shingles
from .comparisons import ReservedBudget, assert_flat
from .recovery_tools import RecoveryTools


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tools = RecoveryTools(e.EXPERIMENT / "examples.sqlite", "CHF", "Assessment: chronic systolic heart failure (CHF).", True)

    def tearDown(self):
        self.tools.close()

    def recovery_args(self):
        return {"interpretation": "The mention denotes chronic systolic heart failure.",
                "standard_names": {s: "chronic systolic heart failure" for s in SYSTEMS},
                "alternatives": {s: ["congestive heart failure"] for s in SYSTEMS},
                "supported_details": [{"detail": "chronic systolic", "quote": "chronic systolic heart failure"}]}

    def test_recovery_retains_original_and_real_dictionary_provenance(self):
        result = self.tools.recover_queries(**self.recovery_args())
        assert_flat(result)
        self.assertEqual(self.tools.recovery_count, 1)
        for s in SYSTEMS:
            self.assertEqual(result["queries"][s][0], "CHF")
            self.assertLessEqual(len(result["results"][s]["results"]), 12)
            self.assertTrue(all(self.tools.valid(s, r["code"]) for r in result["results"][s]["results"]))
        self.assertIn("I50.22", {r["code"] for r in result["results"]["ICD10CM"]["results"]})

    def test_recovery_rejects_fabricated_quotes_and_oversized_expansion(self):
        args = self.recovery_args()
        args["supported_details"][0]["quote"] = "acute left-sided failure"
        with self.assertRaisesRegex(ValueError, "verbatim"):
            self.tools.recover_queries(**args)
        self.assertEqual(self.tools.recovery_count, 0)
        args = self.recovery_args(); args["alternatives"]["ICD10CM"] *= 3
        with self.assertRaises(ValueError):
            self.tools.recover_queries(**args)

    def test_comparison_is_flat_grounded_and_requires_previously_seen_codes(self):
        with self.assertRaisesRegex(ValueError, "previously retrieved"):
            self.tools.compare_candidates("ICD10CM", ["I50.9", "I50.22"])
        self.tools.inspect_concepts("ICD10CM", ["I50.9", "I50.22"])
        result = self.tools.compare_candidates("ICD10CM", ["I50.9", "I50.22"])
        assert_flat(result)
        self.assertEqual(self.tools.compared["ICD10CM"], {"I50.9", "I50.22"})
        for card in result["candidates"]:
            words = set(norm(card["preferred"] + " " + (card["fsn"] or "")).split())
            self.assertTrue(set(card["distinctive_description_terms"]) <= words)
        with self.assertRaises(ValueError):
            self.tools.compare_candidates("ICD10CM", ["I50.9", "I50.9"])

    def test_tools_and_finish_gates_match_conditions(self):
        names = lambda schemas: {s["function"]["name"] for s in schemas}
        self.assertEqual(names(e.available_schemas("query_recovery")), {"recover_queries"})
        self.assertNotIn("recover_queries", names(e.available_schemas("baseline")))
        self.assertNotIn("compare_candidates", names(e.available_schemas("query_recovery", True)))
        self.assertIn("compare_candidates", names(e.available_schemas("recovery_comparison", True)))
        self.assertEqual(names(e.available_schemas("recovery_comparison", True, True)), {"finish"})
        self.tools.inspect_concepts("ICD10CM", ["I50.9", "I50.22"])
        self.tools.inspect_concepts("SNOMEDCT_US", ["84114007", "42343007"])
        choice = {"icd10cm_code": "I50.22", "snomed_code": "84114007", "confidence": .5, "evidence": "test"}
        self.assertIn("recovery", e.finish_problem(choice, self.tools, "query_recovery"))
        self.tools.recovery_count = 1
        self.assertIsNone(e.finish_problem(choice, self.tools, "query_recovery"))
        self.assertIn("Compare", e.finish_problem(choice, self.tools, "recovery_comparison"))
        self.tools.compare_candidates("ICD10CM", ["I50.9", "I50.22"])
        self.tools.compare_candidates("SNOMEDCT_US", ["84114007", "42343007"])
        self.assertIsNone(e.finish_problem(choice, self.tools, "recovery_comparison"))

    def test_actual_store_disjoint_from_new_and_previous_eval(self):
        rows = read_json(e.EXPERIMENT / "labels.json")
        previous = read_json(ARTIFACTS / "smoke20.json") + read_json(ARTIFACTS / "comparisons_v1" / "labels.json")
        self.assertEqual(len(rows), 80)
        self.assertEqual(len(rows), len({r["group_id"] for r in rows}))
        for field in ("input_id", "group_id"):
            self.assertFalse({r[field] for r in rows} & {r[field] for r in previous})
        mentions = {norm(r["entity"]) for r in rows}
        self.assertFalse(mentions & {norm(r["entity"]) for r in previous})
        ids, groups = {r["input_id"] for r in rows}, {r["group_id"] for r in rows}
        contexts = {}
        for r in self.tools.example_db.execute("SELECT * FROM examples"):
            self.assertNotIn(r["input_id"], ids)
            self.assertNotIn(r["group_id"], groups)
            self.assertNotIn(norm(r["entity"]), mentions)
            contexts[r["group_id"]] = shingles(r["context"])
        targets = [shingles(r["chunk_text"]) for r in rows]
        self.assertLess(max(jaccard(a, b) for a in contexts.values() for b in targets), .85)
        self.assertLess(max(jaccard(a, b) for i, a in enumerate(targets) for b in targets[i+1:]), .85)
        self.assertLess(max(jaccard(a, shingles(r["chunk_text"])) for a in targets for r in previous), .85)
        for item in read_json(e.EXPERIMENT / "inputs.json"):
            self.assertEqual(set(item), {"input_id", "entity", "chunk_text"})

    def test_runner_rejects_labels_and_baseline_matches_previous_policy(self):
        with self.assertRaises(ValueError):
            e.run_one({"entity": "CHF", "chunk_text": "", "input_id": "x", "gold": {}}, "baseline", ReservedBudget(.1))
        item = read_json(e.EXPERIMENT / "inputs.json")[0]
        ev = read_json(e.EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
        choice = {"icd10cm_code": ev["fallback"]["ICD10CM"], "snomed_code": ev["fallback"]["SNOMEDCT_US"],
                  "confidence": .5, "evidence": "Offline test"}
        class FakeClient:
            key = "fake"
            def __init__(self, *args): pass
            def complete(inner, messages, schemas, **kwargs):
                self.assertEqual(messages[0]["content"], e.TASK_PROMPT + "\n" + e.BASE_POLICY)
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
                    {"id": "unit", "type": "function", "function": {"name": "finish", "arguments": json.dumps(choice)}}]}}], "usage": {"cost": 0}}, []
        result = e.run_one(item, "baseline", ReservedBudget(.1), FakeClient)
        self.assertEqual(result["status"], "model")
        self.assertEqual(result["prediction"], choice)
        self.assertEqual(result["initial_evidence_sha256"], ev["sha256"])


if __name__ == "__main__":
    unittest.main()
