"""Checks for data separation, valid-code enforcement and offline fallback."""
import tempfile
import unittest
from pathlib import Path

from .agent import run_agent, validate_finish
from .common import ARTIFACTS, jaccard, norm, read_json, shingles
from .tools import TerminologyTools


class FailingClient:
    key = "test-secret-that-must-not-appear"
    model = "offline-test"
    provider = "none"

    def complete(self, *args, **kwargs):
        raise RuntimeError("simulated service failure " + self.key)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tools = TerminologyTools()

    def tearDown(self):
        self.tools.close()

    def test_finishing_requires_string_ids_and_dictionary_membership(self):
        valid = {"icd10cm_code": "I10", "snomed_code": "59621000", "confidence": .5, "evidence": "test"}
        self.assertIn("retrieved", validate_finish(valid, self.tools))
        self.tools.inspect_concepts("ICD10CM", ["I10"])
        self.tools.inspect_concepts("SNOMEDCT_US", ["59621000"])
        self.assertIsNone(validate_finish(valid, self.tools))
        self.assertIsNotNone(validate_finish({**valid, "snomed_code": 59621000}, self.tools))
        self.assertIsNotNone(validate_finish({**valid, "icd10cm_code": "NO_SUCH_CODE"}, self.tools))
        self.assertIsNotNone(validate_finish({**valid, "icd10cm_code": "A00-B99"}, self.tools))

    def test_live_example_store_has_no_test_overlap(self):
        items = read_json(ARTIFACTS / "smoke20.json")
        ids = {r["input_id"] for r in items}
        mentions = {norm(r["entity"]) for r in items}
        groups = {r["group_id"] for r in items}
        targets = [shingles(r["chunk_text"]) for r in items]
        unique_contexts = set()
        for row in self.tools.db.execute("SELECT input_id,group_id,entity,context FROM examples"):
            self.assertNotIn(row["input_id"], ids)
            self.assertNotIn(row["group_id"], groups)
            self.assertNotIn(norm(row["entity"]), mentions)
            unique_contexts.add(row["context"])
        self.assertTrue(unique_contexts)
        self.assertLess(max(jaccard(shingles(c), t) for c in unique_contexts for t in targets), .85)

    def test_agent_rejects_gold_fields(self):
        with self.assertRaises(AssertionError):
            run_agent({"entity": "hypertension", "chunk_text": "", "input_id": "unit", "gold": {"ICD10CM": "I10"}}, FailingClient())

    def test_api_failure_returns_valid_codes_and_redacts_secret(self):
        result = run_agent({"entity": "hypertension", "chunk_text": "Assessment: hypertension.", "input_id": "unit"}, FailingClient())
        self.assertEqual(result["status"], "fallback_error")
        self.assertNotIn(FailingClient.key, result["error"])
        self.assertTrue(self.tools.valid("ICD10CM", result["prediction"]["icd10cm_code"]))
        self.assertTrue(self.tools.valid("SNOMEDCT_US", result["prediction"]["snomed_code"]))

    def test_empty_initial_search_can_reach_llm_instead_of_aborting(self):
        result = run_agent({"entity": "zzzzunfindabletoken", "chunk_text": "", "input_id": "unit-empty"}, FailingClient())
        self.assertEqual(result["status"], "fallback_error")
        self.assertTrue(all(result["initial_search_empty"].values()))
        self.assertIn("simulated service failure", result["error"])
        self.assertTrue(self.tools.valid("ICD10CM", result["prediction"]["icd10cm_code"]))
        self.assertTrue(self.tools.valid("SNOMEDCT_US", result["prediction"]["snomed_code"]))

    def test_relations_do_not_claim_official_snomed_hierarchy(self):
        relatives = self.tools.get_related_concepts("SNOMEDCT_US", "44054006")
        self.assertIn("NOT official", relatives["provenance"])
        children = self.tools.get_related_concepts("ICD10CM", "I50")["children"]
        self.assertIn("I50.9", {r["code"] for r in children})

    def test_tool_dispatch_does_not_expose_database_or_files(self):
        with self.assertRaises(ValueError):
            self.tools.dispatch("execute_sql", {"query": "SELECT * FROM examples"})


if __name__ == "__main__":
    unittest.main()
