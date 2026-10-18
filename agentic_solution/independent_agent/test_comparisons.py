import json
import unittest
from unittest.mock import patch

from . import comparisons as c
from .common import ARTIFACTS, SYSTEMS, jaccard, norm, read_json, shingles


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tools = c.PolicyTools(c.EXPERIMENT / "examples.sqlite")

    def tearDown(self):
        self.tools.close()

    def test_flat_inspection_hides_relationships_and_unreturned_codes(self):
        result = self.tools.inspect_concepts("ICD10CM", ["I50"])
        c.assert_flat(result)
        self.assertNotIn("I50.9", self.tools.seen["ICD10CM"])
        with self.assertRaises(ValueError):
            self.tools.get_related_concepts("ICD10CM", "I50")

    def test_graph_enabled_only_in_graph_arm(self):
        for mode in c.MODES:
            names = {s["function"]["name"] for s in c.schemas_for(mode)}
            self.assertEqual("get_related_concepts" in names, mode == "graph")
            self.assertEqual({s["function"]["name"] for s in c.schemas_for(mode, True)}, {"finish"})
        graph = c.PolicyTools(c.EXPERIMENT / "examples.sqlite", True)
        relatives = graph.get_related_concepts("ICD10CM", "I50")
        self.assertIn("I50.9", {v["code"] for v in relatives["children"]})
        graph.close()

    def test_shared_initial_evidence_is_flat(self):
        for item in read_json(c.EXPERIMENT / "inputs.json"):
            evidence = read_json(c.EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
            c.assert_flat(evidence["payload"])
            self.assertNotIn("gold", item)

    def test_fresh_set_and_actual_example_store_are_disjoint(self):
        items = read_json(c.EXPERIMENT / "labels.json")
        old = read_json(ARTIFACTS / "smoke20.json")
        self.assertFalse({i["group_id"] for i in items} & {i["group_id"] for i in old})
        self.assertEqual(len(items), len({i["group_id"] for i in items}))
        ids, mentions, groups = ({i["input_id"] for i in items}, {norm(i["entity"]) for i in items}, {i["group_id"] for i in items})
        contexts = {}
        for example in self.tools.example_db.execute("SELECT * FROM examples"):
            self.assertNotIn(example["input_id"], ids)
            self.assertNotIn(norm(example["entity"]), mentions)
            self.assertNotIn(example["group_id"], groups)
            contexts[example["group_id"]] = shingles(example["context"])
        targets = [shingles(i["chunk_text"]) for i in items]
        self.assertLess(max(jaccard(sh, t) for sh in contexts.values() for t in targets), .85)

    def test_concurrent_reservations_obey_cap(self):
        b = c.ReservedBudget(.02)
        b.reserve(.015)
        with self.assertRaises(RuntimeError):
            b.reserve(.01)
        b.add(.001); b.release(.015, 0)
        b.reserve(.015); b.release(.015, .015)
        with self.assertRaises(RuntimeError):
            b.reserve(.01)

    def test_one_shot_performs_only_finish_without_network(self):
        item = next(item for item in read_json(c.EXPERIMENT / "inputs.json")
                    if all(read_json(c.EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))["payload"]["initial_candidates"][s]["results"] for s in SYSTEMS))
        evidence = read_json(c.EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
        chosen = {}
        for system, field in [("ICD10CM", "icd10cm_code"), ("SNOMEDCT_US", "snomed_code")]:
            chosen[field] = next(v["code"] for v in evidence["payload"]["initial_candidates"][system]["results"] if self.tools.valid(system, v["code"]))
        chosen.update(confidence=.5, evidence="offline transport test")
        class FakeClient:
            key = "fake"
            def __init__(self, *args): pass
            def complete(self, messages, schemas, force_finish=False):
                if {s["function"]["name"] for s in schemas} != {"finish"} or not force_finish:
                    raise AssertionError("One-shot exposed extra tools")
                return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
                    {"id": "test", "type": "function", "function": {"name": "finish", "arguments": json.dumps(chosen)}}]}}],
                        "usage": {"cost": 0}}, []
        with patch.object(c, "GuardedClient", FakeClient):
            result = c.run_one(item, "qwen9b", "one_shot", c.ReservedBudget(.01))
        self.assertEqual(result["status"], "model")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["prediction"], chosen)


if __name__ == "__main__":
    unittest.main()
