"""Optional, system-specific recovery with explicit inadequacy reasons."""
from __future__ import annotations

import copy

from .common import SYSTEMS, norm
from .comparisons import PolicyTools, assert_flat
from .recovery_tools import RECOVERY_SCHEMA
from .tools import SYSTEM_PROPERTY, tool_schema


REASONS = ("no_candidates", "meaning_mismatch", "unresolved_abbreviation")
SELECTIVE_SCHEMA = tool_schema(
    "recover_queries",
    "Optional recovery for ONE system whose candidate list is inadequate. State why: no candidates, "
    "candidate meanings do not fit the mention, or an unresolved abbreviation. Do not invoke merely "
    "to choose between plausible codes of different specificity. Original mention results are preserved. "
    "At most one successful recovery per system; standard name and alternatives are hypotheses, "
    "and all qualifier details need exact input quotes.",
    {"system": SYSTEM_PROPERTY,
     "reason": {"type": "string", "enum": list(REASONS)},
     "reason_details": {"type": "string", "minLength": 10, "maxLength": 400},
     "standard_name": {"type": "string", "minLength": 1, "maxLength": 200},
     "alternatives": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 200}, "maxItems": 2},
     "supported_details": copy.deepcopy(RECOVERY_SCHEMA["function"]["parameters"]["properties"]["supported_details"])},
    ["system", "reason", "reason_details", "standard_name", "alternatives", "supported_details"])


class SelectiveTools(PolicyTools):
    def __init__(self, example_path, entity, context):
        super().__init__(example_path, graph=False)
        self.entity, self.context = entity, context
        self.recovery_count = 0
        self.recovered_systems = set()
        self.recovery_events = []

    def recover_queries(self, system, reason, reason_details, standard_name, alternatives, supported_details):
        self._system(system)
        if system in self.recovered_systems:
            raise ValueError("Only one successful recovery per system is permitted")
        if reason not in REASONS or not isinstance(reason_details, str) or not 10 <= len(reason_details.strip()) <= 400:
            raise ValueError("Provide an allowed inadequacy reason and a short explanation")
        searches = [r for r in self.search_log if r["system"] == system]
        if reason == "no_candidates" and (not searches or searches[-1]["codes"]):
            raise ValueError("no_candidates requires an observed empty search for this system; use a meaning-based reason when results exist")
        if not isinstance(alternatives, list) or len(alternatives) > 2:
            raise ValueError("At most two alternatives are permitted")
        if any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 200 for q in [standard_name, *alternatives]):
            raise ValueError("Names must be nonempty strings of at most 200 characters")
        if not isinstance(supported_details, list) or len(supported_details) > 6:
            raise ValueError("Provide at most six supported details")
        for detail in supported_details:
            if not isinstance(detail, dict) or set(detail) != {"detail", "quote"}:
                raise ValueError("Each supported detail requires detail and quote")
            if not all(isinstance(detail[k], str) and detail[k].strip() for k in detail):
                raise ValueError("Detail and quote must be nonempty strings")
            if len(detail["detail"]) > 120 or len(detail["quote"]) > 300:
                raise ValueError("Detail or quote too long")
            if not any(detail["quote"].casefold() in text.casefold() for text in (self.entity, self.context)):
                raise ValueError("Supporting quote must occur verbatim in the supplied input")
        queries, normalized = [], set()
        for query in [self.entity, standard_name, *alternatives]:
            if norm(query) not in normalized:
                queries.append(query); normalized.add(norm(query))
        results = self.search_concepts(system, queries, limit=12)
        self.recovered_systems.add(system)
        self.recovery_count += 1
        self.recovery_events.append({"system": system, "reason": reason, "reason_details": reason_details, "queries": queries})
        result = {"system": system, "reason": reason, "reason_details": reason_details,
                  "supported_details": supported_details, "queries": queries, "results": {system: results},
                  "provenance": "Supplied dictionary records. Names are model-generated retrieval hypotheses; quotes verify text presence, not clinical entailment. Original candidates remain available."}
        assert_flat(result)
        return result

    def dispatch(self, name, arguments):
        if name == "recover_queries":
            return self.recover_queries(**arguments)
        return super().dispatch(name, arguments)
