"""Context-driven query recovery and dictionary-grounded candidate comparison.

These tools use local terminology data only. The controller supplies reformulations;
generated names are search hypotheses, never new ontology facts or code labels.
"""
from __future__ import annotations

from .comparisons import PolicyTools, assert_flat
from .common import SYSTEMS, norm
from .tools import STOPWORDS, SYSTEM_PROPERTY, tool_schema


RECOVERY_SCHEMA = tool_schema(
    "recover_queries",
    "Search both dictionaries using the original mention plus context-derived standard names and alternatives. "
    "Describe only the given mention. Preserve uncertainty and supported qualifiers; never invent acuity, "
    "laterality, complications or severity. Each supported detail requires an exact quote from the input. "
    "Generated names are retrieval hypotheses, not verified facts. Supply both coding systems.",
    {
        "interpretation": {"type": "string", "maxLength": 500},
        "standard_names": {"type": "object", "properties": {s: {"type": "string", "maxLength": 200} for s in SYSTEMS},
                           "required": list(SYSTEMS), "additionalProperties": False},
        "alternatives": {"type": "object", "properties": {s: {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 2} for s in SYSTEMS},
                         "required": list(SYSTEMS), "additionalProperties": False},
        "supported_details": {"type": "array", "maxItems": 6, "items": {"type": "object",
            "properties": {"detail": {"type": "string", "maxLength": 120}, "quote": {"type": "string", "maxLength": 300}},
            "required": ["detail", "quote"], "additionalProperties": False}},
    }, ["interpretation", "standard_names", "alternatives", "supported_details"])

COMPARISON_SCHEMA = tool_schema(
    "compare_candidates",
    "Compare 2 to 6 already retrieved codes from one system side by side: dictionary descriptions, synonyms "
    "and distinguishing description words. Differences are lexical clues, not authoritative clinical rules. "
    "Use input evidence to resolve the distinction; if no candidate fits, search again. Batch both systems.",
    {"system": SYSTEM_PROPERTY, "codes": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6}},
    ["system", "codes"])


class RecoveryTools(PolicyTools):
    def __init__(self, example_path, entity: str, context: str, comparison: bool = False):
        super().__init__(example_path, graph=False)
        self.entity, self.context = entity, context
        self.comparison_enabled = comparison
        self.recovery_count = 0
        self.compared = {s: set() for s in SYSTEMS}

    def recover_queries(self, interpretation, standard_names, alternatives, supported_details):
        if not isinstance(interpretation, str) or not 1 <= len(interpretation.strip()) <= 500:
            raise ValueError("Provide a short interpretation of the given mention")
        if not isinstance(standard_names, dict) or not isinstance(alternatives, dict) or set(standard_names) != set(SYSTEMS) or set(alternatives) != set(SYSTEMS):
            raise ValueError("Provide standard names and alternatives for both systems")
        if not isinstance(supported_details, list) or len(supported_details) > 6:
            raise ValueError("Provide at most six supported details")
        for detail in supported_details:
            if not isinstance(detail, dict) or set(detail) != {"detail", "quote"}:
                raise ValueError("Each supported detail needs detail and quote strings")
            if not all(isinstance(detail[k], str) and detail[k].strip() for k in detail):
                raise ValueError("Detail and supporting quote must be nonempty strings")
            if len(detail["detail"]) > 120 or len(detail["quote"]) > 300:
                raise ValueError("Detail or quote too long")
            if not any(detail["quote"].casefold() in text.casefold() for text in (self.entity, self.context)):
                raise ValueError("Supporting quote must occur verbatim in the supplied input")
        queries = {}
        for system in SYSTEMS:
            name, extra = standard_names[system], alternatives[system]
            if not isinstance(extra, list) or len(extra) > 2:
                raise ValueError("At most two alternative names per system")
            if any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 200 for q in [name, *extra]):
                raise ValueError("Search names must be nonempty strings of at most 200 characters")
            # Preserve literal evidence even when the generated name is wrong.
            queries[system] = list(dict.fromkeys([self.entity, name, *extra]))
        results = {s: self.search_concepts(s, queries[s], limit=12) for s in SYSTEMS}
        self.recovery_count += 1
        result = {"interpretation_hypothesis": interpretation, "supported_details": supported_details,
                  "queries": queries, "results": results,
                  "provenance": "Original mention plus model-generated search hypotheses; results are supplied-dictionary records. Quotes establish text presence, not clinical entailment."}
        assert_flat(result)
        return result

    def compare_candidates(self, system, codes):
        if not self.comparison_enabled:
            raise ValueError("Candidate comparison is disabled in this condition")
        self._system(system)
        if not isinstance(codes, list) or not 2 <= len(codes) <= 6 or any(not isinstance(c, str) for c in codes) or len(set(codes)) != len(codes):
            raise ValueError("Provide 2 to 6 distinct code strings")
        if any(code not in self.seen[system] or not self.valid(system, code) for code in codes):
            raise ValueError("Compare only previously retrieved dictionary-valid codes")
        cards = [self.card(system, code, detailed=True) for code in codes]
        tokens = [set(norm(card["preferred"] + " " + (card["fsn"] or "")).split()) - STOPWORDS for card in cards]
        common = set.intersection(*tokens)
        for card, words in zip(cards, tokens):
            card["distinctive_description_terms"] = sorted(words - common)
        self.compared[system].update(codes)
        result = {"system": system, "candidates": cards, "shared_description_terms": sorted(common),
                  "provenance": "Descriptions and aliases from supplied dictionaries. Distinctive terms are literal word differences; no generated definitions, clinical entailment or ontology edges."}
        assert_flat(result)
        return result

    def dispatch(self, name, arguments):
        if name in {"recover_queries", "compare_candidates"}:
            return getattr(self, name)(**arguments)
        return super().dispatch(name, arguments)
