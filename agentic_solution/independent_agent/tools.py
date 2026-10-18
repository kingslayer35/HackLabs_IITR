from __future__ import annotations

import difflib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .common import ARTIFACTS, SYSTEMS, norm

STOPWORDS = {"the", "of", "and", "a", "an", "in", "for", "with", "to", "at", "by", "on", "patient", "has"}


def query_terms(query: str) -> list[str]:
    return [word for word in dict.fromkeys(norm(query).split()) if word not in STOPWORDS][:14]


def fts_query(terms: list[str], operator: str = "OR") -> str:
    return f" {operator} ".join('"' + word.replace('"', '""') + '"' for word in terms)


class TerminologyTools:
    def __init__(self, path: Path = ARTIFACTS / "terminology.sqlite"):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.seen = {system: set() for system in SYSTEMS}
        self.search_log = []

    def close(self):
        self.db.close()

    def _system(self, system: str):
        if system not in SYSTEMS:
            raise ValueError(f"system must be one of {SYSTEMS}")

    def card(self, system: str, code: str, detailed: bool = False) -> dict | None:
        self._system(system)
        code = str(code)
        row = self.db.execute("SELECT * FROM concepts WHERE system=? AND code=?", (system, code)).fetchone()
        if row is None:
            return None
        result = {k: row[k] for k in ("code", "preferred", "tag", "parent", "is_leaf")}
        if detailed:
            result["fsn"] = row["fsn"]
            result["synonyms"] = [r[0] for r in self.db.execute(
                "SELECT DISTINCT text FROM aliases WHERE system=? AND code=? ORDER BY description_order,text LIMIT 8", (system, code))]
            if system == "ICD10CM":
                result["children"] = [dict(r) for r in self.db.execute(
                    "SELECT code,preferred,is_leaf FROM concepts WHERE system=? AND parent=? ORDER BY code LIMIT 35", (system, code))]
                result["child_count"] = self.db.execute("SELECT count(*) FROM concepts WHERE system=? AND parent=?", (system, code)).fetchone()[0]
                self.seen[system].update(r["code"] for r in result["children"])
        self.seen[system].add(code)
        if result["parent"]:
            self.seen[system].add(result["parent"])
        return result

    def search_concepts(self, system: str, queries: list[str], limit: int = 12) -> dict:
        self._system(system)
        limit = min(max(int(limit), 1), 30)
        if isinstance(queries, str):
            queries = [queries]
        if not queries or len(queries) > 6:
            raise ValueError("Provide 1 to 6 queries")
        # Fuse independent query rankings; names are deduplicated into code IDs.
        merged = defaultdict(lambda: {"score": 0.0, "queries": [], "best_alias": ""})
        for query in queries:
            if not isinstance(query, str) or len(query) > 500:
                raise ValueError("Each query must be a string of at most 500 characters")
            ranked = {}
            direct = self.db.execute("SELECT code,preferred FROM concepts WHERE system=? AND code=?", (system, query.strip())).fetchone()
            if direct:
                ranked[direct["code"]] = (100.0, direct["preferred"])
            for row in self.db.execute("SELECT code,text FROM aliases WHERE system=? AND normalized=? ORDER BY code LIMIT 100", (system, norm(query))):
                ranked[row["code"]] = (90.0, row["text"])
            terms = query_terms(query)
            if terms:
                for operator, bonus in [("AND", 20.0), ("OR", 0.0)]:
                    statement = """SELECT a.code,a.text,bm25(alias_fts) AS relevance
                        FROM alias_fts JOIN aliases a ON a.id=alias_fts.rowid
                        WHERE alias_fts MATCH ? AND a.system=?
                        ORDER BY relevance,a.code,a.id LIMIT 350"""
                    for rank, row in enumerate(self.db.execute(statement, (fts_query(terms, operator), system))):
                        overlap = len(set(terms) & set(norm(row["text"]).split())) / len(set(terms))
                        score = bonus + 10 * overlap + 1 / (rank + 1)
                        if row["code"] not in ranked or score > ranked[row["code"]][0]:
                            ranked[row["code"]] = (score, row["text"])
            ordered = sorted(ranked.items(), key=lambda pair: (-pair[1][0], pair[0]))
            for rank, (code, (score, alias)) in enumerate(ordered[:max(limit * 3, 40)]):
                entry = merged[code]
                entry["score"] += 1 / (20 + rank)
                entry["queries"].append(query)
                if not entry["best_alias"]:
                    entry["best_alias"] = alias
        selected = sorted(merged, key=lambda code: (-merged[code]["score"], code))[:limit]
        results = []
        for code in selected:
            item = self.card(system, code)
            item.update({"matched_alias": merged[code]["best_alias"], "retrieval_score": round(merged[code]["score"], 7)})
            results.append(item)
        self.search_log.append({"system": system, "queries": queries, "codes": selected})
        return {"system": system, "queries": queries, "results": results,
                "note": "Search rank is a lexical signal, not a confidence. Reformulate if meaning is absent."}

    def inspect_concepts(self, system: str, codes: list[str]) -> dict:
        if not isinstance(codes, list) or len(codes) > 12:
            raise ValueError("Provide at most 12 code strings")
        return {"system": system, "concepts": [self.card(system, str(code), detailed=True) or {"code": str(code), "error": "not in supplied dictionary"} for code in codes]}

    def get_related_concepts(self, system: str, code: str, limit: int = 20) -> dict:
        self._system(system)
        limit = min(max(int(limit), 1), 40)
        concept = self.db.execute("SELECT * FROM concepts WHERE system=? AND code=?", (system, str(code))).fetchone()
        if concept is None:
            return {"error": "Code not in supplied dictionary"}
        if system == "ICD10CM":
            parent = self.card(system, concept["parent"]) if concept["parent"] else None
            children = [self.card(system, r[0]) for r in self.db.execute("SELECT code FROM concepts WHERE system=? AND parent=? ORDER BY code LIMIT ?", (system, code, limit)).fetchall()]
            siblings = []
            if concept["parent"]:
                siblings = [self.card(system, r[0]) for r in self.db.execute("SELECT code FROM concepts WHERE system=? AND parent=? AND code!=? ORDER BY code LIMIT ?", (system, concept["parent"], code, limit)).fetchall()]
            return {"system": system, "code": code, "parent": parent, "children": children, "siblings": siblings,
                    "provenance": "Hierarchy derived from supplied ICD prefixes and chapter ranges, not a full official tabular-rule graph."}
        relatives = []
        if concept["family"] and concept["tag"]:
            relatives = [self.card(system, r[0]) for r in self.db.execute(
                "SELECT code FROM concepts WHERE system=? AND family=? AND tag=? AND code!=? ORDER BY code LIMIT ?",
                (system, concept["family"], concept["tag"], code, limit)).fetchall()]
        return {"system": system, "code": code, "qualifier_variants": relatives,
                "provenance": "Derived lexical qualifier variants, NOT official SNOMED IS-A edges. No official relationship file supplied."}

    def find_training_examples(self, query: str, system: str | None = None, limit: int = 4) -> dict:
        if system:
            self._system(system)
        limit = min(max(int(limit), 1), 8)
        terms = query_terms(query)
        if not terms:
            return {"examples": []}
        statement = """SELECT e.*, bm25(example_fts,5.0,1.0) AS relevance
            FROM example_fts JOIN examples e ON e.id=example_fts.rowid
            WHERE example_fts MATCH ?"""
        args = [fts_query(terms)]
        if system:
            statement += " AND e.system=?"
            args.append(system)
        statement += " ORDER BY relevance,e.id LIMIT ?"
        args.append(limit)
        examples = []
        for row in self.db.execute(statement, args).fetchall():
            entity, context = row["entity"], row["context"]
            location = context.casefold().find(entity.casefold())
            location = max(location, 0)
            window = context[max(0, location - 180):location + len(entity) + 220]
            examples.append({"system": row["system"], "mention": entity, "context": window,
                             "code": row["code"], "description": self.card(row["system"], row["code"])["preferred"],
                             "example_id": row["input_id"], "source": "training split only"})
        return {"examples": examples, "note": "Examples teach annotation conventions; similarity does not establish equivalence."}

    def dispatch(self, name: str, arguments: dict):
        if name not in {"search_concepts", "inspect_concepts", "get_related_concepts", "find_training_examples"}:
            raise ValueError("Unknown tool")
        return getattr(self, name)(**arguments)

    def valid(self, system: str, code: str) -> bool:
        if not isinstance(code, str):
            return False
        row = self.db.execute("SELECT 1 FROM concepts WHERE system=? AND code=? AND code NOT LIKE '%-%'", (system, code)).fetchone()
        return row is not None


def tool_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


SYSTEM_PROPERTY = {"type": "string", "enum": list(SYSTEMS)}
TOOL_SCHEMAS = [
    tool_schema("search_concepts", "Search the full supplied dictionary. Expand abbreviations or use standard medical names when initial search misses the meaning. Batch multiple queries.",
                {"system": SYSTEM_PROPERTY, "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, ["system", "queries"]),
    tool_schema("inspect_concepts", "Inspect candidate synonyms, semantic tags and ICD children before making a specificity decision.",
                {"system": SYSTEM_PROPERTY, "codes": {"type": "array", "items": {"type": "string"}, "maxItems": 12}}, ["system", "codes"]),
    tool_schema("get_related_concepts", "Inspect ICD parents/children/siblings, or derived SNOMED qualifier variants. SNOMED variants are not authoritative ontology edges.",
                {"system": SYSTEM_PROPERTY, "code": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 40}}, ["system", "code"]),
    tool_schema("find_training_examples", "Find similar training-only mention/context mappings. The held-out smoke inputs, matching mentions and near-duplicate chunks are excluded from this store.",
                {"query": {"type": "string"}, "system": SYSTEM_PROPERTY, "limit": {"type": "integer", "minimum": 1, "maximum": 8}}, ["query"]),
]
