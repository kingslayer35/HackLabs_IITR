"""ICD prefix hierarchy and explicitly approximate SNOMED lexical links."""
from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Explicit allow-list from the SNOMED International Editorial Guide, Semantic Tag:
# Jan 2026 Editorial Guide, Semantic Tag (official specifications route):
# https://docs.snomed.org/snomed-ct-specifications/snomed-ct-editorial-guide/readme/authoring/general-naming-conventions/descriptions/fully-specified-name/index
# This is deliberately NOT a list mined from arbitrary trailing parentheses.
SEMANTIC_TAGS = frozenset({
    "body structure", "cell", "cell structure", "morphologic abnormality", "finding", "disorder",
    "environment", "geographic location", "event", "observable entity", "organism", "clinical drug",
    "medicinal product", "medicinal product form", "physical object", "product", "physical force",
    "procedure", "regime/therapy", "qualifier value", "administration method", "basic dose form",
    "calculation", "disposition", "dose form", "intended site", "product name", "release characteristic",
    "role", "state of matter", "transformation", "supplier", "unit of presentation", "record artifact",
    "situation", "attribute", "core metadata concept", "foundation metadata concept", "link assertion",
    "linkage concept", "namespace concept", "owl metadata concept", "social concept", "ethnic group",
    "occupation", "person", "racial group", "religion/philosophy", "navigational concept",
    "specimen", "staging scale", "assessment scale", "tumor staging", "substance", "metadata",
})
TOP_LEVEL_TAGS = {
    "Body structure": "body structure|cell|cell structure|morphologic abnormality",
    "Clinical finding": "finding|disorder",
    "Environment or geographical location": "environment|geographic location",
    "Event": "event", "Observable entity": "observable entity", "Organism": "organism",
    "Pharmaceutical / biologic product": "clinical drug|medicinal product|medicinal product form|product",
    "Physical force": "physical force", "Physical object": "physical object",
    "Procedure": "procedure|regime/therapy",
    "Qualifier value": "qualifier value|administration method|basic dose form|calculation|disposition|dose form|intended site|product name|release characteristic|role|state of matter|transformation|supplier|unit of presentation",
    "Record artifact": "record artifact", "Situation with explicit context": "situation",
    "SNOMED CT Model Component": "metadata|attribute|core metadata concept|foundation metadata concept|link assertion|linkage concept|namespace concept|owl metadata concept",
    "Social context": "social concept|ethnic group|occupation|person|racial group|religion/philosophy",
    "Special concept": "navigational concept", "Specimen": "specimen",
    "Staging and scales": "staging scale|assessment scale|tumor staging", "Substance": "substance",
}
# Resolve duplicate table entries by the domain's own primary tag:
# product -> Pharmaceutical / biologic product; physical object -> Physical object.
TAG_TO_TOP = {tag:domain for domain,tags in TOP_LEVEL_TAGS.items() for tag in tags.split("|")}
assert set(TAG_TO_TOP)==SEMANTIC_TAGS
STOP = frozenset("a an the of in on to for and or with by from at as is are has have been being was were into due involving involving structure finding disorder procedure unspecified nos history family no not without absent negative detected suspected risk left right bilateral acute chronic subacute stage grade type class initial subsequent sequela".split())
QUALIFIERS = re.compile(r"\b(?:stage|grade|type|class)\s+(?:[0-9]+[a-z]?|[ivx]+)\b", re.I)
CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?$")


def semantic_tag(text: str) -> str:
    match = re.search(r"\(([^()]*)\)\s*$", text)
    value = match.group(1).strip().lower() if match else ""
    return value if value in SEMANTIC_TAGS else ""


def strip_tag(text: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", text) if semantic_tag(text) else text


def core_words(text: str) -> tuple[str, ...]:
    text = QUALIFIERS.sub("", strip_tag(text).lower())
    return tuple(sorted(set(re.findall(r"[a-z0-9]+", text)) - STOP))


def qualifier_flags(text: str) -> dict:
    t = text.lower()
    flags = {}
    for name, pattern in {
        "laterality": r"\b(?:bilateral|left|right)\b",
        "acuity": r"\b(?:acute on chronic|subacute|acute|chronic)\b",
        "absent": r"\b(?:no|absent|not detected|negative|without)\b",
        "history": r"\bhistory of\b",
        "family_history": r"\bfamily history\b",
        "suspected_risk": r"\b(?:suspected|at risk of|risk of)\b",
        "stage": r"\bstage\s+([0-9a-z]+)\b",
        "grade": r"\bgrade\s+([0-9a-z]+)\b",
        "type": r"\btype\s+([0-9a-z]+)\b",
    }.items():
        match = re.search(pattern, t)
        flags[name] = (match.group(1) if match.lastindex else match.group(0)) if match else ""
    return flags


def build_icd(raw: pd.DataFrame, tabular: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    groups = defaultdict(list)
    indices = defaultdict(list)
    ranges = set()
    for row in raw.itertuples(index=False):
        code = row.Code.upper()
        if re.fullmatch(r"[A-Z][0-9A-Z]{2}-[A-Z][0-9A-Z]{2}", code):
            ranges.add(code)
            continue
        if not CODE_RE.fullmatch(code):
            raise ValueError("Unrecognized supplied identifier shape; refusing to discard it")
        if row.Description not in groups[code]:
            groups[code].append(row.Description)
        indices[code].append(row.source_row)
    compact = {c.replace(".", ""): c for c in groups}
    parents, children = {}, defaultdict(list)
    for code in groups:
        key = code.replace(".", "")
        parent = next((compact[key[:i]] for i in range(len(key)-1, 2, -1) if key[:i] in compact), "")
        parents[code] = parent
        if parent:
            children[parent].append(code)
    root = ET.parse(tabular).getroot()
    membership, external_aliases = {}, []
    for chapter in root.findall("chapter"):
        chapter_name = chapter.findtext("name", "") + " | " + chapter.findtext("desc", "")
        for section in chapter.findall("section"):
            block = section.attrib.get("id", "")
            for diag in section.findall(".//diag"):
                name = diag.findtext("name", "").upper()
                membership[name[:3]] = (chapter_name, block)
                if name in groups:
                    for note in diag.findall("./inclusionTerm/note"):
                        text = " ".join(note.itertext()).strip()
                        if text:
                            external_aliases.append(dict(system="ICD", code=name, text=text, source="icd_inclusion", origin="CDC FY2026", weight=1.0))
    depths = {}
    rows = []
    for code in sorted(groups, key=lambda c: (len(c), c)):
        parent = parents[code]
        depths[code] = depths[parent]+1 if parent else 0
        chapter, block = membership.get(code[:3], ("", ""))
        descriptions = groups[code]
        text = " ".join(descriptions)
        kids = sorted(children[code])
        rows.append(dict(code=code, preferred=descriptions[0], descriptions=descriptions,
                         source_rows=indices[code], parent=parent, children=kids,
                         depth=depths[code], category=code[:3], chapter=chapter, block=block,
                         is_leaf=not kids, billable=not kids,
                         unspecified=bool(re.search(r"\b(?:unspecified|NOS)\b", text, re.I)),
                         other=bool(re.search(r"\bother\b", text, re.I)),
                         needs_7th=bool(kids) and all(len(c.replace(".", ""))==7 for c in kids),
                         seventh_char=code[-1] if len(code.replace(".", ""))==7 else "",
                         legacy_gold=False))
    first = {c: int(v[0]) for c, v in indices.items()}
    checks = dict(code_count=len(rows), leaves=sum(r["is_leaf"] for r in rows), range_count=len(ranges),
                  parent_edges=sum(bool(x) for x in parents.values()),
                  parent_order_errors=sum(first[p] >= first[c] for c, p in parents.items() if p),
                  missing_chapter=sum(not r["chapter"] for r in rows),
                  missing_block=sum(not r["block"] for r in rows),
                  missing_category=sum(r["category"] not in groups for r in rows))
    return pd.DataFrame(rows).sort_values("code").reset_index(drop=True), external_aliases, checks


def build_snomed(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    groups = defaultdict(list)
    previous = ""
    fragmented = set()
    for row in raw.itertuples(index=False):
        code = row.snomed_code  # never numeric
        if code != previous and code in groups:
            fragmented.add(code)
        groups[code].append((int(row.description_order), row.description, row.source_row))
        previous = code
    rows, multi_tag = [], 0
    trailing = set()
    for code, values in groups.items():
        values.sort(key=lambda x: (x[0], int(x[2])))
        descriptions = list(dict.fromkeys(v[1] for v in values))
        preferred = next((v[1] for v in values if v[0] == 1), values[0][1])
        tagged = [v for v in values if semantic_tag(v[1])]
        fsn = tagged[-1][1] if tagged else ""
        multi_tag += len(set(semantic_tag(v[1]) for v in tagged)) > 1
        for text in descriptions:
            m = re.search(r"\(([^()]*)\)\s*$", text)
            if m:
                trailing.add(m.group(1))
        extension = code[-3:-1] in {"10", "11", "12"}
        tag = semantic_tag(fsn) or "unknown"
        rows.append(dict(code=code, preferred=preferred, synonyms=[s for s in descriptions if s!=preferred],
                         descriptions=descriptions, source_rows=[v[2] for v in values], fsn=fsn,
                         tag=tag, top_level=TAG_TO_TOP.get(tag,"unknown"), is_extension=extension,
                         namespace=code[-10:-3] if extension else "",
                         core_key=" ".join(core_words(preferred)), **qualifier_flags(fsn+" "+preferred)))
    frame = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    return frame, dict(concepts=len(frame), rows=len(raw), fragmented_concepts=len(fragmented),
                       tagged_concepts=int(frame.tag.ne("unknown").sum()),
                       tag_coverage=float(frame.tag.ne("unknown").mean()), multi_official_tag_concepts=multi_tag,
                       product_primary_domain_count=int(frame.tag.eq("product").sum()),
                       trailing_parenthesis_strings=len(trailing), extension_concepts=int(frame.is_extension.sum()))


def build_snomed_links(concepts: pd.DataFrame, target: Path) -> dict:
    """Stream links to disk; the official ontology is never inferred from IDs."""
    by_key = defaultdict(list)
    for r in concepts.itertuples(index=False):
        if r.tag and r.tag!="unknown" and r.core_key:
            by_key[(r.tag, tuple(r.core_key.split()))].append(r.code)
    schema = pa.schema([("src", pa.string()), ("dst", pa.string()), ("kind", pa.string())])
    counts = defaultdict(int)
    batch = []
    with pq.ParquetWriter(target, schema) as writer:
        for (tag, words), codes in sorted(by_key.items()):
            for src in codes:
                for dst in codes:
                    if src != dst:
                        batch.append((src, dst, "qualifier_sibling"))
                        counts["qualifier_sibling"] += 1
                for remove in (1, 2):
                    if len(words) <= remove:
                        continue
                    for removed in itertools.combinations(range(len(words)), remove):
                        subset = tuple(w for i, w in enumerate(words) if i not in removed)
                        for dst in by_key.get((tag, subset), ()):
                            batch.append((src, dst, "lexical_parent"))
                            counts["lexical_parent"] += 1
                if len(batch) >= 100000:
                    writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, x)) for x in batch], schema=schema))
                    batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, x)) for x in batch], schema=schema))
    return dict(counts)


class OfficialSnomedGraph:
    """Optional RF2 plug-in interface, intentionally unavailable in this session."""
    def parents(self, code: str):
        raise NotImplementedError("No licensed RF2 graph is configured")

    def children(self, code: str):
        raise NotImplementedError("No licensed RF2 graph is configured")
