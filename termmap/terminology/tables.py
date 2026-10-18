"""Terminology tables, training-only priors and public alias provenance."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

from termmap.preprocessing.normalize import clean


def index_aliases(index: Path, valid: set[str]) -> list[dict]:
    root = ET.parse(index).getroot()
    rows = []
    def visit(node, trail):
        title = node.find("title")
        label = " ".join(title.itertext()).strip() if title is not None else ""
        path = trail + ([label] if label else [])
        for code_node in node.findall("code"):
            code = (code_node.text or "").strip().upper()
            if code in valid and path:
                rows.append(dict(system="ICD", code=code, text=", ".join(path), source="icd_index", origin="CDC FY2026", weight=1.0))
        for child in node:
            if child.tag in {"term", "mainTerm"}:
                visit(child, path)
    for letter in root.findall("letter"):
        for node in letter.findall("mainTerm"):
            visit(node, [])
    return rows


def aliases_for(icd: pd.DataFrame, snomed: pd.DataFrame, external: list[dict]) -> pd.DataFrame:
    rows = list(external)
    for system, frame in [("ICD", icd), ("SNOMED", snomed)]:
        for r in frame.itertuples(index=False):
            for i, text in enumerate(r.descriptions):
                source = "dict_pref" if text==r.preferred else ("icd_desc" if system=="ICD" else "dict_syn")
                if system=="SNOMED" and text==r.fsn:
                    from termmap.terminology.hierarchy import strip_tag
                    text, source = strip_tag(text), "dict_fsn"
                rows.append(dict(system=system, code=r.code, text=text, source=source, origin="supplied", weight=1.0))
    aliases = pd.DataFrame(rows)
    aliases["norm_text"] = aliases.text.map(clean)
    return aliases.loc[aliases.norm_text.ne("")].drop_duplicates(["system", "code", "norm_text", "source"]).reset_index(drop=True)


def clean_training(raw: pd.DataFrame, icd: pd.DataFrame, snomed: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().rename(columns={"source_row": "row_id"})
    frame["system"] = frame.coding_system.map({"ICD10CM": "ICD", "SNOMEDCT_US": "SNOMED"})
    if frame.system.isna().any():
        raise ValueError("Unexpected coding system in labelled input")
    mask = frame.system.eq("ICD")
    frame.loc[mask, "code"] = frame.loc[mask, "code"].str.upper()
    legal = {"ICD": set(icd.code), "SNOMED": set(snomed.code)}
    descriptions = {"ICD": {r.code: {clean(s) for s in r.descriptions} for r in icd.itertuples()},
                    "SNOMED": {r.code: {clean(s) for s in r.descriptions} for r in snomed.itertuples()}}
    frame["valid_gold"] = [r.code in legal[r.system] for r in frame.itertuples()]
    frame["dirty_label"] = frame.system.eq("ICD") & frame.code.eq("L6.0")
    frame["display_matches"] = [clean(r.display) in descriptions[r.system].get(r.code, set()) for r in frame.itertuples()]
    return frame


def abbreviation_table(aliases, external_csv: Path, train=None) -> pd.DataFrame:
    entries = Counter()
    for text in aliases.text:
        match = re.match(r"^([A-Z][A-Z0-9/.-]{1,11})\s+-\s+(.+)$", text)
        if match:
            entries[(match.group(1).lower(), clean(match.group(2)), "dictionary")] += 1
    meta = pd.read_csv(external_csv, sep="|", dtype=str, keep_default_na=False)
    # Inventory v1 has `abbreviation`, `long_form` (validate, do not guess silently).
    lower = {c.lower():c for c in meta}
    a = next((lower[k] for k in ["abbreviation", "short_form", "sf"] if k in lower), None)
    e = next((lower[k] for k in ["long_form", "expansion", "lf", "sense"] if k in lower), None)
    if not a or not e:
        raise ValueError(f"Unsupported public inventory schema: {list(meta.columns)}")
    for abbr, expansion in zip(meta[a], meta[e]):
        abbr, expansion = clean(abbr), clean(expansion)
        if abbr and expansion and len(abbr)<=12:
            entries[(abbr, expansion, "meta_inventory")] += 1
    if train is not None:
        for r in train.loc[train.valid_gold].itertuples():
            words = re.findall(r"[a-z]+", r.display.lower())
            for token in re.findall(r"\b[a-z]{2,8}\b", r.entity.lower()):
                if len(words)>=2 and "".join(w[0] for w in words)==token:
                    entries[(token, " ".join(words), "train_initials")] += 1
    totals, shares = Counter(), Counter()
    for (a,e,s), n in entries.items():
        totals[a] += n; shares[(a,e)] += n
    significant = Counter(a for (a,e),n in shares.items() if n/totals[a]>=0.1)
    ambiguous = {a for a,n in significant.items() if n>1}
    return pd.DataFrame([dict(abbr=a, expansion=e, prior_count=n, source=s, ambiguous=a in ambiguous)
                         for (a,e,s),n in sorted(entries.items())])


def abbreviation_alias_support(aliases, abbreviations, train):
    """Track groups behind nonpublic alias assignments for leave-group-out search."""
    base=aliases.loc[aliases.source.ne("abbr_variant")]
    expansions=defaultdict(set)
    for row in base[["system","code","norm_text"]].drop_duplicates().itertuples(index=False):
        expansions[row.norm_text].add((row.system,row.code))
    public={(r.abbr,r.expansion) for r in abbreviations.itertuples() if r.source!="train_initials"}
    groups=defaultdict(set)
    for row in train.loc[train.valid_gold].itertuples():
        words=re.findall(r"[a-z]+",row.display.lower())
        for token in re.findall(r"\b[a-z]{2,8}\b",row.entity.lower()):
            if len(words)>=2 and "".join(w[0] for w in words)==token:
                groups[(token," ".join(words))].add(row.group_id)
    evidence=defaultdict(set);permanent=set()
    for row in abbreviations.loc[~abbreviations.ambiguous].itertuples():
        for system,code in expansions.get(row.expansion,()):
            key=(system,code,row.abbr)
            if (row.abbr,row.expansion) in public:permanent.add(key)
            evidence[key].update(groups.get((row.abbr,row.expansion),()))
    permanent.update((r.system,r.code,r.norm_text) for r in base.itertuples())
    rows=[dict(system=s,code=c,norm_text=n,support_groups=sorted(g))
          for (s,c,n),g in sorted(evidence.items()) if g and (s,c,n) not in permanent]
    return pd.DataFrame(rows,columns=["system","code","norm_text","support_groups"])


def make_priors(train, snomed, icd, aliases):
    valid = train.loc[train.valid_gold]
    frequency = valid.groupby(["system","code"]).size().rename("frequency").reset_index()
    tags = snomed.set_index("code").tag.to_dict()
    tagged = valid.loc[valid.system.eq("SNOMED"), ["subconcept","code","group_id"]].copy()
    tagged["tag"] = tagged.code.map(tags).fillna("")
    tag_priors = tagged.groupby(["subconcept","tag"]).size().rename("count").reset_index()
    tag_priors["probability"] = tag_priors["count"] / tag_priors.groupby("subconcept")["count"].transform("sum")
    tagged["top_level"] = tagged.code.map(snomed.set_index("code").top_level.to_dict()).fillna("unknown")
    top_priors = tagged.groupby(["subconcept","top_level"]).size().rename("count").reset_index()
    top_priors["probability"] = top_priors["count"] / top_priors.groupby("subconcept")["count"].transform("sum")
    paired = valid.loc[valid.system.eq("ICD")].merge(valid.loc[valid.system.eq("SNOMED")],
                on=["entity","chunk_text","group_id"], suffixes=("_icd","_snomed"))
    pairs = paired[["code_icd","code_snomed","group_id"]].rename(columns={"code_icd":"icd","code_snomed":"snomed"})
    counts = pairs.groupby(["snomed","icd"]).size().rename("count").reset_index()
    counts["total"] = counts.groupby("snomed")["count"].transform("sum")
    counts["share"] = counts["count"]/counts.total
    bridge = counts.loc[counts.total.ge(2) & counts.share.ge(.6)].copy()
    memory = valid[["row_id","system","code","entity","group_id"]].rename(columns={"entity":"text"})
    memory["norm_text"] = memory.text.map(clean)
    seen = memory.sort_values("row_id").groupby(["system","code"]).head(3)
    leaves = set(icd.loc[icd.is_leaf, "code"])
    legacy = sorted(set(valid.loc[valid.system.eq("ICD"), "code"]) - leaves)
    return dict(priors_train=frequency, tag_priors_train=tag_priors, top_level_priors_train=top_priors, cooccurrence_train=counts,
                pair_evidence_train=pairs, bridge_train=bridge, train_memory=memory, seen_as_train=seen), legacy
