"""Train-only row coverage and private manual-review cases; CPU, no model work."""
import argparse
import hashlib
import json
import re
from collections import Counter

import joblib
import pandas as pd

from termmap.common.io import load_config, path_for, read_partition, sha256, term_dir, write_json
from termmap.preprocessing.noise import PROTECTED
from termmap.preprocessing.normalize import Normalizer, clean, context_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["before", "after"], default="before")
    args = parser.parse_args()
    cfg = load_config()
    out = term_dir(cfg)
    destination = path_for(cfg, "run_dir") / "preprocessing_row_review" / args.stage
    destination.mkdir(parents=True, exist_ok=True)
    train = read_partition(cfg, "train")
    aliases = pd.read_parquet(out / "aliases.parquet")
    held = set(pd.read_parquet(out / "probe_set.parquet").heldout_norm)
    normalizer = Normalizer(aliases, train, pd.read_parquet(out / "abbreviations.parquet"),
                            pd.read_parquet(out / "abbreviation_alias_support_train.parquet"), held)
    normalizer.tightened = True
    legal = pd.read_parquet(out / "icd_concepts.parquet").code
    records, coverage, grouped = [], [], {}
    for row in train.itertuples():
        context = context_features(row.entity, row.chunk_text, legal, cfg["preprocessing"]["window_chars"])
        views = normalizer.views(row.entity, "" if context["no_context"] else context["window"], row.group_id)
        decisions = normalizer.abbreviation_diagnostics(row.entity, "" if context["no_context"] else context["window"], row.group_id) if hasattr(normalizer,"abbreviation_diagnostics") else []
        canonical = clean(row.entity)
        changed = {view: values for view,values in views.items()
                   if view in {"V2", "V3", "V4"} and values != [canonical]}
        base_tokens = canonical.split()
        preserved = Counter(w for w in base_tokens if w in PROTECTED or any(c.isdigit() for c in w))
        protected_violations = []
        for view, values in changed.items():
            for text in values:
                tokens = Counter(text.split())
                if any(tokens[token] < count for token,count in preserved.items()):
                    protected_violations.append(view)
        flags = dict(changed_views=sorted(changed), no_context=context["no_context"],
                     ambiguous_context=context["ambiguous_context"],
                     protected_violation=sorted(set(protected_violations)))
        flags["unresolved_abbreviation"] = any(d["status"]=="unresolved" for d in decisions)
        needs_review = bool(changed or flags["no_context"] or flags["ambiguous_context"] or protected_violations or flags["unresolved_abbreviation"])
        # Group only identical mention + full extracted window + output views.
        # Shared ICD/SNOMED rows remain represented individually in row coverage.
        key = hashlib.sha256(json.dumps([row.entity,context["window"],views],sort_keys=True).encode()).hexdigest()[:16]
        coverage.append(dict(row_id=row.row_id, case_id=key, system=row.system, **flags,
                             review_status="pending" if needs_review else "automated_checks_only"))
        record=dict(row_id=row.row_id, entity=row.entity, group_id=row.group_id, views=views, context=context)
        records.append(record)
        if needs_review:
            if key not in grouped:
                grouped[key] = dict(case_id=key,mention=row.entity,context=context["window"],views=changed,
                                    flags=flags,abbreviation_decisions=decisions,row_ids=[],gold_descriptions=[],judgment="pending",reason="")
            grouped[key]["row_ids"].append(row.row_id)
            if row.display not in grouped[key]["gold_descriptions"]:
                grouped[key]["gold_descriptions"].append(row.display)
    cases = sorted(grouped.values(),key=lambda x:(x["mention"].lower(),x["case_id"]))
    for index, case in enumerate(cases):case["index"] = index
    write_json(destination / "manual_cases.json",cases)
    pd.DataFrame(coverage).to_parquet(destination / "coverage.parquet",index=False)
    joblib.dump(records,destination / "train_views.joblib")
    summary=dict(n_rows=len(train),n_cases=len(cases),n_rows_requiring_review=sum(x["review_status"]=="pending" for x in coverage),
                 changed_by_view={v:sum(v in r["changed_views"] for r in coverage) for v in ["V2","V3","V4"]},
                 missing_context=sum(r["no_context"] for r in coverage),ambiguous_context=sum(r["ambiguous_context"] for r in coverage),
                 protected_violations=sum(bool(r["protected_violation"]) for r in coverage),
                 unresolved_abbreviations=sum(r["unresolved_abbreviation"] for r in coverage),
                 test_opened=False,dev_opened=False,seed=cfg["seed"],
                 source_sha256=sha256(cfg["root"] / "termmap/preprocessing/normalize.py"))
    write_json(destination / "summary.json",summary)
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
