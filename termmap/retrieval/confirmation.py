"""The six fixed encoder checks and independent per-system fallback."""
import json
import numpy as np
import pandas as pd
import joblib

from termmap.common.io import path_for, term_dir, write_json, sha256
from termmap.retrieval.cache import directory
from termmap.evaluation.metrics import group_ci
from termmap.retrieval.asymmetric import SYSTEMS, base_model, selected_paths

RULES = [("dense_R@64", 1., "dense R@64 ≥ base + 1.0"),
         ("dense_R@1", 0., "dense R@1 not lower"),
         ("dense_unseen_R@64", -.5, "unseen dense R@64 ≥ base − 0.5"),
         ("probe_dense_R@64", -1., "probe dense R@64 ≥ base − 1.0"),
         ("pipeline_R@64", 0., "full-pipeline R@64 not lower"),
         ("memory_R@10", 0., "memory R@10 not lower")]


def rule_checks(adapted, base):
    return {s: {key: dict(base=float(base[s][key]), adapted=float(adapted[s][key]),
                          delta=float(adapted[s][key] - base[s][key]),
                          passed=bool(adapted[s][key] >= base[s][key] + minimum))
                for key, minimum, _ in RULES} for s in SYSTEMS}


def choose_models(checks, candidates, original):
    return {s: str(candidates[s]) if all(row["passed"] for row in checks[s].values()) else str(original)
            for s in SYSTEMS}


def summary(cfg, run):
    folder = path_for(cfg, "run_dir") / run
    frequency = {(r.system, r.code): r.frequency for r in pd.read_parquet(term_dir(cfg) / "priors_train.parquet").itertuples()}
    dev = joblib.load(folder / "dev_pools.joblib")
    probe = joblib.load(folder / "probe_pools.joblib")
    rows = []
    for dataset, items in [("dev", dev), ("probe", probe)]:
        for item in items:
            row = item["row"]
            value = dict(dataset=dataset, row_id=str(row["row_id"]), system=row["system"],
                         code=row["code"], group=str(row["group_id"]),
                         inferred_type=row.get("inferred_subconcept",row.get("subconcept","Unknown")),
                         unseen=frequency.get((row["system"], row["code"]), 0) == 0,
                         pipeline_R64=row["code"] in dict(item["pool"]))
            for channel in ["dense", "memory"]:
                codes = [code for code, _ in item["channels"][channel]]
                value[channel + "_rank"] = codes.index(row["code"]) + 1 if row["code"] in codes else 10**9
            rows.append(value)
    frame = pd.DataFrame(rows)
    result = {}
    for system in SYSTEMS:
        d = frame.loc[frame.system.eq(system) & frame.dataset.eq("dev")]
        p = frame.loc[frame.system.eq(system) & frame.dataset.eq("probe")]
        r = {f"dense_R@{k}": float((d.dense_rank <= k).mean() * 100) for k in [1, 10, 32, 64]}
        r.update(dense_unseen_R64_ci=group_ci(d.loc[d.unseen, "dense_rank"] <= 64, d.loc[d.unseen, "group"], cfg["split"]["bootstrap_samples"], 13))
        r.update({"dense_unseen_R@64": float((d.loc[d.unseen].dense_rank <= 64).mean() * 100),
                  "probe_dense_R@64": float((p.dense_rank <= 64).mean() * 100),
                  "pipeline_R@64": float(d.pipeline_R64.mean() * 100),
                  "memory_R@10": float((d.memory_rank <= 10).mean() * 100)})
        result[system] = r
    return result, frame


def confirm(cfg):
    target = path_for(cfg, "run_dir") / "C2_v2"
    path = target / "confirmation.json"
    baseline, left = summary(cfg, "C1")
    adapted, right = summary(cfg, "C2_v2")
    if not left[["dataset", "row_id", "system", "code", "group","inferred_type"]].equals(right[["dataset", "row_id", "system", "code", "group","inferred_type"]]):
        raise RuntimeError("The two encoder arms use different evaluation rows or gold labels")
    cache = directory(cfg)
    original = base_model(cfg)
    candidates, manifest = selected_paths(cfg, candidate=True)
    checks = rule_checks(adapted, baseline)
    previous = json.loads(path.read_text()) if path.exists() else None
    candidate_evidence = previous["candidate_checks"] if previous else checks
    effective = choose_models(candidate_evidence, candidates, original)
    encoding = json.loads((cache / "C2_v2/encoding_metadata.json").read_text())
    baseline_encoding = json.loads((cache / "C1/encoding_metadata.json").read_text())
    if previous and previous["query_generation_sha256"] != encoding["queries_sha256"]:
        raise RuntimeError("Confirmation evidence belongs to a different query generation")
    if encoding["queries_sha256"] != baseline_encoding["queries_sha256"]:
        raise RuntimeError("The encoder comparison mixed query generations")
    if encoding["gpu_architecture"] != baseline_encoding["gpu_architecture"]:
        raise RuntimeError("The encoder comparison mixed GPU architectures")
    if sha256(cache / "C1/query_embeddings.npy") != sha256(cache / "C2_v2/query_embeddings.npy"):
        raise RuntimeError("Type-classifier query vectors differ between encoder arms")
    ready = encoding["models"] == effective
    used = {s: effective[s] != str(original) for s in SYSTEMS}
    if ready:
        for s in SYSTEMS:
            if used[s] and not all(row["passed"] for row in checks[s].values()):
                raise RuntimeError("A retained encoder no longer passes its fixed confirmation rules")
            if not used[s]:
                cols = ["dense_rank", "memory_rank", "pipeline_R64"]
                if not left.loc[left.system.eq(s), cols].reset_index(drop=True).equals(right.loc[right.system.eq(s), cols].reset_index(drop=True)):
                    raise RuntimeError("Off-the-shelf system fallback does not reproduce the base arm")
    paired = {}
    for s in SYSTEMS:
        ids = left.system.eq(s) & left.dataset.eq("dev")
        diff = (right.loc[ids, "dense_rank"] <= 64).astype(float) - (left.loc[ids, "dense_rank"] <= 64).astype(float)
        paired[s] = dict(delta=float(diff.mean() * 100), ci=group_ci(diff, left.loc[ids, "group"], cfg["split"]["bootstrap_samples"], 13))
    result = dict(ready=ready, adapted=used, effective_models=effective, candidate_checks=candidate_evidence,
                  effective_checks=checks, baseline=baseline, effective=adapted, paired_dense_R64=paired,
                  query_generation_sha256=encoding["queries_sha256"],
                  selection_sha256=sha256(path_for(cfg, "model_dir") / "sapbert_v2/selection.json"),
                  seed=13, test_opened=False, retrained=False)
    write_json(path, result)
    text = "# Updated-generation SapBERT confirmation\n\nDEV chart mentions and held-out-alias probe; no training and no sealed-test access. Both arms use the same revised queries, original alias vectors and off-the-shelf type-classifier queries. Each system uses its selected encoder for both query and memory vectors.\n\n"
    text += "| Metric | System | New C1 | C2_v2 | Delta |\n|---|---|---:|---:|---:|\n"
    for key in ["dense_R@1", "dense_R@10", "dense_R@32", "dense_R@64", "dense_unseen_R@64", "probe_dense_R@64", "pipeline_R@64", "memory_R@10"]:
        for s in SYSTEMS:
            text += f"| {key} | {s} | {baseline[s][key]:.4f} | {adapted[s][key]:.4f} | {adapted[s][key]-baseline[s][key]:+.4f} |\n"
    text += "\n| Candidate pass rule | ICD | SNOMED |\n|---|---|---|\n"
    for key, _, label in RULES:
        text += f"| {label} | " + " | ".join("pass" if candidate_evidence[s][key]["passed"] else "FAIL → original encoder" for s in SYSTEMS) + " |\n"
    text += f"\nAdapted encoders retained: {used}. Confirmation ready for downstream fitting: {ready}. Paired grouped-bootstrap dense R@64 changes: {paired}. Retrieval recall is not final prediction accuracy.\n"
    (path_for(cfg, "report_dir") / "v2_confirmation.md").write_text(text)
    print("Updated v2 confirmation:", json.dumps(dict(ready=ready, adapted=used, paired_dense_R64=paired)), flush=True)
    return result
