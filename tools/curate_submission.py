"""Copy a reviewed source tree into component packages, without modifying it.

Usage: python tools/curate_submission.py /path/to/HackLabs
Only explicit source files are copied. Data, credentials, logs, caches, Git
metadata and experiment controllers are never traversed by this curator.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


MODULES = {
    "io": "common.io", "workflow": "common.workflow",
    "normalize": "preprocessing.normalize", "noise": "preprocessing.noise",
    "hierarchy": "terminology.hierarchy", "structure": "terminology.structure",
    "terminology": "terminology.tables", "bridge_index": "terminology.bridge",
    "retrieve": "retrieval.lexical", "dense": "retrieval.dense",
    "cache": "retrieval.cache", "cached_retrieval": "retrieval.candidates",
    "pipeline": "retrieval.pipeline", "online": "retrieval.online",
    "v2_inference": "retrieval.asymmetric", "v2_confirmation": "retrieval.confirmation",
    "adapt_v2": "pretraining.trainer", "rerank_training": "posttraining.trainer",
    "rerank": "reranking.model", "prompts": "reranking.prompts",
    "vllm_rerank": "reranking.vllm", "rank_client": "reranking.client",
    "card_cache": "reranking.cards", "hosted_prediction": "reranking.hosted",
    "serving_evidence": "reranking.evidence", "serving_provenance": "reranking.provenance",
    "decision": "decision.layer", "joint": "decision.coupling",
    "flow_export": "evaluation.export", "evaluate": "evaluation.metrics",
    "splits": "evaluation.splits", "flow_scope": "evaluation.scope",
    "test_authorization": "evaluation.authorization",
}

SCRIPTS = {
    "download_resources": "resources.download",
    "build_terminology": "terminology.build", "upgrade_p0": "terminology.enrich",
    "report_terminology": "terminology.report", "audit_links": "terminology.audit_links",
    "check_p0": "terminology.check",
    "prepare_retrieval": "preprocessing.prepare_queries",
    "audit_preprocessing_rows": "preprocessing.audit_rows",
    "measure_preprocessing": "preprocessing.measure",
    "run_c1": "retrieval.baseline", "encode_updated_generation": "retrieval.encode",
    "encode_pretrain_v2": "retrieval.encode_adapted", "confirm_pretrain_v2": "retrieval.confirm",
    "train_biencoder_v2": "pretraining.train", "eval_pretrain_v2": "pretraining.evaluate",
    "audit_pretrain_v2": "pretraining.audit",
    "train_reranker": "posttraining.train", "audit_reranker_groups": "posttraining.audit",
    "score_flows": "reranking.score", "score_vllm_flows": "reranking.score_hosted",
    "run_vllm_service": "reranking.serve", "prepare_vllm_service": "reranking.prepare_service",
    "audit_vllm_batching": "reranking.audit_batching",
    "fit_flows": "decision.fit", "parallel_flow_decisions": "decision.parallel",
    "report_flow_diagnostics": "evaluation.diagnostics", "evaluate": "evaluation.checkpoints",
    "report_longtail": "evaluation.longtail", "plot_metrics": "evaluation.plot",
    "prepare_joint": "retrieval.prepare_joint", "prepare_drill": "evaluation.prepare_drill",
    "check_determinism": "inference.check_determinism",
    "predict": "inference.predict", "launch_predict": "inference.launch",
}

TESTS = [
    "abbreviation_evidence", "alias_provenance", "card_cache", "decision", "determinism",
    "evaluate", "flow_exports", "hierarchy", "joint", "lexical_cache", "normalize", "prompts",
    "preprocessing_revision", "retrieve", "validator", "adapt_v2", "v2_inference",
    "vllm_rerank", "rank_client", "flow_scope", "test_authorization", "parallel_decisions",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replacement_map():
    result = {}
    for package, entries in [("termmap", MODULES), ("scripts", SCRIPTS)]:
        for old, new in entries.items():
            result[f"{package}.{old}"] = f"{package}.{new}"
            result[f"{package}/{old}.py"] = f"{package}/{new.replace('.', '/')}.py"
    return result


def rewrite(text):
    def from_import(match):
        package, name, alias = match.group(1), match.group(2), match.group(3)
        entries = MODULES if package == "termmap" else SCRIPTS
        if name not in entries:
            return match[0]
        return f"import {package}.{entries[name]} as {alias or name}"
    mapping = replacement_map()
    pattern = "|".join(re.escape(x) for x in sorted(mapping, key=len, reverse=True))
    text = re.sub(r"(?<![\w])(?:" + pattern + r")(?![\w])", lambda m: mapping[m[0]], text)
    text = re.sub(r"from (termmap|scripts) import (\w+)(?: as (\w+))?", from_import, text)
    # Provenance paths sometimes store the file relative to the termmap folder.
    names = {old + ".py": new.replace(".", "/") + ".py" for old, new in MODULES.items()}
    text = re.sub(r"(['\"])([A-Za-z_0-9]+\.py)\1", lambda m: m[1] + names.get(m[2], m[2]) + m[1], text)
    # Sibling files remain siblings inside their component package.
    text = re.sub(r"\.with_name\((['\"])([^'\"]+/)([^/'\"]+\.py)\1\)", r".with_name(\1\3\1)", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    target = Path(__file__).resolve().parents[1]
    if source == target:
        raise ValueError("The submission must be a separate directory")
    records = []
    for package, entries in [("termmap", MODULES), ("scripts", SCRIPTS)]:
        for old, new in entries.items():
            original = source / package / (old + ".py")
            relative = Path(package) / (new.replace(".", "/") + ".py")
            content = original.read_bytes()
            rewritten = rewrite(content.decode()).encode()
            ast.parse(rewritten, filename=str(relative))
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(rewritten)
            records.append(dict(source=str(original.relative_to(source)), destination=str(relative),
                                source_sha256=digest(content), packaged_sha256=digest(rewritten)))
    for folder in [target / "termmap", target / "scripts"]:
        for directory in [folder, *sorted(p for p in folder.iterdir() if p.is_dir())]:
            (directory / "__init__.py").write_text('"""TermMap submission component."""\n')
    (target / "tests").mkdir(exist_ok=True)
    for name in TESTS:
        relative = Path("tests") / ("test_" + name + ".py")
        text = rewrite((source / relative).read_text())
        if name == "v2_inference":
            text = text.replace("path.parent.mkdir(exist_ok=True)", "path.parent.mkdir(parents=True,exist_ok=True)")
        (target / relative).write_text(text)
    for name in ["requirements.txt", "requirements-vllm.lock", "pytest.ini"]:
        (target / name).write_bytes((source / name).read_bytes())
    runtime = target / "slurm/vllm_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "sitecustomize.py").write_bytes((source / "slurm/vllm_runtime/sitecustomize.py").read_bytes())
    (target / "docs").mkdir(exist_ok=True)
    (target / "docs/SOURCE_MANIFEST.json").write_text(json.dumps(dict(
        source_project="HackLabs", source_modified=False,
        transformation="component packages; import and source-provenance paths rewritten",
        omitted=["legacy SapBERT v1", "experiment recovery/continuation controllers", "logs", "score caches", "credentials"],
        files=records), indent=2) + "\n")
    print(f"Curated {len(records)} production files and {len(TESTS)} test modules")


if __name__ == "__main__":
    main()
