"""Freeze the numerical inference contract along with fitted decisions."""
import json
from pathlib import Path

from termmap.common.io import path_for, sha256


def serving_metadata(cfg):
    if cfg.get("reranker",{}).get("backend")!="vllm":return {}
    service=path_for(cfg,"run_dir")/"vllm_service"
    pointer=service/"DEV_EVIDENCE.json"
    evidence=json.loads(pointer.read_text())
    hashes={}
    for name in ["runtime.json","server.json","SMOKE.json","BATCH_AUDIT.json"]:
        entry=evidence["files"][name];file=Path(entry["path"])
        if file.name!=name or not file.resolve().is_relative_to((service/"evidence/dev").resolve()):
            raise RuntimeError("DEV serving evidence must stay in its immutable archive")
        actual=sha256(file)
        if actual!=entry["sha256"]:raise RuntimeError("DEV serving evidence changed: "+name)
        hashes[name]=actual
    implementation=["termmap/reranking/vllm.py","scripts/reranking/serve.py","slurm/vllm_runtime/sitecustomize.py"]
    if cfg["vllm"].get("input_cache_enabled",False):
        implementation += ["termmap/reranking/client.py","termmap/reranking/cards.py"]
    return dict(serving_settings=cfg["vllm"],
        serving_implementation={name:sha256(cfg["root"]/name) for name in implementation},
        serving_environment_sha256=sha256(cfg["root"]/"requirements-vllm.lock"),
        dev_serving_evidence_sha256=sha256(pointer),dev_serving_audit_sha256=hashes)
