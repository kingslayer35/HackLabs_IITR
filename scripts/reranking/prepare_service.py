"""Prepare local service metadata and tiny DEV audits before allocating a GPU."""
import json

import joblib

from termmap.common.io import load_config, path_for, sha256, write_json


def prepare(cfg):
    root=path_for(cfg,"run_dir");dest=root/"vllm_service";dest.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads((path_for(cfg,"external_dir")/"reranker_snapshot.json").read_text())
    adapter=path_for(cfg,"model_dir")/"reranker_lora/best"
    settings=dict(cfg["vllm"],base=snapshot["path"],adapter=str(adapter),
        adapter_sha256=sha256(adapter/"adapter_model.safetensors"),
        adapter_config_sha256=sha256(adapter/"adapter_config.json"),
        run_dir=str(root),root=str(cfg["root"]))
    write_json(dest/"server.json",settings)
    for source in ["C1","C2_v2"]:
        file=dest/(source+"_audit.joblib")
        if not file.exists():joblib.dump(joblib.load(root/source/"dev_pools.joblib")[:4],file)


if __name__=="__main__":prepare(load_config())
