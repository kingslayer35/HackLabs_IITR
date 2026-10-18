"""Authorize scoring fixed TEST inputs before the DEV decision fit completes."""
import json
from pathlib import Path

from termmap.common.io import path_for,sha256,write_json


def verify_fixed_test_scorer(cfg):
    if cfg.get("inference_flows")!=["F11"] or not cfg.get("fixed_test_scoring_only"):
        raise RuntimeError("Fixed-weight TEST scoring is restricted to requested F11")
    epoch2=path_for(cfg,"epoch2_dev_run_dir")
    policy=json.loads((epoch2/"PRE_TEST_POLICY_FREEZE.json").read_text())
    if policy["selected_epoch"]!=2 or not all(sha256(Path(p))==h for p,h in policy["files"].items()):
        raise RuntimeError("The epoch-2 choice or fixed DEV fitting policy changed")
    selected=path_for(cfg,"model_dir")/"reranker_lora/selection.json"
    manifest=json.loads(selected.read_text())
    if manifest["best_epoch"]!=2 or sha256(path_for(cfg,"model_dir")/"reranker_lora/best/adapter_model.safetensors")!=manifest["source_adapter_sha256"]:
        raise RuntimeError("The TEST scorer is not the user's selected epoch 2")
    prepared=path_for(cfg,"run_dir")/"test_adapted"
    info=json.loads((prepared/"pool_provenance.json").read_text())
    if info["profile"]!="eval_test" or info["datasets"]!=["test"]:
        raise RuntimeError("Use the already prepared shared TEST pool")
    write_json(path_for(cfg,"run_dir")/"TEST_SCORER_AUTHORIZED.json",dict(
        selected_epoch=2,model_fixed_before_test=True,dev_fitting_policy_fixed=True,
        decisions_not_fitted_on_test=True,prepared_pool_sha256=sha256(prepared/"test_pools.joblib")))
