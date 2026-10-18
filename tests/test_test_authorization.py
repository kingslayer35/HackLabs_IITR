import hashlib
import json

import pytest

from termmap.evaluation.authorization import verify_fixed_test_scorer


def fixed_setup(tmp_path):
    dev=tmp_path/"dev";dev.mkdir()
    policy=tmp_path/"fit-policy.py";policy.write_text("fixed DEV algorithm")
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (dev/"PRE_TEST_POLICY_FREEZE.json").write_text(json.dumps(dict(selected_epoch=2,files={str(policy):digest(policy)})))
    models=tmp_path/"models/reranker_lora";(models/"best").mkdir(parents=True)
    weights=models/"best/adapter_model.safetensors";weights.write_bytes(b"public dummy epoch 2")
    (models/"selection.json").write_text(json.dumps(dict(best_epoch=2,source_adapter_sha256=digest(weights))))
    run=tmp_path/"test";pool=run/"test_adapted";pool.mkdir(parents=True)
    (pool/"pool_provenance.json").write_text(json.dumps(dict(profile="eval_test",datasets=["test"])))
    (pool/"test_pools.joblib").write_bytes(b"prepared inputs; not opened by authorization")
    cfg=dict(root=tmp_path,paths=dict(run_dir=str(run),model_dir=str(tmp_path/"models"),epoch2_dev_run_dir=str(dev)),
        inference_flows=["F11"],fixed_test_scoring_only=True)
    return cfg,policy,weights


def test_fixed_test_scoring_does_not_require_or_fit_a_decision_layer(tmp_path):
    cfg,_,_=fixed_setup(tmp_path)
    verify_fixed_test_scorer(cfg)
    result=json.loads((tmp_path/"test/TEST_SCORER_AUTHORIZED.json").read_text())
    assert result["selected_epoch"]==2 and result["decisions_not_fitted_on_test"]
    assert not (tmp_path/"models/decision").exists()


@pytest.mark.parametrize("changed",["policy","weights","scope"])
def test_test_scoring_rejects_changes_to_frozen_choices(tmp_path,changed):
    cfg,policy,weights=fixed_setup(tmp_path)
    if changed=="policy":policy.write_text("changed after seeing TEST")
    elif changed=="weights":weights.write_bytes(b"different checkpoint")
    else:cfg["inference_flows"]=["F01","F11"]
    with pytest.raises(RuntimeError):verify_fixed_test_scorer(cfg)
