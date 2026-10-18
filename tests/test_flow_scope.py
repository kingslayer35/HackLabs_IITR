import pytest

from termmap.evaluation.scope import ALL_FLOWS, inference_flows


def test_epoch2_run_excludes_both_zero_shot_controls():
    assert inference_flows({"inference_flows": ["F01", "F11"]}) == ("F01", "F11")
    assert inference_flows({}) == ALL_FLOWS


@pytest.mark.parametrize("flows", [[], ["F01", "F01"], ["F20"]])
def test_invalid_flow_scope_fails_before_scoring(flows):
    with pytest.raises(ValueError):inference_flows({"inference_flows": flows})


def test_epoch2_scorer_never_instantiates_or_scores_zero_shot_controls(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import scripts.reranking.score_hosted as scorer
    import scripts.evaluation.checkpoints as checkpoints

    service=tmp_path/"vllm_service";service.mkdir()
    (service/"SMOKE.json").write_text(json.dumps({"passed":True}))
    cfg={"root":tmp_path,"paths":{"run_dir":str(tmp_path)},"inference_flows":["F01","F11"]}
    calls=[];roles=[]
    def ranker(config,adapter=False,cards=None):
        roles.append(adapter)
        return SimpleNamespace(cards=None,identity="epoch2",close=lambda:None)
    def score(config,ranker,structures,flow,source,dataset):
        calls.append((flow,source,dataset))
        folder=tmp_path/flow;folder.mkdir(exist_ok=True)
        (folder/(dataset+"_raw_metrics.json")).write_text("{}")
        if dataset=="probe":(folder/"probe_raw_summary.json").write_text("{}")
    monkeypatch.setattr(scorer,"load_config",lambda:cfg)
    monkeypatch.setattr(scorer,"metadata",lambda cfg:None)
    monkeypatch.setattr(scorer,"make_ranker",ranker)
    monkeypatch.setattr(scorer,"score_dataset",score)
    monkeypatch.setattr(checkpoints,"save_checkpoint",lambda *args:None)
    monkeypatch.setenv("SLURM_JOB_ID","test-allocation")
    monkeypatch.setattr("sys.argv",["score_vllm_flows"])
    scorer.main()
    assert roles==[True]
    assert calls==[("F01","C1","dev"),("F11","C2_v2","dev"),
                   ("F01","C1","probe"),("F11","C2_v2","probe")]
    assert not (tmp_path/"F00").exists() and not (tmp_path/"F10").exists()
    assert json.loads((tmp_path/"SCORES_READY.json").read_text())["flows"]==["F01","F11"]


def test_smoke_precision_matching_only_casts_loaded_adapter_parameters():
    import torch
    from types import SimpleNamespace
    from scripts.reranking.score_hosted import match_reference_adapter_dtype
    base=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
    adapter=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
    model=SimpleNamespace(named_parameters=lambda:[("base.weight",base),("base.lora_A.default.weight",adapter)])
    assert match_reference_adapter_dtype(SimpleNamespace(model=model),"bfloat16")==1
    assert adapter.dtype==torch.bfloat16
    assert base.dtype==torch.float32


def test_control_test_scorer_uses_only_original_qwen_and_both_frozen_flows(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import scripts.reranking.score_hosted as scorer
    service=tmp_path/"vllm_service";service.mkdir()
    (service/"SMOKE.json").write_text(json.dumps({"passed":True}))
    cfg={"root":tmp_path,"paths":{"run_dir":str(tmp_path)},"inference_flows":["F00","F10"]}
    roles=[];calls=[];frozen=[]
    def ranker(config,adapter=False):
        roles.append(adapter);return SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(scorer,"load_config",lambda:cfg)
    monkeypatch.setattr(scorer,"metadata",lambda cfg:None)
    monkeypatch.setattr(scorer,"verify_frozen",lambda cfg:frozen.append(True))
    monkeypatch.setattr(scorer,"make_ranker",ranker)
    monkeypatch.setattr(scorer,"score_dataset",lambda c,r,s,f,src,ds:calls.append((f,src,ds)))
    monkeypatch.setattr("sys.argv",["score_vllm_flows","--dataset","test"])
    scorer.main()
    assert frozen==[True] and roles==[False]
    assert calls==[("F00","test_base","test"),("F10","test_adapted","test")]
