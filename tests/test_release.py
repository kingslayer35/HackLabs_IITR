import json
from pathlib import Path
import pytest
from termmap.common.io import sha256
from termmap.common.release import verify_release


def test_missing_decision_handoff_fails_before_inference(tmp_path):
    cfg={"root":tmp_path,"paths":{"model_dir":"models"}}
    with pytest.raises(RuntimeError,match="handoff is pending"):
        verify_release(cfg)


def test_released_weights_cannot_change(monkeypatch,tmp_path):
    import termmap.common.release as release
    monkeypatch.setattr(release,"dictionary_hash",lambda cfg:"dictionary")
    cfg={"root":tmp_path,"paths":{"model_dir":"models"},
        **{key:{} for key in ["retrieval","preprocessing","reranker","decision","deployment"]}}
    model=tmp_path/"models";model.mkdir();weight=model/"weights.bin";weight.write_bytes(b"fixed")
    manifest=dict(complete=True,dictionary_hash="dictionary",flow="F11",
        settings={key:cfg[key] for key in ["retrieval","preprocessing","reranker","decision","deployment"]},
        files={"models/weights.bin":sha256(weight)})
    (model/"RELEASE.json").write_text(json.dumps(manifest))
    assert verify_release(cfg)["flow"]=="F11"
    weight.write_bytes(b"changed")
    with pytest.raises(RuntimeError,match="artifact changed"):
        verify_release(cfg)


def test_confirmed_encoders_resolve_under_relocated_checkout(monkeypatch,tmp_path):
    from termmap.retrieval.asymmetric import selected_paths
    root=tmp_path/"restored";models=root/"private_assets/models"
    selection={"selection":{}}
    effective={}
    for system in ["ICD","SNOMED"]:
        model=models/"sapbert_v2"/("best_"+system);model.mkdir(parents=True)
        relative=str(model.relative_to(root))
        selection["selection"][system]={"checkpoint":relative}
        effective[system]=relative
    (models/"sapbert_v2/selection.json").write_text(json.dumps(selection))
    run=root/"runtime/runs/C2_v2";run.mkdir(parents=True)
    (run/"confirmation.json").write_text(json.dumps({"effective_models":effective}))
    monkeypatch.chdir(tmp_path)
    cfg={"root":root,"paths":{"model_dir":"private_assets/models","run_dir":"runtime/runs"}}
    selected,_=selected_paths(cfg)
    assert selected=={system:root/path for system,path in effective.items()}
