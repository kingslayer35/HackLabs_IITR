"""Train-only deployment must not fall through to all-label tables/TEST."""
import json
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd


def test_eval_deployment_uses_train_tables_and_keeps_probe_exclusions(monkeypatch,tmp_path):
    import termmap.retrieval.online as online
    calls=[]
    data=pd.DataFrame([dict(code="A",text="public description",norm_text="public description")])
    def cached(self,cfg,run,profile):
        calls.append((run,profile));self.cfg=cfg;self.dense_dir=tmp_path
        self.qvectors=np.zeros((1,2));self.system_qvectors={s:np.zeros((1,2)) for s in ["ICD","SNOMED"]}
        self.char={};self.dense={};self.word={};self.memory_scores={};self.bridge_vectors=None;self.bridge_scores={}
        self.alias_rows={s:data for s in ["ICD","SNOMED"]};self.embeddings={s:np.zeros((1,2)) for s in ["ICD","SNOMED"]}
        self.aliases=data
    monkeypatch.setattr(online.CachedRetriever,"__init__",cached)
    monkeypatch.setattr(online,"base_model",lambda cfg:tmp_path/"original")
    monkeypatch.setattr(online,"selected_paths",lambda cfg:({s:tmp_path/"adapted" for s in ["ICD","SNOMED"]},{}))
    monkeypatch.setattr(online,"Encoder",lambda *args:SimpleNamespace())
    monkeypatch.setattr(online,"DenseIndex",lambda *args:None)
    monkeypatch.setattr(online,"term_dir",lambda cfg:tmp_path)
    monkeypatch.setattr(online,"directory",lambda *args:tmp_path)
    monkeypatch.setattr(online,"all_rows",lambda cfg:(_ for _ in ()).throw(AssertionError("all-label path used")))
    monkeypatch.setattr(online.torch.cuda,"get_device_capability",lambda:(8,6))
    (tmp_path/"encoding_metadata.json").write_text(json.dumps(dict(gpu_architecture="sm86")))
    for system in ["ICD","SNOMED"]:np.save(tmp_path/(system+"_memory.npy"),np.zeros((1,2)))
    pd.DataFrame(dict(row_id=["train-only"])).to_parquet(tmp_path/"train.parquet")
    pd.DataFrame(dict(heldout_norm=["held alias"])).to_parquet(tmp_path/"probe_set.parquet")
    pd.DataFrame(dict(abbr=["abbr"])).to_parquet(tmp_path/"abbreviations.parquet")
    def normalizer(aliases,training,abbreviations,support,**kwargs):
        assert training.row_id.tolist()==["train-only"]
        assert kwargs["excluded_names"]=={"held alias"}
        return SimpleNamespace()
    monkeypatch.setattr(online,"Normalizer",normalizer)
    classifier=tmp_path/"classifier.joblib";joblib.dump({},classifier)
    monkeypatch.setattr(online,"type_cache_path",lambda *args,**kwargs:classifier)
    online.OnlineRetriever(dict(deployment=dict(profile="eval"),training=dict(max_length=40),
        retrieval=dict(encode_batch=8,dense_chunk=4)),"F11")
    assert calls==[("C2_v2","eval")]
