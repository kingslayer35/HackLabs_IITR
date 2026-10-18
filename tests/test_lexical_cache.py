"""An interrupted score writer must not masquerade as a complete channel."""
import json
import numpy as np
import pandas as pd
import pytest
import termmap.retrieval.cache as cache


def test_incomplete_lexical_mmap_is_rebuilt(monkeypatch,tmp_path):
    query_dir=tmp_path/"eval";query_dir.mkdir()
    (query_dir/"queries.json").write_text(json.dumps(["public query"]))
    aliases=pd.DataFrame([dict(system=s,code=c,text="public description",norm_text="public description",source="dict_pref")
                          for s,c in [("ICD","I10"),("SNOMED","38341003")]])
    aliases.to_parquet(tmp_path/"aliases.parquet")
    monkeypatch.setattr(cache,"term_dir",lambda cfg:tmp_path)
    monkeypatch.setattr(cache,"directory",lambda cfg,profile:query_dir)
    class Char:
        def __init__(self,rows):self.codes=rows.code.to_numpy(dtype=str)
        def score_all(self,texts):return np.full((len(texts),len(self.codes)),.75)
    class Word:
        def __init__(self,rows):self.code=rows.code.iloc[0]
        def search(self,texts,k):return [[(self.code,1.)] for _ in texts]
    monkeypatch.setattr(cache,"CharIndex",Char)
    monkeypatch.setattr(cache,"BM25Index",Word)
    # Serialization itself is not under test; keep local synthetic index
    # classes out of pickle while preserving the completion-file contract.
    monkeypatch.setattr(cache.joblib,"dump",lambda value,path:open(path,"wb").close())
    np.save(query_dir/"ICD_cdc1_char_scores.npy",np.zeros((1,1),dtype=np.float16))
    cache.build_lexical({})
    assert np.load(query_dir/"ICD_cdc1_char_scores.npy")[0,0]==.75
    assert (query_dir/"ICD_cdc1_lexical_complete.json").exists()


def test_failed_test_status_does_not_unlock_all_label_rebuild(tmp_path):
    path=tmp_path/"runs/test";path.mkdir(parents=True)
    (path/"STATUS").write_text("FAILED\n")
    with pytest.raises(RuntimeError,match="completed sealed-test"):
        cache.all_rows(dict(root=tmp_path,paths=dict(run_dir="runs")))
