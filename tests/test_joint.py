import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import termmap.decision.coupling as joint


def test_type_fitting_excludes_held_alias_including_official_tag(monkeypatch,tmp_path):
    pd.DataFrame({"heldout_norm":["held"]}).to_parquet(tmp_path/"probe_set.parquet")
    monkeypatch.setattr(joint,"term_dir",lambda cfg:tmp_path)
    captured={}
    class Spy:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def fit(self,X,y):
            captured["X"]=X.copy()
            return LogisticRegression(**self.kwargs).fit(X,y)
    monkeypatch.setattr(joint,"LogisticRegression",Spy)
    def row(text,typ):return dict(entity=text,subconcept=typ,views={"V1":[text]})
    records=dict(train=[row("safe one","DiagnosisCode"),row("held (disorder)","ProcedureCode"),row("safe two","ProcedureCode")],probe=[row("held","Unknown")])
    retriever=SimpleNamespace(qid={"safe one":0,"held (disorder)":1,"held":1,"safe two":2},qvectors=np.array([[1,0],[9,9],[0,1]],dtype=np.float32))
    cfg=dict(root=tmp_path,paths=dict(model_dir="models"))
    joint.infer_types(cfg,retriever,records,"C1")
    np.testing.assert_array_equal(captured["X"],np.array([[1,0],[0,1]],dtype=np.float32))
    audit=json.loads((tmp_path/"models"/"type_eval_C1.json").read_text())
    assert audit["train_rows"]==2 and audit["excluded_rows"]==1
    assert records["probe"][0]["inferred_subconcept"] in {"DiagnosisCode","ProcedureCode"}


def test_test_only_queries_reuse_train_classifier_without_test_labels(monkeypatch,tmp_path):
    pd.DataFrame({"heldout_norm":[]}).to_parquet(tmp_path/"probe_set.parquet")
    monkeypatch.setattr(joint,"term_dir",lambda cfg:tmp_path)
    cfg=dict(root=tmp_path,paths=dict(model_dir="models"),generation="new")
    def row(text,typ):return dict(entity=text,subconcept=typ,views={"V1":[text]})
    records=dict(train=[row("one","DiagnosisCode"),row("two","ProcedureCode")])
    retriever=SimpleNamespace(qid={"one":0,"two":1},qvectors=np.eye(2,dtype=np.float32))
    joint.infer_types(cfg,retriever,records,"C1")
    test=dict(test=[row("new public query","Unknown")])
    queries=SimpleNamespace(qid={"new public query":0},qvectors=np.array([[1.,0]],dtype=np.float32))
    before=joint.type_cache_path(cfg,"C1").read_bytes()
    joint.infer_types(cfg,queries,test,"test_adapted")
    assert test["test"][0]["inferred_subconcept"]=="DiagnosisCode"
    assert joint.type_cache_path(cfg,"C1").read_bytes()==before
