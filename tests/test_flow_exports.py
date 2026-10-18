import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from termmap.decision.layer import apply_decision,cross_fit
from termmap.evaluation.export import export_predictions


def fixture():
    matrix=np.zeros((2,26));matrix[:,0]=[1.,-1.]
    features=SimpleNamespace(icd={c:dict(is_leaf=True,parent="",pre_in=i,pre_out=i,unspecified=False) for i,c in enumerate(["A12.1","A12.2"])},
        sn={"100":{},"200":{}},exceptions=set(),co={},children={},freq={},
        similarity=lambda a,b:0.,matrix=lambda item:matrix.copy(),descend=lambda row,codes,scores:(codes,scores))
    other=dict(row=dict(row_id="aux:r",system="SNOMED",code="",entity="x",group_id="g",is_auxiliary=True),pool=[("100",1.),("200",0.)],features={})
    primary=dict(row=dict(row_id="r",system="ICD",code="A12.1",entity="x",group_id="g",subconcept="DiagnosisCode"),
                 pool=[("A12.1",1.),("A12.2",0.)],features={},other_system=other)
    return features,primary,matrix


def test_auxiliary_decisions_export_without_creating_gold_labels(tmp_path):
    features,item,matrix=fixture()
    model=dict(model=SimpleNamespace(decision_function=lambda x:x[:,0]),offset=0.,w=0.,tau=1.,calibration=dict(temperature=1.,correct=.8,missing=.1))
    pools,conf,missing,aux=apply_decision(model,[item],[matrix],features,return_aux=True)
    assert aux[0]["pool"][0][0]=="100" and item["other_system"]["row"]["code"]==""
    cfg=dict(root=tmp_path,paths=dict(run_dir="runs"))
    output=tmp_path/"runs"/"F00";output.mkdir(parents=True)
    (output/"cv_manifest.json").write_text(json.dumps([dict(fit_groups=["other"],evaluated_groups=["g"])]))
    export_predictions(cfg,"F00","dev",[item],pools,conf,missing,.2,aux,features)
    output=tmp_path/"runs"/"F00"
    frame=pd.read_csv(output/"dev_predictions.csv",dtype=str)
    assert frame.loc[0,"icd10cm_code"]=="A12.1" and frame.loc[0,"snomed_code"]=="100"
    result=json.loads((output/"dev_validation.json").read_text())
    assert result["passed"] and not result["auxiliary_labels_used"]


def test_cross_fitted_export_retains_disjoint_group_manifest():
    features,item,matrix=fixture();items=[];matrices=[]
    for i in range(10):
        row=dict(item["row"],row_id=str(i),group_id=str(i))
        other=dict(item["other_system"],row=dict(item["other_system"]["row"],row_id="aux:"+str(i),group_id=str(i)))
        items.append(dict(item,row=row,other_system=other));matrices.append(matrix.copy())
    pools,conf,missing,model,choices,thresholds,without,aux=cross_fit(items,matrices,features)
    assert len(pools)==len(aux)==10 and all(x is not None for x in aux)
    assert len(model["cv_manifest"])==5
    for fold in model["cv_manifest"]:
        assert not set(fold["fit_groups"])&set(fold["evaluated_groups"])
