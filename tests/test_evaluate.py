import pandas as pd
from termmap.evaluation.metrics import evaluate

def test_snomed_error_classes_are_ordered():
    rows=pd.DataFrame([dict(row_id=str(i),system="SNOMED",code="gold",entity=f"m{i}",
                           chunk_text=f"c{i}",group_id=str(i),subconcept="DiagnosisCode") for i in range(4)])
    concepts={"gold":dict(tag="disorder",top_level="Clinical finding",core_key="x"),
              "near":dict(tag="disorder",top_level="Clinical finding",core_key="x"),
              "same_tag":dict(tag="disorder",top_level="Clinical finding",core_key="y"),
              "same_top":dict(tag="finding",top_level="Clinical finding",core_key="z"),
              "wrong":dict(tag="procedure",top_level="Procedure",core_key="w")}
    pools=[[(c,1)] for c in ["near","same_tag","same_top","wrong"]]
    metrics,details=evaluate(rows,pools,{}, {},concepts,n_boot=10,links={"gold":{"near"}})
    assert details.error_class.tolist()==["near-miss","same tag, different concept","same top-level, different tag","wrong top-level"]
    assert metrics["systems"]["SNOMED"]["top_level_accuracy"]==75

def test_training_memory_hides_all_same_group_rows():
    from termmap.retrieval.lexical import CharIndex
    aliases=pd.DataFrame([dict(code="A",text="exact mention",norm_text="exact mention",group_id="g"),
                          dict(code="B",text="other mention",norm_text="other mention",group_id="h")])
    index=CharIndex(aliases)
    assert index.search(["exact mention"],2,exclude_group="g")[0][0][0]=="B"


def test_paired_flow_delta_includes_both_correct():
    from termmap.evaluation.metrics import paired_flow_delta
    frame=pd.DataFrame([dict(row_id=f"{g}:{s}",group_id=str(g),entity=str(g),chunk_text="",system=s,
        subconcept="DiagnosisCode",bucket="unseen",correct=True) for g in range(2) for s in ["ICD","SNOMED"]])
    other=frame.copy();other.loc[1,"correct"]=False
    result=paired_flow_delta(frame,other,50,13)
    assert result["all"]["both_correct"]["delta"]==-50.


def test_paired_delta_survives_private_text_removal():
    from termmap.evaluation.metrics import paired_flow_delta
    frame=pd.DataFrame([dict(row_id=f"{g}:{s}",group_id=str(g),pair_key=f"digest{g}",system=s,
        subconcept="DiagnosisCode",bucket="unseen",correct=True) for g in range(2) for s in ["ICD","SNOMED"]])
    other=frame.copy();other.loc[1,"correct"]=False
    assert paired_flow_delta(frame,other,50,13)["all"]["both_correct"]["delta"]==-50.


def test_all_type_minimum_and_paired_ci_include_procedure_labels():
    from termmap.evaluation.metrics import paired_flow_delta
    rows=pd.DataFrame([dict(row_id=str(i),group_id="g",entity=str(i),chunk_text="",system=s,
        code=c,subconcept=kind) for i,(s,c,kind) in enumerate([
        ("ICD","I10","DiagnosisCode"),("SNOMED","100","DiagnosisCode"),("SNOMED","200","ProcedureCode")])])
    sn={c:dict(tag="finding",top_level="Clinical finding") for c in ["100","200","300"]}
    metrics,details=evaluate(rows,[[('I10',1.)],[('100',1.)],[('300',1.)]],{},
        {"I10":dict(is_leaf=True)},sn,n_boot=10)
    assert metrics["headline_min_acc"]==100.
    assert metrics["headline_min_acc_all_types"]==50.
    assert metrics["headline_min_acc_all_types_ci"]==[50.,50.]
    perfect=details.copy();perfect["correct"]=True
    delta=paired_flow_delta(perfect,details,n_boot=10)
    assert delta["all"]["min_acc"]["delta"]==0.
    assert delta["all"]["min_acc_all_types"]==dict(delta=-50.,ci=[-50.,-50.])
