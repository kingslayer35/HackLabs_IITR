import numpy as np
from termmap.decision.layer import Features


def policy():
    f=Features.__new__(Features)
    f.icd={
        "A12":dict(is_leaf=False,children=["A12.1","A12.9"],refinement_text="",unspecified=False),
        "A12.1":dict(is_leaf=True,children=[],refinement_text="right",unspecified=False,seventh_char_meaning=""),
        "A12.9":dict(is_leaf=True,children=[],refinement_text="unspecified",unspecified=True,seventh_char_meaning=""),
    };f.exceptions=set();f.seventh={}
    return f


def test_header_descent_and_context():
    f=policy();row=dict(system="ICD",entity="condition",context=dict(window="[M]condition[/M]",cues={}))
    codes,scores=f.descend(row,["A12","A12.1","A12.9"],np.array([3.,2.,1.]))
    assert codes[int(np.argmax(scores))]=="A12.9"
    row["context"]["window"]="right [M]condition[/M]"
    codes,scores=f.descend(row,["A12","A12.1","A12.9"],np.array([3.,1.,2.]))
    assert codes[int(np.argmax(scores))]=="A12.1"
    f.exceptions={"A12"};row["context"]["window"]="condition"
    codes,scores=f.descend(row,["A12","A12.9"],np.array([3.,2.]))
    assert codes[int(np.argmax(scores))]=="A12"


def test_drill_never_reintroduces_gold_child():
    f=policy();row=dict(system="ICD",entity="right condition",context=dict(window="right condition",cues={}),forbidden_code="A12.1")
    codes,scores=f.descend(row,["A12","A12.9"],np.array([3.,2.]))
    assert "A12.1" not in codes
    assert codes[int(np.argmax(scores))]=="A12.9"


def test_coupling_header_exception_ignores_forbidden_drill_child():
    from termmap.decision.layer import legal_icd
    f=policy();f.exceptions={"A12"}
    row=dict(system="ICD",entity="right condition",context=dict(window="right condition",cues={}),forbidden_code="A12.1")
    assert legal_icd(row,"A12",f)
    assert not legal_icd(row,"A12.1",f)


def test_singleton_confidence_is_never_certain():
    from termmap.decision.layer import confidence
    item=dict(row=dict(system="SNOMED",entity="example",group_id="g"),pool=[("100",0.)],features={})
    model=dict(temperature=1.,correct=1.,missing=0.)
    value,missing=confidence(model,[item],[[('100',0.)]],None)
    assert 0<value[0]<1 and missing[0]==0


def test_all_type_objective_includes_non_diagnosis_and_excludes_auxiliary():
    from termmap.decision.layer import headline
    def item(system,code,kind,aux=False):
        return dict(row=dict(system=system,code=code,subconcept=kind,is_auxiliary=aux))
    items=[item("ICD","I10","DiagnosisCode"),item("ICD","Z99.3","ProcedureCode"),
           item("SNOMED","100","DiagnosisCode"),item("SNOMED","200","ProcedureCode"),
           item("SNOMED","","ProcedureCode",True)]
    predictions=["I10","Z99.3","100","300","400"]
    assert headline(items,predictions,"diagnosis")==1.
    assert headline(items,predictions,"all_types")==.5


def test_coupling_guard_rejects_unseen_loss_despite_equal_headline():
    from scripts.decision.fit import keep_coupling
    def item(system,code):return dict(row=dict(system=system,code=code,subconcept="DiagnosisCode"))
    items=[item("ICD","A12.1"),item("ICD","A12.2"),item("SNOMED","100"),item("SNOMED","200")]
    on=[[('A12.2',1.)],[('A12.2',1.)],[('100',1.)],[('200',1.)]]
    off=[[('A12.1',1.)],[('A12.1',1.)],[('100',1.)],[('200',1.)]]
    probe=[item("ICD","A12.1"),item("SNOMED","100")];pp=[[('A12.1',1.)],[('100',1.)]]
    check=keep_coupling(items,on,off,probe,pp,pp,{('ICD','A12.2'):3})
    assert check['with_min_acc']==check['without_min_acc']==50.
    assert not check['keep']


def test_feature_reference_breaks_reranker_ties_by_code_string():
    from termmap.decision.layer import FEATURES
    f=Features.__new__(Features)
    f.icd={c:dict(parent="",depth=0,is_leaf=True,preferred="example") for c in ["A12.1","B12.1"]}
    f.sn={};f.freq={};f.tags={};f.tops={};f.seventh={};f.evidence=lambda row,code:(0.,0.)
    item=dict(row=dict(system="ICD",entity="example",subconcept="DiagnosisCode"),pool=[("B12.1",.2),("A12.1",.1)],
              reranker_scores={"B12.1":1.,"A12.1":1.},features={})
    assert f.matrix(item)[:,FEATURES.index("same_category")].tolist()==[0.,1.]


def test_incremental_coupling_threshold_matches_exhaustive_breakpoints():
    from types import SimpleNamespace
    from termmap.decision.layer import fit_coupling,couple,headline
    rng=np.random.default_rng(13);items=[];pools=[];sims={}
    icd={"A12.1":dict(is_leaf=True),"A12.2":dict(is_leaf=True)}
    for i in icd:
        for s in ["100","200"]:sims[i,s]=round(float(rng.uniform(.1,.9)),5)
    f=SimpleNamespace(icd=icd,exceptions=set(),co={("A12.1","100"):.8},similarity=lambda i,s:sims[i,s])
    for group in range(3):
        for system,codes in [("ICD",list(icd)),("SNOMED",["100","200"])]:
            scores=rng.normal(size=2)
            pools.append(sorted(zip(codes,scores),key=lambda x:(-x[1],x[0])))
            items.append(dict(row=dict(system=system,entity=str(group),group_id=str(group),chunk_text="",subconcept="DiagnosisCode",code=codes[int(rng.integers(2))])))
    w,tau=fit_coupling(items,pools,f)
    value=headline(items,[p[0][0] for p in couple(items,pools,f,w,tau)])
    expected=max(headline(items,[p[0][0] for p in couple(items,pools,f,weight,threshold)]) for weight in [0,.5,1,2] for threshold in [*sims.values(),max(sims.values())+1e-5])
    assert value==expected


def test_unlabelled_opposite_system_is_not_a_second_gold_label():
    from types import SimpleNamespace
    from termmap.decision.layer import fit_coupling,couple,headline,joint_candidates
    f=SimpleNamespace(icd={"A12.1":dict(is_leaf=True),"A12.2":dict(is_leaf=True)},exceptions=set(),co={},similarity=lambda i,s:.9 if (i,s)==("A12.1","100") else .1,matrix=lambda item:np.zeros((len(item['pool']),26)))
    primary=dict(row=dict(row_id="r1",system="ICD",entity="x",group_id="g",chunk_text="",subconcept="DiagnosisCode",code="A12.1"),pool=[("A12.2",.2),("A12.1",0.)])
    primary["other_system"]=dict(row=dict(row_id="aux:r1",system="SNOMED",entity="x",group_id="g",chunk_text="",subconcept="DiagnosisCode",code="",is_auxiliary=True),pool=[("100",.2),("200",0.)])
    joint,matrices,positions,partners=joint_candidates([primary],[np.zeros((2,26))],f)
    assert positions==[0] and partners=={0:1}
    assert joint[0]['row']['decision_pair_id']==joint[1]['row']['decision_pair_id']=='r1'
    pools=[x['pool'] for x in joint];w,tau=fit_coupling(joint,pools,f)
    result=couple(joint,pools,f,w,tau)
    assert headline(joint,[p[0][0] for p in result])==1.
