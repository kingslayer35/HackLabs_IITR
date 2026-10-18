from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

from termmap.retrieval.candidates import CachedRetriever
from termmap.retrieval.lexical import CharIndex,BM25Index
from termmap.terminology.tables import abbreviation_alias_support
from termmap.preprocessing.normalize import Normalizer
from termmap.reranking.model import Cards


def names():
    return pd.DataFrame([
        dict(system="SNOMED",code="100",text="alpha beta condition",norm_text="alpha beta condition",source="dict_pref"),
        dict(system="SNOMED",code="100",text="abc",norm_text="abc",source="abbr_variant"),
        dict(system="SNOMED",code="200",text="abc",norm_text="abc",source="dict_pref"),
    ])


def test_alias_support_preserves_public_and_other_group_evidence():
    abbr=pd.DataFrame([dict(abbr="abc",expansion="alpha beta condition",source="train_initials",ambiguous=False)])
    train=pd.DataFrame([dict(entity="abc",display="Alpha beta condition",valid_gold=True,group_id=g) for g in ["g1","g2"]])
    support=abbreviation_alias_support(names(),abbr,train)
    assert support.iloc[0].support_groups==["g1","g2"]
    public=pd.concat([abbr,pd.DataFrame([dict(abbr="abc",expansion="alpha beta condition",source="meta_inventory",ambiguous=False)])])
    assert abbreviation_alias_support(names(),public,train).empty


def test_self_only_alias_is_hidden_in_all_search_channels():
    aliases=names();char=CharIndex(aliases);word=BM25Index(aliases);key=("SNOMED",True)
    # Small fixed scores stand in for encoded vectors; this check needs no GPU.
    retriever=object.__new__(CachedRetriever)
    retriever.qid={"abc":0};retriever.qvectors=np.array([[1.,0.]])
    retriever.system_qvectors={"SNOMED":retriever.qvectors}
    retriever.self_only_aliases=defaultdict(set,{("SNOMED","g1"):{("100","abc")}})
    retriever.char={key:char.score_all(["abc"])};retriever.dense={key:np.array([[1.,1.]])}
    retriever.word={key:[[("100",5.),("200",5.)]]}
    retriever.codes={key:char.codes};retriever.code_index={key:{c:i for i,c in enumerate(char.codes)}}
    retriever.dcodes={"SNOMED":char.codes};retriever.dcode_index={"SNOMED":retriever.code_index[key]}
    retriever.lex_objects={key:(char,word)};retriever.alias_rows={"SNOMED":aliases}
    retriever.alias_indices={"SNOMED":aliases.groupby("code").indices}
    retriever.embeddings={"SNOMED":np.array([[0.,1.],[1.,0.],[1.,0.]])}
    retriever.exact={key:{"abc":{"100","200"}}}
    retriever.memory={"SNOMED":pd.DataFrame(columns=["code","norm_text","group_id"])}
    retriever.memory_scores={"SNOMED":np.zeros((1,0))}
    row=dict(lexical_queries=["abc"],dense_queries=["abc"],excluded_group="g1")
    channels,exact,_,_,_,_=retriever.basic(row,"SNOMED")
    assert exact==["200"]
    assert dict(channels["dense"])["100"]==0
    assert dict(channels["dense"])["200"]==1
    assert dict(channels["char"])["100"]<dict(channels["char"])["200"]
    assert dict(channels["bm25"]).get("100",0)==0
    assert dict(channels["bm25"])["200"]>0
    # The learned sense is available for an unrelated group.
    channels,exact,*_=retriever.basic(dict(row,excluded_group="g2"),"SNOMED")
    assert exact==["100","200"]
    assert dict(channels["dense"])["100"]==1


def test_preprocessing_cannot_complete_from_own_group_alias(tmp_path):
    aliases=pd.DataFrame([
        dict(system="SNOMED",code="100",text="alpha condition",norm_text="alpha condition",source="dict_pref"),
        dict(system="SNOMED",code="100",text="alpha beta chronic disease",norm_text="alpha beta chronic disease",source="abbr_variant"),
    ])
    train=pd.DataFrame([dict(entity="alpha beta chronic disease",chunk_text="",display="alpha condition",group_id="g1",valid_gold=True)])
    support=pd.DataFrame([dict(system="SNOMED",code="100",norm_text="alpha beta chronic disease",support_groups=["g1"])])
    normalizer=Normalizer(aliases,train,alias_support=support);normalizer.tightened=True
    with (tmp_path/"normalizer.pkl").open("wb") as stream:pickle.dump(normalizer,stream,protocol=pickle.HIGHEST_PROTOCOL)
    with (tmp_path/"normalizer.pkl").open("rb") as stream:normalizer=pickle.load(stream)
    assert normalizer.views("alpha beta chronic dis",exclude_group="g1")["V4"]==["alpha beta chronic dis"]
    assert normalizer.views("alpha beta chronic dis",exclude_group="g2")["V4"]==["alpha beta chronic disease"]


def test_candidate_card_masks_only_own_group_learned_alias():
    cards=object.__new__(Cards);cards.held=set()
    cards.sn={"100":dict(preferred="alpha beta condition",fsn="alpha beta condition (finding)",tag="finding",top_level="Clinical finding")}
    cards.aliases=defaultdict(list,{("SNOMED","100"):["alpha beta condition","abcd"]})
    cards.examples=defaultdict(list)
    cards.self_only_names=defaultdict(set,{("SNOMED","100","g1"):{"abcd"}})
    row=dict(system="SNOMED",excluded_group="g1")
    assert "abcd" not in cards.card(row,"100")
    assert "alpha beta condition" in cards.card(row,"100")
    assert "abcd" in cards.card(dict(row,excluded_group="g2"),"100")


def test_shared_terminology_produces_identical_reranker_inputs(tmp_path,monkeypatch):
    icd=pd.DataFrame([dict(code="I10",preferred="Hypertension",parent="",category="I10",is_leaf=True,refinement_text="",seventh_char_meaning="")])
    sn=pd.DataFrame([
        dict(code="100",preferred="alpha beta condition",fsn="alpha beta condition (finding)",tag="finding",top_level="Clinical finding"),
        dict(code="200",preferred="abc",fsn="abc (finding)",tag="finding",top_level="Clinical finding"),
    ])
    aliases=pd.concat([names(),pd.DataFrame([dict(system="ICD",code="I10",text="Hypertension",norm_text="hypertension",source="dict_pref")])],ignore_index=True)
    for name,frame in [("icd_concepts",icd),("snomed_concepts",sn),("aliases",aliases)]:frame.to_parquet(tmp_path/(name+".parquet"),index=False)
    pd.DataFrame([dict(row_id="0",system="SNOMED",code="100",entity="abc",group_id="g1",valid_gold=True)]).to_parquet(tmp_path/"train.parquet",index=False)
    pd.DataFrame({"heldout_norm":["withheld"]}).to_parquet(tmp_path/"probe_set.parquet",index=False)
    pd.DataFrame([dict(system="SNOMED",code="100",norm_text="abc",support_groups=["g1"])]).to_parquet(tmp_path/"abbreviation_alias_support_train.parquet",index=False)
    monkeypatch.setattr("termmap.reranking.model.term_dir",lambda cfg:tmp_path)
    cfg={"retrieval":{"enabled_views":["V1"]}}
    reference=Cards(cfg)
    shared=Cards(cfg,terminology=dict(icd=icd.set_index("code").to_dict("index"),sn=sn.set_index("code").to_dict("index"),aliases=aliases))
    for system,code in [("ICD","I10"),("SNOMED","100"),("SNOMED","200")]:
        row=dict(system=system,entity="abc",excluded_group="g1",subconcept="DiagnosisCode",views={"V1":["abc"]},context={"window":"[M]abc[/M]"})
        assert shared.payload(row,code)==reference.payload(row,code)
