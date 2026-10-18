import numpy as np
import pandas as pd
import torch
from collections import Counter
from termmap.pretraining.trainer import (FrozenIndex,concept_scores,gold_rank,info_nce,mask_tensor,memory_rank,
                              parameter_groups,schedule,score_examples,view_index,weight_rare)
from termmap.retrieval.lexical import rank_codes


def toy_index(seed=0):
    rng=np.random.default_rng(seed)
    rows=pd.DataFrame(dict(code=["b","a","a","c","c","d","e"],
                           text=["heart failure","heart failure","cardiac failure","renal failure","kidney failure","edema","Edema (finding)"],
                           source=["dict_pref"]*7))
    v=rng.normal(size=(len(rows),8)).astype(np.float32);v/=np.linalg.norm(v,axis=1,keepdims=True)
    return FrozenIndex("SNOMED",rows,v)


def example(index,code,texts,keys=None,self_keys=()):
    keys=index.gold_keys(code)|set(texts) if keys is None else keys
    return dict(system="SNOMED",code=code,gold=index.code_index[code],texts=texts,masked=index.masked(code,keys,self_keys),
                group_id="g",bucket=None,variant=None)


def test_concept_scores_are_max_over_views_and_names():
    index=toy_index();rng=np.random.default_rng(1)
    q=torch.tensor(rng.normal(size=(3,8)),dtype=torch.float32);q=torch.nn.functional.normalize(q,dim=-1)
    views=torch.tensor([[0,1],[2,2]])
    got=concept_scores(q,views,index).numpy()
    alias=np.maximum(q[[0,2]].numpy()@index.vectors.T,q[[1,2]].numpy()@index.vectors.T)
    want=np.array([[alias[b,index.alias_code==c].max() for c in range(len(index.codes))] for b in range(2)])
    np.testing.assert_allclose(got,want,rtol=1e-6)


def test_gold_rank_equals_pipeline_rank_codes_with_ties():
    rng=np.random.default_rng(2);codes=np.array(sorted(f"c{i:03d}" for i in range(300)),dtype=object)
    scores=np.round(rng.uniform(-1,1,size=(20,300)),2).astype(np.float32)  # many exact ties
    gold=rng.integers(0,300,20)
    got=gold_rank(torch.tensor(scores),torch.tensor(gold)).numpy()
    for b in range(20):
        ranked=[c for c,_ in rank_codes(scores[b].astype(np.float16).astype(np.float32),codes,300)]
        assert got[b]==ranked.index(codes[gold[b]])+1


def test_masks_hide_shared_names_of_other_codes_only():
    index=toy_index()
    e=example(index,"a",["heart failure"])
    assert index.rows.code[e["masked"]].tolist()==["b"]  # same text under another code
    assert index.usable("a",e["masked"])
    # An alias query also hides the gold's own copy of the query name (leave-self-out).
    s=example(index,"a",["heart failure"],self_keys={"heart failure"})
    assert sorted(index.rows.text[s["masked"]])==["heart failure","heart failure"]
    assert index.usable("a",s["masked"])
    # FSN tags are ignored when matching names.
    assert index.gold_keys("e")=={"edema"} and index.rows.code[example(index,"e",["edema"])["masked"]].tolist()==["d"]


def test_info_nce_ignores_masked_rows_and_matches_manual():
    index=toy_index();q=torch.nn.functional.normalize(torch.randn(1,8,generator=torch.Generator().manual_seed(3)),dim=-1)
    e=example(index,"a",["heart failure"]);views=view_index([e],{"heart failure":0},"cpu")
    scores=concept_scores(q,views,index,mask_tensor([e],len(index.rows),"cpu"))
    loss=info_nce(scores,torch.tensor([e["gold"]]),.03)
    alias=(q.numpy()@index.vectors.T)[0];alias[e["masked"]]=-np.inf
    per=np.array([alias[index.alias_code==c].max() for c in range(len(index.codes))])/.03
    want=np.log(np.exp(per[np.isfinite(per)]).sum())-per[e["gold"]]
    np.testing.assert_allclose(loss.item(),want,rtol=1e-5)
    assert not np.isfinite(scores[0,index.code_index["b"]].item())


def test_gradient_reaches_queries_only_and_is_finite():
    index=toy_index();q=torch.nn.functional.normalize(torch.randn(2,8,generator=torch.Generator().manual_seed(4)),dim=-1).requires_grad_()
    ex=[example(index,"a",["x"]),example(index,"c",["y"])]
    scores=concept_scores(q,view_index(ex,{"x":0,"y":1},"cpu"),index,mask_tensor(ex,len(index.rows),"cpu"))
    info_nce(scores,torch.tensor([e["gold"] for e in ex]),.03).sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
    assert not index.tensors("cpu")["vectors"].requires_grad


def test_view_padding_repeats_last_view():
    ex=[dict(texts=["a","b","c"]),dict(texts=["d"])]
    assert view_index(ex,{t:i for i,t in enumerate("abcd")},"cpu").tolist()==[[0,1,2],[3,3,3]]


def test_memory_rank_excludes_same_group():
    q=torch.eye(3);views=torch.tensor([[0],[1]])
    memory=dict(vectors=torch.eye(3),group=np.array(["g1","g2","g3"]),code=torch.tensor([0,1,2]))
    rank=memory_rank(q,views,memory,torch.tensor([0,1]),["g1","other"],3)
    assert rank.tolist()==[10**9,1]  # row 0's only gold mention is its own chunk group


def test_score_examples_matches_parts():
    index=toy_index();q=torch.nn.functional.normalize(torch.randn(2,8,generator=torch.Generator().manual_seed(5)),dim=-1)
    ex=[example(index,"a",["x"]),example(index,"d",["y"])];position={"x":0,"y":1}
    rank,loss,_=score_examples(q,position,ex,index,.03)
    scores=concept_scores(q,view_index(ex,position,"cpu"),index)
    assert rank.tolist()==gold_rank(scores,torch.tensor([e["gold"] for e in ex])).tolist() and np.isfinite(loss).all()


def test_schedule_warmup_then_linear_decay():
    values=[schedule(s,100,10) for s in range(100)]
    assert values[0]==.1 and values[9]==1 and values[10]==1 and abs(values[55]-.5)<1e-9 and values[-1]>0 and schedule(100,100,10)==0


def test_no_weight_decay_on_bias_and_layernorm():
    from transformers import BertConfig,BertModel
    model=BertModel(BertConfig(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,vocab_size=50))
    decay,other=parameter_groups(model,.01)
    names={id(p):n for n,p in model.named_parameters()}
    assert all(not (names[id(p)].endswith("bias") or "LayerNorm" in names[id(p)]) for p in decay["params"])
    assert all(names[id(p)].endswith("bias") or "LayerNorm" in names[id(p)] for p in other["params"]) and other["weight_decay"]==0
    assert len(decay["params"])+len(other["params"])==len(list(model.parameters()))


def test_rare_codes_weigh_more_and_mean_is_one():
    ex=[dict(system="ICD",code=c) for c in ["a","b","b","b","b"]]
    w=[e["weight"] for e in weight_rare(ex,Counter({("ICD","a"):1,("ICD","b"):4}))]
    assert abs(np.mean(w)-1)<1e-12 and w[0]==2*w[1]


def test_index_tensors_built_in_inference_mode_still_train():
    index=toy_index()
    with torch.inference_mode():index.tensors("cpu")
    q=torch.nn.functional.normalize(torch.randn(1,8),dim=-1).requires_grad_()
    e=example(index,"a",["x"])
    info_nce(concept_scores(q,view_index([e],{"x":0},"cpu"),index),torch.tensor([e["gold"]]),.03).sum().backward()
    assert torch.isfinite(q.grad).all()
