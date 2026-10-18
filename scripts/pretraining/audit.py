"""CPU audit of the P2-v2 training data; every check is a hard assertion."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from termmap.common.io import load_config,term_dir,path_for,write_json
from termmap.pretraining.trainer import SYSTEMS,load_data,key,pinned


def main():
    cfg=pinned(load_config());settings=dict(cfg["training_v2"]);settings["max_length"]=cfg["training"]["max_length"]
    out=term_dir(cfg);indexes,train,sets,sampler,skipped=load_data(cfg,settings)
    tr=pd.read_parquet(out/"train.parquet");dv=pd.read_parquet(out/"dev.parquet");probe=pd.read_parquet(out/"probe_set.parquet")
    held=set(probe.heldout_norm);probe_codes=set(zip(probe.system,probe.code));report={"skipped":skipped}
    # 1. Index = exactly the rows the dense channel searches (ICD without CDC names under the current policy).
    for s,index in indexes.items():
        aliases=pd.read_parquet(out/"aliases.parquet");aliases=aliases.loc[aliases.system.eq(s)]
        if s=="ICD" and not cfg["retrieval"].get("cdc_enabled",True):aliases=aliases.loc[~aliases.source.isin(["icd_index","icd_inclusion"])]
        assert len(index.rows)==len(aliases)==len(index.vectors),s
        assert not index.rows.norm_text.isin(held).any(),"probe name left in the frozen index"
        norms=np.linalg.norm(index.vectors[:5000].astype(np.float32),axis=1)
        assert np.allclose(norms,1,atol=2e-3),"stored alias vectors are not unit length"
        report[f"index/{s}"]=dict(aliases=len(index.rows),concepts=len(index.codes),sources=index.rows.source.value_counts().to_dict())
    # 2. Every valid train row is one example; nothing from dev, test or the probe.
    valid=tr.loc[tr.valid_gold]
    assert len(train)+sum(skipped["train_skipped"].values())==len(valid)
    assert len({e["row_id"] for e in train})==len(train)
    assert {e["row_id"] for e in train}<=set(valid.row_id.astype(str))
    assert not {e["row_id"] for e in train}&set(dv.row_id.astype(str))
    assert not {e["group_id"] for e in train}&set(dv.group_id.astype(str)),"a chunk group spans train and dev"
    assert not any(key(t) in held for e in train for t in e["texts"]),"probe name used as a training query"
    assert not {(e["system"],e["code"]) for e in train}&probe_codes
    assert len(sets["dev"])==int(dv.valid_gold.sum())
    report["train"]=dict(examples=len(train),by_system=dict(Counter(e["system"] for e in train)),
                         views=dict(Counter(len(e["texts"]) for e in train)))
    # 3. Masks: never hide every gold name; mention masks never hide a gold name at all.
    for name,examples in [("train",train),*sets.items()]:
        for e in examples:
            index=indexes[e["system"]];gold=index.by_code[e["code"]]
            assert np.setdiff1d(gold,e["masked"]).size,(name,e["row_id"])
            assert not np.intersect1d(gold,e["masked"]).size,(name,e["row_id"],"mention mask hides a gold name")
    report["masks"]={name:dict(rows_with_mask=int(sum(len(e["masked"])>0 for e in ex)),mean_masked=float(np.mean([len(e["masked"]) for e in ex])))
                     for name,ex in [("train",train),*sets.items()]}
    # 4. Rare-code weights: every row counts, rare codes count more.
    w=np.array([e["weight"] for e in train]);report["weights"]=dict(mean=float(w.mean()),min=float(w.min()),max=float(w.max()))
    assert abs(w.mean()-1)<1e-9 and w.min()>0
    # 5. Alias queries: outside the probe, query names owned by one code, deterministic, about half noisy.
    n=int(round(settings["alias_query_ratio"]*len(train)));a=sampler.sample(1,n);b=sampler.sample(1,n);c=sampler.sample(2,n)
    assert [(e["code"],e["texts"]) for e in a]==[(e["code"],e["texts"]) for e in b],"alias sampling is not deterministic"
    assert [(e["code"],e["texts"]) for e in a]!=[(e["code"],e["texts"]) for e in c],"alias sampling repeats across epochs"
    assert len(a)==n and not {(e["system"],e["code"]) for e in a}&probe_codes
    for e in a:
        index=indexes[e["system"]];gold=index.by_code[e["code"]]
        assert np.setdiff1d(gold,e["masked"]).size
        assert all(index.keys[i]!=key(e["texts"][0]) for i in np.setdiff1d(gold,e["masked"])),"alias query can see its own name"
        assert key(e["texts"][0]) not in held
    report["alias"]=dict(per_epoch=n,by_system=dict(Counter(e["system"] for e in a)),by_source=dict(Counter(e["source"] for e in a)),
                         by_noise=dict(Counter(e["noise"] for e in a)),examples=[(e["system"],e["code"],e["texts"][0],e["noise"]) for e in a[:6]+a[-6:]])
    sn=pd.read_parquet(out/"snomed_concepts.parquet",columns=["code","tag"]).set_index("code").tag
    report["alias"]["snomed_tags"]=dict(Counter(sn.reindex([e["code"] for e in a if e["system"]=="SNOMED"])).most_common(8))
    # 6. Cross-system wording overlap between dev and train (inherent to the data; reported, not removed).
    train_text={t for e in train for t in e["texts"]};dev_text=[e["texts"][0] for e in sets["dev"]]
    report["dev_v1_text_seen_in_train_any_system"]=int(sum(t in train_text for t in dev_text))
    report["eval_sets"]={k:dict(n=len(v),by_system=dict(Counter(e["system"] for e in v)),buckets=dict(Counter(e["bucket"] for e in v))) for k,v in sets.items()}
    target=path_for(cfg,"run_dir")/"pretrain_v2"/"data_audit.json";write_json(target,report)
    print(json.dumps(report,indent=1,default=str))


if __name__=="__main__":main()
