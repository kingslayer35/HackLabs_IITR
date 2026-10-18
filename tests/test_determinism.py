import pandas as pd
import pytest
from termmap.common.io import load_config,term_dir,read_partition
from termmap.retrieval.lexical import rrf,rank_codes
from termmap.preprocessing.noise import corrupt
from termmap.evaluation.metrics import group_ci

def test_seeded_noise_and_bootstrap():
    assert corrupt("chronic kidney disease")==corrupt("chronic kidney disease")
    assert group_ci([0,1,1],["a","a","b"],20)==group_ci([0,1,1],["a","a","b"],20)
    assert rank_codes([.5000001,.5000002],["B","A"],2)[0][0]=="A"
    assert rrf({"a":[("B",1)],"b":[("A",1)]})[0][0]=="A"

def test_split_and_probe_isolation():
    cfg=load_config();out=term_dir(cfg)
    splits=pd.read_parquet(out/"splits.parquet")
    assert splits.groupby("group_id").split.nunique().max()==1
    with pytest.raises(ValueError):read_partition(cfg,"test")
    probe=pd.read_parquet(out/"probe_set.parquet")
    aliases=pd.read_parquet(out/"aliases.parquet")
    assert not set(probe.heldout_norm)&set(aliases.norm_text)
    for s in ["ICD","SNOMED"]:assert probe.loc[probe.system.eq(s),"code"].nunique()==1500

def test_preserved_icd_subconcept():
    out=term_dir(load_config())
    train=read_partition(load_config(),"train")
    dev=read_partition(load_config(),"dev")
    for frame in [train,dev]:
        if (frame.system.eq("ICD") & frame.code.eq("Z99.3") & frame.subconcept.eq("ProcedureCode")).any():
            return
    # The row may legitimately be sealed; aggregate metadata, never inspect test.
    import json
    c=json.loads((out/"split_checks.json").read_text())["system_subconcept_counts"]
    assert c["ICD/ProcedureCode"]==1

def test_training_tables_and_probe_exclusions():
    cfg=load_config();out=term_dir(cfg);train=read_partition(cfg,"train")
    memory=pd.read_parquet(out/"train_memory.parquet")
    assert set(memory.row_id)<=set(train.row_id)
    assert not memory.code.eq("L6.0").any()
    probe=pd.read_parquet(out/"probe_set.parquet")
    assert not set(memory.norm_text)&set(probe.heldout_norm)
    top=pd.read_parquet(out/"top_level_priors_train.parquet")
    assert ((top.groupby("subconcept").probability.sum()-1).abs()<1e-12).all()
