"""Exact ICD scores for the actual top-five SNOMED preferred-term queries."""
import gc
import json
import numpy as np
from termmap.common.io import path_for,write_json
from termmap.retrieval.cache import directory
from termmap.retrieval.lexical import rrf


def build_bridge_index(cfg,model_path,run,profile="eval",datasets=None,smoke=False):
    import joblib
    import torch
    from termmap.retrieval.dense import Encoder,DenseIndex
    from termmap.retrieval.candidates import CachedRetriever
    root=directory(cfg,profile);target=root/run
    if (target/"bridge_complete.json").exists() and not smoke:return
    retriever=CachedRetriever(cfg,run,profile);records=joblib.load(root/"records.joblib");selected=set()
    for dataset,rows in records.items():
        if datasets is not None and dataset not in datasets:continue
        for i,row in enumerate(rows[:2] if smoke else rows):
            try:channels,*_=retriever.snomed_basic(row)
            except KeyError:
                if smoke:continue
                raise
            selected.update(c for c,_ in rrf(channels,60,5))
            if i%500==0:print("Bridge preferred queries",run,dataset,i,"unique",len(selected),flush=True)
    codes=sorted(selected);excluded=set()
    if profile!="final":
        import pandas as pd
        from termmap.common.io import term_dir
        excluded=set(pd.read_parquet(term_dir(cfg)/"probe_set.parquet").heldout_norm)
    from termmap.preprocessing.normalize import clean
    from termmap.terminology.hierarchy import strip_tag
    names=[]
    for code in codes:
        text=retriever.sn[code]["preferred"]
        if clean(strip_tag(text)) in excluded:
            options=retriever.alias_rows["SNOMED"];available=options.loc[options.code.eq(code),"text"]
            text=available.iloc[0] if len(available) else code
        names.append(text)
    encoder=Encoder(model_path,40,cfg["retrieval"]["encode_batch"]);vectors=encoder.encode(names)
    aliases=retriever.alias_rows["ICD"];index=DenseIndex(aliases,retriever.embeddings["ICD"],cfg["retrieval"]["dense_chunk"])
    for cdc in [True,False]:
        allowed=None if cdc else ~aliases.source.isin(["icd_index","icd_inclusion"]).to_numpy()
        result=np.empty((len(codes),len(index.codes)),dtype=np.float16) if smoke else np.lib.format.open_memmap(target/f"bridge_cdc{int(cdc)}.npy",mode="w+",dtype=np.float16,shape=(len(codes),len(index.codes)))
        for start in range(0,len(codes),64):result[start:start+64]=index.score_all(vectors[start:start+64],allowed=allowed).astype(np.float16)
        if hasattr(result,"flush"):result.flush()
        del result
    if not smoke:
        np.save(target/"bridge_snomed_codes.npy",np.asarray(codes,dtype=str));np.save(target/"bridge_vectors.npy",vectors)
        write_json(target/"bridge_complete.json",dict(n_preferred_queries=len(codes),query_source="SNOMED preferred terms",seed=13))
    del encoder,index,retriever;gc.collect();torch.cuda.empty_cache()
