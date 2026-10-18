"""Stages 1–2, shared by C1 and C2; no reranker or decision layer yet."""
from __future__ import annotations
from collections import defaultdict
import json
import time
import joblib
import numpy as np
import pandas as pd

from termmap.common.io import term_dir,path_for
from termmap.preprocessing.normalize import Normalizer,context_features,clean
from termmap.retrieval.lexical import CharIndex,BM25Index,rrf,expand_pool
from termmap.retrieval.dense import DenseIndex


def link_adjacency(table):
    result=defaultdict(set)
    for r in table.itertuples(index=False):result[r.src].add(r.dst);result[r.dst].add(r.src)
    return {c:sorted(v) for c,v in result.items()}


def combine_views(rows,k):
    values={}
    for row in rows:
        for code,score in row:values[code]=max(values.get(code,-1),score)
    return sorted(values.items(),key=lambda x:(-round(x[1],5),x[0]))[:k]


class Retriever:
    def __init__(self,cfg,encoder,embedding_dir,cdc=True,bridge=True,smoke=False):
        self.cfg=cfg;self.encoder=encoder;self.cdc=cdc;self.bridge=bridge
        out=term_dir(cfg)
        self.aliases=pd.read_parquet(out/"aliases.parquet").reset_index(drop=True)
        self.memory=pd.read_parquet(out/"train_memory.parquet")
        self.icd=pd.read_parquet(out/"icd_concepts.parquet").set_index("code").to_dict("index")
        self.snomed=pd.read_parquet(out/"snomed_concepts.parquet").set_index("code").to_dict("index")
        self.links=link_adjacency(pd.read_parquet(out/"snomed_links.parquet"))
        self.pairs=pd.read_parquet(out/"pair_evidence_train.parquet")
        self.probe=pd.read_parquet(out/"probe_set.parquet")
        support=out/"abbreviation_alias_support_train.parquet"
        self.normalizer=Normalizer(self.aliases,train=pd.read_parquet(out/"train.parquet"),abbreviations=pd.read_parquet(out/"abbreviations.parquet"),
            alias_support=pd.read_parquet(support) if support.exists() else None,excluded_names=set(self.probe.heldout_norm))
        self.normalizer.tightened=True
        self.lex={};self.word={};self.dense={};self.memlex={};self.memdense={};self.exact={}
        for system in ["ICD","SNOMED"]:
            a=self.aliases.loc[self.aliases.system.eq(system)].reset_index(drop=True)
            if smoke:a=a.head(512)
            use=a.loc[~a.source.isin(["icd_index","icd_inclusion"])] if not cdc else a
            self.exact[system]=defaultdict(set)
            for r in use.itertuples():self.exact[system][r.norm_text].add(r.code)
            cache=path_for(cfg,"index_dir")/("smoke" if smoke else "lexical")/f"{system}_cdc{int(cdc)}.joblib"
            if cache.exists():self.lex[system],self.word[system]=joblib.load(cache)
            else:
                self.lex[system],self.word[system]=CharIndex(use),BM25Index(use)
                cache.parent.mkdir(parents=True,exist_ok=True);joblib.dump((self.lex[system],self.word[system]),cache)
            self.dense[system]=DenseIndex(a,np.load(embedding_dir/f"{system}_aliases.npy",mmap_mode="r"),cfg["retrieval"]["dense_chunk"])
            self.dense[system].allowed=(~a.source.isin(["icd_index","icd_inclusion"])).to_numpy() if not cdc else None
            mem=self.memory.loc[self.memory.system.eq(system)].reset_index(drop=True)
            if smoke:mem=mem.head(32)
            self.memlex[system]=CharIndex(mem)
            self.memdense[system]=DenseIndex(mem,np.load(embedding_dir/f"{system}_memory.npy",mmap_mode="r"),cfg["retrieval"]["dense_chunk"])
        self.bridge_aliases=defaultdict(list)
        for r in self.aliases.loc[self.aliases.system.eq("SNOMED")].itertuples():self.bridge_aliases[r.code].append(r.text)

    def bridge_codes(self,snomed_pool,exclude_group=None):
        pairs=self.pairs
        if exclude_group is not None:pairs=pairs.loc[pairs.group_id.ne(exclude_group)]
        counts=pairs.groupby(["snomed","icd"]).size()
        result=[]
        for sn,_ in snomed_pool[:5]:
            if sn not in counts.index.get_level_values(0):continue
            values=counts.loc[sn];total=values.sum()
            if total<2:continue
            result.extend((str(c),float(n/total)) for c,n in values.items() if n/total>=.6)
        return sorted(dict(result).items(),key=lambda x:(-x[1],x[0]))

    def retrieve_one(self,mention,chunk="",exclude_group=None,views_enabled=None):
        start=time.perf_counter();cfg=self.cfg;limit=cfg["retrieval"]["channel_k"]
        features=context_features(mention,chunk,self.icd,cfg["preprocessing"]["window_chars"])
        views=self.normalizer.views(mention,"" if features["no_context"] else features["window"],exclude_group)
        pools={};diagnostics={}
        # SNOMED is retrieved first so ICD can use its top-five bridge.
        for system in ["SNOMED","ICD"]:
            enabled=views_enabled or cfg["retrieval"].get("enabled_by_system",{}).get(system,cfg["retrieval"]["enabled_views"])
            texts=list(dict.fromkeys(v for key in enabled for v in views[key]))
            dense_texts=list(dict.fromkeys(views[key][0] for key in ["V1","V2","V3","V4"] if key=="V1" or key in enabled))
            embeddings=self.encoder.encode(dense_texts)
            channels={
                "char":combine_views(self.lex[system].search(texts,limit),limit),
                "bm25":combine_views(self.word[system].search(texts,limit),limit),
                "dense":combine_views(self.dense[system].search(embeddings,limit,allowed=self.dense[system].allowed),limit),
                "memory":combine_views(self.memdense[system].search(embeddings,limit,exclude_group=exclude_group),limit),
            }
            exact=sorted({c for t in texts for c in self.exact[system].get(clean(t),())})
            channels["exact"]=[(c,1.0) for c in exact[:limit]]
            if system=="ICD":
                channels["literal"]=[(c,1.0) for c in features["literal_codes"]]
                if self.bridge:
                    bridge=self.bridge_codes(pools["SNOMED"],exclude_group)
                    if bridge:
                        extra=self.encoder.encode([self.icd[c]["preferred"] for c,_ in bridge if c in self.icd])
                        queries=self.dense[system].search(extra,limit,allowed=self.dense[system].allowed) if len(extra) else []
                        channels["bridge"]=combine_views([bridge,*queries],limit)
            fused=rrf(channels,cfg["retrieval"]["rrf_k"],cfg["retrieval"]["fused_k"])
            pool=expand_pool(fused,system,self.icd,self.links,cfg["retrieval"]["pool_k"],exact)
            pools[system]=pool
            diagnostics[system]=dict(channels=channels,fused=fused,views=views,features=features)
        return pools,diagnostics,time.perf_counter()-start
