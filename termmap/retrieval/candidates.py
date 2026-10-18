"""Full retrieval from exact cached scores; no fitted or gold-derived queries."""
from collections import defaultdict
import json
import time
import joblib
import numpy as np
import pandas as pd
from termmap.common.io import term_dir
from termmap.retrieval.cache import directory,score_rows
from termmap.preprocessing.normalize import clean
from termmap.retrieval.lexical import rank_codes,rrf


class CachedRetriever:
    def __init__(self,cfg,run="C1",profile="eval"):
        self.cfg=cfg;self.root=directory(cfg,profile);self.dense_dir=self.root/run;out=term_dir(cfg)
        self.queries=json.loads((self.root/"queries.json").read_text());self.qid={q:i for i,q in enumerate(self.queries)}
        self.qvectors=score_rows(self.dense_dir/"query_embeddings.npy")
        # P2-v2 runs encode each system's queries with its own checkpoint (frozen alias index).
        self.system_qvectors={s:score_rows(self.dense_dir/f"{s}_query_embeddings.npy") if (self.dense_dir/f"{s}_query_embeddings.npy").exists()
                              else self.qvectors for s in ["ICD","SNOMED"]}
        self.icd_frame=pd.read_parquet(out/("icd_concepts_final.parquet" if profile=="final" else "icd_concepts.parquet")).sort_values("pre_in")
        self.icd=self.icd_frame.set_index("code").to_dict("index")
        self.sn_frame=pd.read_parquet(out/"snomed_concepts.parquet")
        self.sn=self.sn_frame.set_index("code").to_dict("index")
        self.family={c:g.code.tolist() for c,g in self.sn_frame.groupby("family_id",sort=False)}
        self.links=defaultdict(lambda:defaultdict(set));self.near=defaultdict(set);self.trusted=defaultdict(set)
        for r in pd.read_parquet(out/"snomed_links.parquet").itertuples():
            self.links[r.kind][r.src].add(r.dst);self.links[r.kind][r.dst].add(r.src)
            if r.kind in {"specifier_child","qualifier_sibling"}:self.near[r.src].add(r.dst);self.near[r.dst].add(r.src)
            if r.kind=="specifier_child":self.trusted[r.src].add(r.dst)
        self.near["__specifier_children__"]=self.trusted
        self.aliases=pd.read_parquet(out/("aliases_final.parquet" if profile=="final" else "aliases.parquet"))
        self.self_only_aliases=defaultdict(set)
        support=out/f"abbreviation_alias_support_{'final' if profile=='final' else 'train'}.parquet"
        if support.exists():
            for r in pd.read_parquet(support).itertuples():
                if len(r.support_groups)==1:self.self_only_aliases[(r.system,r.support_groups[0])].add((r.code,r.norm_text))
        self.exact={};self.char={};self.dense={};self.word={};self.lex_objects={};self.codes={};self.code_index={}
        self.dcodes={};self.dcode_index={};self.embeddings={};self.alias_rows={};self.memory={};self.memory_scores={}
        self.alias_indices={}
        for system in ["ICD","SNOMED"]:
            for cdc in ([True,False] if system=="ICD" else [True]):
                name=f"{system}_cdc{int(cdc)}";key=(system,cdc)
                self.char[key]=score_rows(self.root/(name+"_char_scores.npy"))
                self.word[key]=joblib.load(self.root/(name+"_bm25.joblib"))
                self.codes[key]=np.load(self.root/(name+"_codes.npy"))
                self.code_index[key]={c:i for i,c in enumerate(self.codes[key])}
                self.dense[key]=score_rows(self.dense_dir/(name+"_dense.npy"))
                a=self.aliases.loc[self.aliases.system.eq(system)]
                if not cdc:a=a.loc[~a.source.isin(["icd_index","icd_inclusion"])]
                exact=defaultdict(set)
                for r in a.itertuples():exact[r.norm_text].add(r.code)
                self.exact[key]=exact
            self.dcodes[system]=np.load(self.dense_dir/f"{system}_codes.npy")
            self.dcode_index[system]={c:i for i,c in enumerate(self.dcodes[system])}
            # Global probe-name withholding can remove every name of an unrelated
            # concept. Keep those concepts reachable by exact code and expansion.
            legal=self.icd if system=="ICD" else self.sn
            for code in legal.keys()-self.dcode_index[system].keys():
                for key in self.exact:
                    if key[0]==system:self.exact[key][clean(code)].add(code)
            self.alias_rows[system]=pd.read_parquet(self.dense_dir/f"{system}_alias_rows.parquet")
            self.alias_indices[system]=self.alias_rows[system].groupby("code",sort=False).indices
            self.embeddings[system]=np.load(self.dense_dir/f"{system}_aliases.npy",mmap_mode="r")
            self.memory[system]=pd.read_parquet(self.dense_dir/f"{system}_memory_rows.parquet")
            self.memory_scores[system]=score_rows(self.dense_dir/f"{system}_memory_scores.npy")
        pairfile="pair_evidence_train.parquet" if profile!="final" else "pair_evidence_final.parquet"
        self.pairs=pd.read_parquet(out/pairfile)
        self.bridge_cache={}
        self.bridge_ids={};self.bridge_scores={};self.bridge_vectors=None
        if (self.dense_dir/"bridge_snomed_codes.npy").exists():
            self.bridge_ids={c:i for i,c in enumerate(np.load(self.dense_dir/"bridge_snomed_codes.npy"))}
            self.bridge_vectors=np.load(self.dense_dir/"bridge_vectors.npy",mmap_mode="r")
            for cdc in [True,False]:
                p=self.dense_dir/f"bridge_cdc{int(cdc)}.npy"
                if p.exists():self.bridge_scores[cdc]=np.load(p,mmap_mode="r")

    def ensure_bridge(self,codes):
        missing=set(codes)-self.bridge_ids.keys()
        if missing:raise RuntimeError(f"Bridge preferred-term cache misses {len(missing)} SNOMED concepts")

    def snomed_basic(self,row):
        if row["system"]=="SNOMED":return self.basic(row,"SNOMED",True)
        enabled=self.cfg["retrieval"].get("enabled_by_system",{}).get("SNOMED",self.cfg["retrieval"]["enabled_views"])
        lexical=row.get("bridge_lexical_queries",list(dict.fromkeys(t for v in enabled for t in row["views"][v])))
        dense=row.get("bridge_dense_queries",list(dict.fromkeys(row["views"][v][0] for v in ["V1","V2","V3","V4"] if v=="V1" or v in enabled)))
        other=dict(row,lexical_queries=lexical,dense_queries=dense)
        return self.basic(other,"SNOMED",True)

    def lex_objects_for(self,key):
        if key not in self.lex_objects:
            system,cdc=key;self.lex_objects[key]=joblib.load(self.root/f"{system}_cdc{int(cdc)}_lex.joblib")
        return self.lex_objects[key]

    def basic(self,row,system,cdc=True,forbidden=None):
        key=(system,cdc if (system,cdc) in self.char else True)
        lexical=row["lexical_queries"];dense=row["dense_queries"];excluded=row.get("excluded_alias")
        blocked=self.self_only_aliases.get((system,row.get("excluded_group")),set())
        qids=[self.qid[t] for t in lexical];dids=[self.qid[t] for t in dense]
        char=np.max(self.char[key][qids],axis=0).astype(np.float32)
        dscore=np.max(self.dense[key][dids],axis=0).astype(np.float32)
        bm={}
        for i in qids:
            for c,s in self.word[key][i]:bm[c]=max(bm.get(c,0),s)
        if excluded or blocked:
            ci,wi=self.lex_objects_for(key)
            if excluded:char=ci.scores_without_alias(lexical,excluded,char)
            if blocked:char=ci.scores_without_pairs(lexical,blocked,char,excluded)
            bm={}
            for result in wi.search(lexical,100,exclude_norm=excluded,excluded_pairs=blocked):
                for c,s in result:bm[c]=max(bm.get(c,0),s)
            a=self.alias_rows[system];affected=set(self.exact[key].get(excluded,()))|{c for c,_ in blocked}
            for c in affected:
                if c not in self.alias_indices[system]:continue
                ids=self.alias_indices[system][c];mask=a.iloc[ids].norm_text.ne(excluded).to_numpy()
                if blocked:mask &= ~a.iloc[ids].norm_text.isin([n for code,n in blocked if code==c]).to_numpy()
                if not key[1]:mask &= ~a.iloc[ids].source.isin(["icd_index","icd_inclusion"]).to_numpy()
                ids=ids[mask]
                score=float((np.asarray(self.system_qvectors[system][dids],dtype=np.float32)@np.asarray(self.embeddings[system][ids],dtype=np.float32).T).max()) if len(ids) else -1.
                dscore[self.dcode_index[system][c]]=round(score,5)
        memory=self.memory[system];ms=np.max(self.memory_scores[system][dids],axis=0).astype(np.float32)
        allowed=np.ones(len(memory),dtype=bool)
        if row.get("excluded_group") is not None:allowed &= memory.group_id.ne(row["excluded_group"]).to_numpy()
        if excluded:allowed &= memory.norm_text.ne(excluded).to_numpy()
        mem={}
        for c,s,keep in zip(memory.code,ms,allowed):
            if keep:mem[c]=max(mem.get(c,-1),float(s))
        if forbidden:
            if forbidden in self.code_index[key]:char[self.code_index[key][forbidden]]=-1
            if forbidden in self.dcode_index[system]:dscore[self.dcode_index[system][forbidden]]=-1
            bm.pop(forbidden,None);mem.pop(forbidden,None)
        channels={"char":rank_codes(char,self.codes[key],100),"dense":rank_codes(dscore,self.dcodes[system],100),
                  "bm25":sorted(bm.items(),key=lambda x:(-round(x[1],5),x[0]))[:100],
                  "memory":sorted(mem.items(),key=lambda x:(-round(x[1],5),x[0]))[:100]}
        exact=sorted({c for t in lexical if clean(t)!=excluded for c in self.exact[key].get(clean(t),()) if c!=forbidden and (c,clean(t)) not in blocked})
        channels["exact"]=[(c,1.) for c in exact[:100]]
        def score(code):
            i=self.code_index[key].get(code);j=self.dcode_index[system].get(code)
            return (float(char[i]) if i is not None else 0)+(float(dscore[j]) if j is not None else 0)
        return channels,exact,char,dscore,mem,score

    def retrieve(self,row,flags=None,forbidden=None):
        flags=flags or self.cfg["retrieval"]
        system=row["system"];cdc=flags.get("cdc_enabled",True)
        channels,exact,char,dense,mem,score=self.basic(row,system,cdc,forbidden)
        if system=="ICD":
            channels["literal"]=[(c,1.) for c in row["context"]["literal_codes"] if c!=forbidden]
            if flags.get("bridge_enabled",True):
                snchan,*_=self.snomed_basic(row)
                top=rrf(snchan,60,5)
                group=row.get("excluded_group")
                if group not in self.bridge_cache:
                    p=self.pairs if group is None else self.pairs.loc[self.pairs.group_id.ne(group)]
                    counts=p.groupby(["snomed","icd"]).size();bridge=defaultdict(list)
                    for sn,g in counts.groupby(level=0):
                        total=g.sum()
                        if total>=2:bridge[sn]=[(c,n/total) for (_,c),n in g.items() if n/total>=.6]
                    self.bridge_cache[group]=bridge
                bridges={c:s for sn,_ in top for c,s in self.bridge_cache[group].get(sn,[]) if c!=forbidden}
                values=dict(bridges)
                self.ensure_bridge([sn for sn,_ in top])
                for sn,_ in top:
                    ds=np.asarray(self.bridge_scores[cdc][self.bridge_ids[sn]],dtype=np.float32).copy()
                    excluded=row.get("excluded_alias")
                    if excluded:
                        a=self.alias_rows["ICD"]
                        for c in self.exact[("ICD",cdc)].get(excluded,()):
                            ids=self.alias_indices["ICD"][c];mask=a.iloc[ids].norm_text.ne(excluded).to_numpy()
                            if not cdc:mask &= ~a.iloc[ids].source.isin(["icd_index","icd_inclusion"]).to_numpy()
                            ids=ids[mask];q=np.asarray(self.bridge_vectors[self.bridge_ids[sn]],dtype=np.float32)
                            ds[self.dcode_index["ICD"][c]]=float((q@np.asarray(self.embeddings["ICD"][ids],dtype=np.float32).T).max()) if len(ids) else -1.
                    if forbidden in self.dcode_index["ICD"]:ds[self.dcode_index["ICD"][forbidden]]=-1.
                    for c,s in rank_codes(ds,self.dcodes["ICD"],100):
                        if c!=forbidden:values[c]=max(values.get(c,-1),s)
                channels["bridge"]=sorted(values.items(),key=lambda x:(-round(x[1],5),x[0]))[:100]
        fused=rrf(channels,60,50);initial=[c for c,_ in fused];exp=set()
        if system=="ICD":
            for c in initial[:10]:
                r=self.icd[c];p=r["parent"]
                if p:exp.add(p);exp.update(list(self.icd[p]["children"])[:10])
                exp.update(list(r["children"])[:20])
            if flags.get("subtree_enabled",True):
                votes=defaultdict(float)
                for c,s in fused:votes[c[:3]]+=s
                for category in sorted(votes,key=lambda c:(-votes[c],c))[:3]:
                    n=self.icd[category];codes=self.icd_frame.iloc[n["pre_in"]:n["pre_out"]+1].code
                    exp.update(sorted(codes,key=lambda c:(-round(score(c),5),c))[:10])
        else:
            for c in initial[:5]:
                exp.update(self.links["qualifier_sibling"].get(c,()))
                exp.update(self.links["specifier_child"].get(c,()))
                if flags.get("lookalike_enabled",True):exp.update(self.links["lookalike"].get(c,()))
                if flags.get("family_enabled",True):
                    members=self.family.get(self.sn[c]["family_id"],())
                    exp.update(sorted(members,key=lambda code:(-round(score(code),5),code))[:5])
        exp.discard(forbidden);exp.difference_update(initial)
        must=[c for c in exact if c not in initial]
        additions=must+[c for c in sorted(exp,key=lambda c:(-round(score(c),5),c)) if c not in must]
        codes=(initial+additions)[:64]
        if any(c not in codes for c in must):
            mandatory=set(must);codes=[c for c in initial if c not in mandatory][:max(0,64-len(must))]+must[:64]
        rrf_scores=dict(fused);features={}
        key=(system,cdc if (system,cdc) in self.char else True)
        for code in codes:
            i=self.code_index[key].get(code);j=self.dcode_index[system].get(code)
            features[code]=dict(dense=float(dense[j]) if j is not None else 0,
                                lexical=float(char[i]) if i is not None else 0,memory=float(mem.get(code,0)),
                                rrf=rrf_scores.get(code,0),literal=float(code in row["context"]["literal_codes"]),
                                agreement=sum(code in [c for c,_ in values[:10]] for values in channels.values()))
        pool=[(c,rrf_scores.get(c,0.)) for c in codes]
        return dict(row=row,pool=pool,channels=channels,features=features,expansion=[c for c in codes if c not in initial])
