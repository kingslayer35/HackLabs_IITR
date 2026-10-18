"""Private reusable query channels for fixed splits and model checkpoints."""
from __future__ import annotations
import gc
import hashlib
import json
import pickle
from pathlib import Path
import time
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd

from termmap.common.io import load_config,term_dir,path_for,read_partition,write_json,sha256
from termmap.preprocessing.normalize import Normalizer,clean,context_features,normalizer_signature
from termmap.preprocessing.noise import corrupt
from termmap.retrieval.lexical import CharIndex,BM25Index,rank_codes,rrf


def directory(cfg,profile="eval"):
    # Query text, context and the dense/lexical score rows must be rebuilt
    # together after preprocessing changes. A separate configured namespace
    # preserves historical caches and prevents silently mixing generations.
    key="query_cache_dir" if "query_cache_dir" in cfg["paths"] else "index_dir"
    p=path_for(cfg,key)/profile;p.mkdir(parents=True,exist_ok=True);return p


class ExtendedRows:
    """Read-only base memmap plus a small append-only query correction cache."""
    def __init__(self,base,extra):self.base=base;self.extra=extra
    def __len__(self):return len(self.base)+len(self.extra)
    def __getitem__(self,index):
        if np.isscalar(index):return self.base[index] if index<len(self.base) else self.extra[index-len(self.base)]
        return np.stack([self[i] for i in index])


def score_rows(path):
    base=np.load(path,mmap_mode="r");extra=path.with_name(path.stem+"_extra.npy")
    return ExtendedRows(base,np.load(extra,mmap_mode="r")) if extra.exists() else base


def tables(cfg,profile="eval"):
    out=term_dir(cfg)
    train=read_partition(cfg,"train") if profile=="eval" else all_rows(cfg)
    return out,train


def all_rows(cfg):
    status=path_for(cfg,"run_dir")/"test"/"STATUS"
    if not status.exists() or status.read_text().strip()!="DONE":raise RuntimeError("All-row rebuild waits for completed sealed-test evaluation")
    frames=[read_partition(cfg,"train"),read_partition(cfg,"dev")]
    frames.append(pd.read_parquet(term_dir(cfg)/"sealed_test.parquet"))
    return pd.concat(frames,ignore_index=True)


def prepare_queries(cfg,profile="eval",include_test=False):
    out,train=tables(cfg,profile);dest=directory(cfg,profile)
    aliases=pd.read_parquet(out/("aliases.parquet" if profile=="eval" else "aliases_final.parquet"))
    abbr=pd.read_parquet(out/("abbreviations_final.parquet" if profile=="final" else "abbreviations.parquet"))
    support_file=out/f"abbreviation_alias_support_{'final' if profile=='final' else 'train'}.parquet"
    support=pd.read_parquet(support_file) if support_file.exists() else None
    held=set(pd.read_parquet(out/"probe_set.parquet").heldout_norm) if profile!="final" else set()
    normalizer=Normalizer(aliases,train,abbr,support,excluded_names=held);normalizer.tightened=True
    if profile=="final":
        temporary=dest/"normalizer.pkl.partial"
        with temporary.open("wb") as stream:pickle.dump(normalizer,stream,protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(dest/"normalizer.pkl")
        write_json(dest/"normalizer_signature.json",normalizer_signature(cfg,out))
    legal=pd.read_parquet(out/"icd_concepts.parquet").code.tolist()
    datasets={"train":train,"dev":read_partition(cfg,"dev")} if profile=="eval" else {"all":train}
    if include_test:datasets={"test":read_partition(cfg,"test")}
    if profile=="eval" and not include_test:
        probe=pd.read_parquet(out/"probe_set.parquet").copy()
        probe["entity"]=probe.text;probe["chunk_text"]="";probe["subconcept"]="Unknown"
        probe["group_id"]=probe.system+":"+probe.code
        probe["row_id"]=[f"probe:{i}" for i in range(len(probe))]
        probe["valid_gold"]=True;datasets["probe"]=probe
    if not include_test:
        pref={s:pd.read_parquet(out/f"{'icd' if s=='ICD' else 'snomed'}_concepts.parquet").set_index("code").preferred.to_dict() for s in ["ICD","SNOMED"]}
        groups=defaultdict(list)
        for r in aliases.itertuples():
            if r.norm_text!=clean(pref[r.system][r.code]):groups[(r.system,r.code)].append(r.norm_text)
        rng=np.random.default_rng(cfg["seed"]);keys=sorted(groups)
        n=int(np.ceil((len(train)+train.system.eq("ICD").sum())*.25))
        rows=[]
        for i in range(n):
            system,code=keys[rng.integers(len(keys))];texts=sorted(set(groups[(system,code)]));text=texts[rng.integers(len(texts))]
            query=corrupt(text,seed=13,key=f"alias:{i}",abbreviations=normalizer.abbr)[0] if i%2 else text
            rows.append(dict(row_id=f"alias:{i}",system=system,code=code,entity=query,chunk_text="",subconcept="Unknown",group_id=f"alias:{i}",valid_gold=True,excluded_alias=text))
        datasets["alias"]=pd.DataFrame(rows)
    query_texts=set();records={}
    enabled=cfg["retrieval"].get("enabled_by_system",{})
    for name,frame in datasets.items():
        values=[]
        for row in frame.itertuples():
            exclude=row.group_id if name in {"train","all"} else None
            features=context_features(row.entity,row.chunk_text,legal,cfg["preprocessing"]["window_chars"])
            views=normalizer.views(row.entity,"" if features["no_context"] else features["window"],exclude)
            views_allowed=enabled.get(row.system,cfg["retrieval"]["enabled_views"])
            lexical=list(dict.fromkeys(t for v in views_allowed for t in views[v]))
            dense=list(dict.fromkeys(views[v][0] for v in ["V1","V2","V3","V4"] if v in views_allowed or v=="V1"))
            query_texts.update(lexical);query_texts.update(dense)
            record=row._asdict();record.pop("Index",None)
            record.update(views=views,lexical_queries=lexical,dense_queries=dense,context=features,
                          excluded_group=exclude,excluded_alias=getattr(row,"excluded_alias",None),
                          abbreviation_decisions=normalizer.abbreviation_diagnostics(row.entity,"" if features["no_context"] else features["window"],exclude))
            if row.system=="ICD":
                snviews=enabled.get("SNOMED",cfg["retrieval"]["enabled_views"])
                record["bridge_lexical_queries"]=list(dict.fromkeys(t for v in snviews for t in views[v]))
                record["bridge_dense_queries"]=list(dict.fromkeys(views[v][0] for v in ["V1","V2","V3","V4"] if v=="V1" or v in snviews))
                query_texts.update(record["bridge_lexical_queries"]+record["bridge_dense_queries"])
            values.append(record)
        records[name]=values
    priors=pd.read_parquet(out/("bridge_train.parquet" if profile=="eval" else "bridge_final.parquet"))
    icd=pd.read_parquet(out/"icd_concepts.parquet").set_index("code")
    query_texts.update(icd.loc[icd.index.isin(priors.icd),"preferred"].tolist())
    queries=sorted(query_texts)
    if include_test:
        # Separate test-only cache; never mutate the development query layout.
        dest=directory(cfg,"eval_test")
    write_json(dest/"queries.json",queries)
    joblib.dump(records,dest/"records.joblib")
    print("Prepared query counts",{k:len(v) for k,v in records.items()},len(queries),flush=True)
    return dest


def build_lexical(cfg,profile="eval",smoke=False):
    dest=directory(cfg,profile);out=term_dir(cfg)
    alias_file=out/("aliases_final.parquet" if profile=="final" else "aliases.parquet")
    aliases=pd.read_parquet(alias_file)
    queries=json.loads((dest/"queries.json").read_text())
    if smoke:queries=queries[:8]
    provenance=dict(aliases_sha256=sha256(alias_file),queries_sha256=hashlib.sha256(json.dumps(queries).encode()).hexdigest(),
                    retrieval_source_sha256=sha256(Path(__file__).with_name("lexical.py")),profile=profile,smoke=smoke)
    for system in ["ICD","SNOMED"]:
        for cdc in ([True,False] if system=="ICD" else [True]):
            name=f"{system}_cdc{int(cdc)}";cache=dest/(name+"_lex.joblib")
            marker=dest/(name+"_lexical_complete.json")
            products=[cache,dest/(name+"_codes.npy"),dest/(name+"_char_scores.npy"),dest/(name+"_bm25.joblib")]
            if marker.exists() and json.loads(marker.read_text())==provenance and all(p.exists() for p in products):continue
            marker.unlink(missing_ok=True)
            # A full-size mmap header does not prove all query rows were
            # written. Rebuild an incomplete channel rather than reuse zeros.
            for product in products:product.unlink(missing_ok=True)
            selected=aliases.loc[aliases.system.eq(system)]
            if not cdc:selected=selected.loc[~selected.source.isin(["icd_index","icd_inclusion"])]
            if cache.exists():char,word=joblib.load(cache)
            else:
                char,word=CharIndex(selected),BM25Index(selected);joblib.dump((char,word),cache)
            np.save(dest/(name+"_codes.npy"),char.codes.astype(str))
            filename=dest/(name+"_char_scores.npy")
            if not filename.exists():
                scores=np.lib.format.open_memmap(filename,mode="w+",dtype=np.float16,shape=(len(queries),len(char.codes)))
                ranked=[]
                for start in range(0,len(queries),32):
                    batch=queries[start:start+32]
                    scores[start:start+len(batch)]=char.score_all(batch).astype(np.float16)
                    ranked.extend(word.search(batch,100))
                    if start%1024==0:print(name,"CPU",start,"/",len(queries),flush=True)
                scores.flush();del scores;joblib.dump(ranked,dest/(name+"_bm25.joblib"))
            write_json(marker,provenance)
            del char,word;gc.collect()


def encode_dense(cfg,model_path,run,profile="eval",smoke=False):
    from termmap.retrieval.dense import Encoder,DenseIndex
    from termmap.common.io import seed_everything
    seed_everything(13,torch_enabled=True)
    dest=directory(cfg,profile);out=term_dir(cfg);target=dest/run;target.mkdir(parents=True,exist_ok=True)
    if not (target/"encoding_metadata.json").exists():
        # A killed mmap writer leaves a full-size header with partially filled rows.
        # The saved model epoch is authoritative; rebuild that incomplete index.
        for partial in [*target.glob("*.npy"),*target.glob("*.partial")]:partial.unlink()
    aliases=pd.read_parquet(out/("aliases_final.parquet" if profile=="final" else "aliases.parquet"))
    _,train=tables(cfg,"final" if profile=="final" else "eval")
    memory=train.loc[train.valid_gold,["row_id","system","code","entity","group_id"]].rename(columns={"entity":"text"})
    memory["norm_text"]=memory.text.map(clean)
    excluded=set(pd.read_parquet(out/"probe_set.parquet").heldout_norm) if profile!="final" else set()
    memory=memory.loc[~memory.norm_text.isin(excluded)]
    memory.to_parquet(dest/"memory.parquet",index=False)
    queries=json.loads((dest/"queries.json").read_text())
    if smoke:queries=queries[:8]
    encoder=Encoder(model_path,cfg["training"]["max_length"],cfg["retrieval"]["encode_batch"])
    qpath=target/"query_embeddings.npy"
    qvectors=np.load(qpath,mmap_mode="r") if qpath.exists() else encoder.encode(queries,qpath)
    for system in ["ICD","SNOMED"]:
        a=aliases.loc[aliases.system.eq(system)].reset_index(drop=True)
        if smoke:a=a.head(256)
        a.to_parquet(target/f"{system}_alias_rows.parquet",index=False)
        f=target/f"{system}_aliases.npy"
        vectors=np.load(f,mmap_mode="r") if f.exists() else encoder.encode(a.text.tolist(),f)
        index=DenseIndex(a,vectors,cfg["retrieval"]["dense_chunk"])
        np.save(target/f"{system}_codes.npy",index.codes.astype(str))
        for cdc in ([True,False] if system=="ICD" else [True]):
            f=target/f"{system}_cdc{int(cdc)}_dense.npy"
            if f.exists():continue
            allowed=None if cdc else ~a.source.isin(["icd_index","icd_inclusion"]).to_numpy()
            scores=np.lib.format.open_memmap(f,mode="w+",dtype=np.float16,shape=(len(queries),len(index.codes)))
            for start in range(0,len(queries),64):
                scores[start:start+64]=index.score_all(qvectors[start:start+64],allowed=allowed).astype(np.float16)
                if start%1024==0:print(system,"dense",start,"/",len(queries),flush=True)
            scores.flush()
        del index;gc.collect()
        mem=memory.loc[memory.system.eq(system)].reset_index(drop=True)
        if smoke:mem=mem.head(16)
        mem.to_parquet(target/f"{system}_memory_rows.parquet",index=False)
        f=target/f"{system}_memory.npy"
        mv=np.load(f,mmap_mode="r") if f.exists() else encoder.encode(mem.text.tolist(),f)
        import torch
        f=target/f"{system}_memory_scores.npy"
        if not f.exists():
            scores=np.lib.format.open_memmap(f,mode="w+",dtype=np.float16,shape=(len(queries),len(mem)))
            m=torch.tensor(np.asarray(mv),device="cuda",dtype=torch.float16)
            for start in range(0,len(queries),64):
                q=torch.tensor(np.asarray(qvectors[start:start+64]),device="cuda",dtype=torch.float16)
                scores[start:start+64]=(q@m.T).cpu().numpy()
            scores.flush();del m
        if system=="SNOMED" and run=="C1" and not smoke:
            from termmap.terminology.hierarchy import strip_tag
            concepts=pd.read_parquet(out/"snomed_concepts.parquet")
            representative={r.code:r.text for r in a.iloc[::-1].itertuples()}
            texts=[r.preferred if clean(strip_tag(r.preferred)) not in excluded else representative.get(r.code,r.code) for r in concepts.itertuples()]
            encoder.encode(texts,target/"snomed_preferred.npy")
            np.save(target/"snomed_preferred_codes.npy",concepts.code.to_numpy(dtype=str))
    del encoder;gc.collect()
    import torch
    torch.cuda.empty_cache()
    write_json(target/"encoding_metadata.json",dict(model=str(model_path),queries=len(queries),seed=13,profile=profile))
    if cfg["retrieval"].get("bridge_enabled",True):
        from termmap.terminology.bridge import build_bridge_index
        build_bridge_index(cfg,model_path,run,profile,datasets={"dev","probe"} if run.startswith("C2_epoch_") else None,smoke=smoke)
    return target
