"""The same cached retriever, with exact score rows computed for new queries."""
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from termmap.retrieval.candidates import CachedRetriever
from termmap.retrieval.cache import directory,all_rows
from termmap.retrieval.dense import Encoder,DenseIndex
from termmap.common.io import path_for,term_dir
from termmap.preprocessing.normalize import Normalizer,context_features,clean,normalizer_signature
from termmap.decision.coupling import infer_types,type_cache_path
from termmap.retrieval.asymmetric import base_model,selected_paths


class ExtraRows:
    def __init__(self,base):self.base=base;self.extra=[]
    def __len__(self):return len(self.base)+len(self.extra)
    def append(self,rows):self.extra.extend(np.asarray(rows))
    def __getitem__(self,index):
        if np.isscalar(index):return self.base[index] if index<len(self.base) else self.extra[index-len(self.base)]
        return np.stack([self[i] for i in index])


class OnlineRetriever(CachedRetriever):
    def __init__(self,cfg,flow):
        profile=cfg.get("deployment",{}).get("profile","final")
        run=("final_adapted" if flow[1]=="1" else "final_base") if profile=="final" else ("C2_v2" if flow[1]=="1" else "C1")
        super().__init__(cfg,run,profile)
        original=base_model(cfg)
        self.encoder=Encoder(original,cfg["training"]["max_length"],cfg["retrieval"]["encode_batch"])
        models=selected_paths(cfg)[0] if flow[1]=="1" else {s:original for s in ["ICD","SNOMED"]}
        encoders={str(original.resolve()):self.encoder};self.query_encoders={}
        for system,model in models.items():
            key=str(model.resolve())
            if key not in encoders:encoders[key]=Encoder(model,cfg["training"]["max_length"],cfg["retrieval"]["encode_batch"])
            self.query_encoders[system]=encoders[key]
        self.qvectors=ExtraRows(self.qvectors)
        self.system_qvectors={s:ExtraRows(v) for s,v in self.system_qvectors.items()}
        for table in [self.char,self.dense,self.memory_scores]:
            for key,value in table.items():table[key]=ExtraRows(value)
        self.indexes={s:DenseIndex(self.alias_rows[s],self.embeddings[s],cfg["retrieval"]["dense_chunk"]) for s in ["ICD","SNOMED"]}
        self.memory_vectors={s:np.load(self.dense_dir/f"{s}_memory.npy",mmap_mode="r") for s in ["ICD","SNOMED"]}
        encoding=json.loads((self.dense_dir/"encoding_metadata.json").read_text())
        current_arch="sm"+"".join(map(str,torch.cuda.get_device_capability()))
        self.runtime_provenance=dict(index_architecture=encoding["gpu_architecture"],gpu_architecture=current_arch,
                                     cached_gpu_query_rows_reused=encoding["gpu_architecture"]==current_arch)
        if encoding["gpu_architecture"]!=current_arch:
            # Original alias vectors are the explicitly approved fixed index.
            # Query/memory score rows are architecture-specific. Preserve the
            # base row offsets but route every requested query to a new row,
            # encoding memory with the matching per-system model on this GPU.
            self.qid={}
            self.memory_vectors={s:self.query_encoders[s].encode(self.memory[s].text.tolist()) for s in ["ICD","SNOMED"]}
        if self.bridge_vectors is not None:self.bridge_vectors=ExtraRows(self.bridge_vectors)
        self.bridge_scores={k:ExtraRows(v) for k,v in self.bridge_scores.items()}
        out=term_dir(cfg)
        cache=directory(cfg,profile)/"normalizer.pkl";signature=cache.with_name("normalizer_signature.json")
        if profile=="final" and cache.exists() and signature.exists() and json.loads(signature.read_text())==normalizer_signature(cfg,out):
            with cache.open("rb") as stream:self.normalizer=pickle.load(stream)
        else:
            support=out/("abbreviation_alias_support_final.parquet" if profile=="final" else "abbreviation_alias_support_train.parquet")
            training=all_rows(cfg) if profile=="final" else pd.read_parquet(out/"train.parquet")
            self.normalizer=Normalizer(self.aliases,training,pd.read_parquet(out/("abbreviations_final.parquet" if profile=="final" else "abbreviations.parquet")),
                pd.read_parquet(support) if support.exists() else None,
                excluded_names=set() if profile=="final" else set(pd.read_parquet(out/"probe_set.parquet").heldout_norm))
        self.normalizer.tightened=True
        classifier=type_cache_path(cfg,"final_base" if profile=="final" else "C1",final=profile=="final")
        if classifier.exists():self.type_classifier=joblib.load(classifier)
        else:
            records=joblib.load(directory(cfg,profile)/"records.joblib")
            infer_types(cfg,self,records,"final_base" if profile=="final" else "C1")
            self.type_classifier=joblib.load(classifier)

    def ensure_bridge(self,codes):
        new=sorted(set(codes)-self.bridge_ids.keys())
        if not new:return
        vectors=self.encoder.encode([self.sn[c]["preferred"] for c in new]);start=len(self.bridge_ids)
        if self.bridge_vectors is None:self.bridge_vectors=ExtraRows(np.empty((0,vectors.shape[1]),dtype=np.float16))
        self.bridge_vectors.append(vectors)
        for cdc in [True,False]:
            a=self.alias_rows["ICD"];allowed=None if cdc else ~a.source.isin(["icd_index","icd_inclusion"]).to_numpy()
            if cdc not in self.bridge_scores:self.bridge_scores[cdc]=ExtraRows(np.empty((0,len(self.dcodes["ICD"])),dtype=np.float16))
            self.bridge_scores[cdc].append(self.indexes["ICD"].score_all(vectors,allowed=allowed))
        self.bridge_ids.update({c:start+i for i,c in enumerate(new)})

    def add_queries(self,texts):
        new=sorted(set(texts)-self.qid.keys())
        if not new:return
        vectors=self.encoder.encode(new)
        system_vectors={s:(vectors if encoder is self.encoder else encoder.encode(new)) for s,encoder in self.query_encoders.items()}
        for offset in range(0,len(new),64):
            batch=new[offset:offset+64]
            for key in self.char:
                char,word=self.lex_objects_for(key);self.char[key].append(char.score_all(batch))
                self.word[key].extend(word.search(batch,100))
                system,cdc=key;a=self.alias_rows[system]
                qv=system_vectors[system][offset:offset+64]
                allowed=None if cdc else ~a.source.isin(["icd_index","icd_inclusion"]).to_numpy()
                self.dense[key].append(self.indexes[system].score_all(qv,allowed=allowed))
            for system in ["ICD","SNOMED"]:
                qv=system_vectors[system][offset:offset+64]
                q=torch.tensor(qv,device="cuda",dtype=torch.float16);m=torch.tensor(np.asarray(self.memory_vectors[system]),device="cuda",dtype=torch.float16)
                self.memory_scores[system].append((q@m.T).cpu().numpy());del q,m
        start=len(self.queries);self.queries.extend(new);self.qid.update({q:start+i for i,q in enumerate(new)})
        self.qvectors.append(vectors)
        for system in ["ICD","SNOMED"]:self.system_qvectors[system].append(system_vectors[system])

    def prepare_inputs(self,rows,mention_column):
        """Sort the complete query set so shuffled rows have identical GPU batches."""
        prepared=[];queries=set()
        for i,row in enumerate(rows):
            entity=row[mention_column];chunk=row.get("chunk_text","")
            ctx=context_features(entity,chunk,self.icd,self.cfg["preprocessing"]["window_chars"])
            views=self.normalizer.views(entity,"" if ctx["no_context"] else ctx["window"])
            abbreviation_decisions=self.normalizer.abbreviation_diagnostics(entity,"" if ctx["no_context"] else ctx["window"])
            pair=[];queries.add(views["V1"][0])
            for system in ["ICD","SNOMED"]:
                enabled=self.cfg["retrieval"].get("enabled_by_system",{}).get(system,self.cfg["retrieval"]["enabled_views"])
                lex=list(dict.fromkeys(t for v in enabled for t in views[v]));dense=list(dict.fromkeys(views[v][0] for v in ["V1","V2","V3","V4"] if v=="V1" or v in enabled))
                queries.update(lex+dense)
                pair.append(dict(entity=entity,chunk_text=chunk,subconcept=row.get("subconcept","Unknown"),system=system,row_id=f"{i}:{system}",chunk_id=row.get("chunk_id",str(i)),group_id=f"predict:{i}",views=views,context=ctx,abbreviation_decisions=abbreviation_decisions,lexical_queries=lex,dense_queries=dense,excluded_group=None,excluded_alias=None))
            prepared.append(pair)
        self.add_queries(sorted(queries));bridge_codes=set()
        from termmap.retrieval.lexical import rrf
        for pair in prepared:
            for row in pair:
                if row["subconcept"] not in self.type_classifier.classes_:
                    vector=np.asarray(self.qvectors[[self.qid[row["views"]["V1"][0]]]],dtype=np.float32)
                    row["subconcept"]=str(self.type_classifier.predict(vector)[0])
            if self.cfg["retrieval"].get("bridge_enabled",True):
                channels,*_=self.snomed_basic(pair[0])
                bridge_codes.update(c for c,_ in rrf(channels,60,5))
        if bridge_codes:self.ensure_bridge(sorted(bridge_codes))
        return prepared

    def mention(self,entity,chunk="",subconcept="Unknown",row_id="0",chunk_id=""):
        context=context_features(entity,chunk,self.icd,self.cfg["preprocessing"]["window_chars"])
        views=self.normalizer.views(entity,"" if context["no_context"] else context["window"]);self.add_queries([views["V1"][0]])
        abbreviation_decisions=self.normalizer.abbreviation_diagnostics(entity,"" if context["no_context"] else context["window"])
        known=set(self.type_classifier.classes_)
        if subconcept not in known:subconcept=str(self.type_classifier.predict(np.asarray(self.qvectors[[self.qid[views["V1"][0]]]],dtype=np.float32))[0])
        records=[]
        for system in ["ICD","SNOMED"]:
            enabled=self.cfg["retrieval"].get("enabled_by_system",{}).get(system,self.cfg["retrieval"]["enabled_views"])
            lexical=list(dict.fromkeys(t for v in enabled for t in views[v]));dense=list(dict.fromkeys(views[v][0] for v in ["V1","V2","V3","V4"] if v=="V1" or v in enabled))
            self.add_queries(lexical+dense)
            row=dict(entity=entity,chunk_text=chunk,subconcept=subconcept,system=system,row_id=row_id+":"+system,chunk_id=chunk_id,
                     group_id="predict:"+row_id,views=views,context=context,abbreviation_decisions=abbreviation_decisions,lexical_queries=lexical,dense_queries=dense,excluded_group=None,excluded_alias=None)
            records.append(self.retrieve(row))
        return records
