"""P2-v2 SapBERT adaptation: train the query side against the frozen original index.

v1 retrained both sides and moved dictionary names far from where SapBERT had them
(mean cosine to their own original vector: 0.43 ICD, 0.57 SNOMED). Here the dictionary
vectors are the off-the-shelf C1 ones and never change; only chart text -> vector moves.
Loss: InfoNCE over every concept of the system, scored exactly like the dense channel
(max over a row's dense query views, max over a concept's names). A name the gold shares
with another code, or the query's own exact text, is never used as a negative.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import copy
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from termmap.common.io import term_dir,path_for,write_json,seed_everything
from termmap.retrieval.cache import directory
from termmap.preprocessing.normalize import clean
from termmap.terminology.hierarchy import strip_tag
from termmap.preprocessing.noise import corrupt

SYSTEMS=("ICD","SNOMED")
CDC_SOURCES=("icd_index","icd_inclusion")


def pinned(cfg):
    """P2-v2 is trained and judged on one fixed query generation (training_v2.index_root)."""
    cfg=copy.deepcopy(cfg);root=cfg["training_v2"].get("index_root")
    if root:cfg["paths"]["query_cache_dir"]=root
    # The retrieval policy that generation was scored with (the live policy file may move on).
    policy=path_for(cfg,"run_dir")/"pretrain_v2"/"retrieval_policy_C1.json"
    if policy.exists():cfg["retrieval"].update(json.loads(policy.read_text()))
    return cfg


def key(text):
    return clean(strip_tag(text))


def bucket(frequency):
    return "unseen" if frequency==0 else "rare" if frequency<=5 else "common"


class FrozenIndex:
    """One system's original alias vectors: the rows the dense channel searches."""
    def __init__(self,system,rows,vectors,keys=None):
        self.system=system;self.rows=rows.reset_index(drop=True);self.vectors=np.asarray(vectors)
        self.codes=np.array(sorted(self.rows.code.unique()),dtype=object)
        self.code_index={c:i for i,c in enumerate(self.codes)}
        self.alias_code=self.rows.code.map(self.code_index).to_numpy(np.int64)
        self.keys=self.rows.text.map(key).to_numpy(dtype=object) if keys is None else keys
        self.by_key=pd.Series(np.arange(len(self.rows))).groupby(self.keys).indices
        self.by_code=self.rows.groupby("code",sort=False).indices
        self.device={}

    @classmethod
    def load(cls,cfg,system,run="C1",profile="eval"):
        root=directory(cfg,profile)/run
        rows=pd.read_parquet(root/f"{system}_alias_rows.parquet")
        keep=np.ones(len(rows),dtype=bool)
        if system=="ICD" and not cfg["retrieval"].get("cdc_enabled",True):keep=~rows.source.isin(CDC_SOURCES).to_numpy()
        ids=np.flatnonzero(keep);rows=rows.iloc[ids].reset_index(drop=True)
        vectors=np.load(root/f"{system}_aliases.npy",mmap_mode="r")[ids]
        # key() runs a per-character loop; cache it against the exact alias texts.
        cache=path_for(cfg,"run_dir")/"pretrain_v2"/"cache"/f"{run}_{profile}_{system}_keys.joblib"
        fingerprint=hashlib.sha256("\n".join(rows.text).encode()).hexdigest()
        keys=None
        if cache.exists():
            stored,values=joblib.load(cache)
            if stored==fingerprint:keys=values
        if keys is None:
            keys=rows.text.map(key).to_numpy(dtype=object)
            cache.parent.mkdir(parents=True,exist_ok=True);joblib.dump((fingerprint,keys),cache)
        return cls(system,rows,vectors,keys)

    def gold_keys(self,code):
        return set(self.keys[self.by_code[code]])

    def masked(self,code,keys,self_keys=()):
        """Alias rows that may not act as negatives (or, for self_keys, positives)."""
        found=[self.by_key[k] for k in keys if k in self.by_key]
        ids=np.unique(np.concatenate(found)) if found else np.empty(0,np.int64)
        ids=ids[self.alias_code[ids]!=self.code_index[code]]
        own=[self.by_key[k] for k in self_keys if k in self.by_key]
        if own:ids=np.union1d(ids,np.concatenate(own))
        return ids.astype(np.int64)

    def usable(self,code,masked):
        return bool(np.setdiff1d(self.by_code[code],masked).size)

    def tensors(self,device):
        import torch
        if str(device) not in self.device:
            # Built outside inference mode so the training loss can use them after a monitor pass.
            with torch.inference_mode(False),torch.no_grad():
                self.device[str(device)]=dict(vectors=torch.tensor(self.vectors,device=device,dtype=torch.float32),
                                              alias_code=torch.tensor(self.alias_code,device=device))
        return self.device[str(device)]


def train_frequency(cfg):
    frame=pd.read_parquet(term_dir(cfg)/"priors_train.parquet")
    return Counter({(r.system,r.code):int(r.frequency) for r in frame.itertuples()})


def mention_examples(cfg,indexes,dataset,held,frequency):
    """One example per labelled row, with the same dense views the pipeline searches."""
    out=[];skipped=Counter()
    for r in joblib.load(directory(cfg)/"records.joblib")[dataset]:
        if not r.get("valid_gold",True):skipped["invalid_gold"]+=1;continue
        # A training query may not be a held-out probe name (the memory channel drops them too).
        if dataset=="train" and key(r["entity"]) in held:skipped["probe_name"]+=1;continue
        index=indexes[r["system"]];code=r["code"]
        if code not in index.by_code:skipped["gold_not_indexed"]+=1;continue
        texts=list(r["dense_queries"])
        masked=index.masked(code,index.gold_keys(code)|{key(t) for t in texts})
        if not index.usable(code,masked):skipped["gold_fully_masked"]+=1;continue
        out.append(dict(system=r["system"],code=code,gold=index.code_index[code],texts=texts,masked=masked,weight=1.,
                        source="mention",row_id=str(r["row_id"]),group_id=str(r["group_id"]),subconcept=r.get("subconcept"),
                        bucket=bucket(frequency[(r["system"],code)]),variant=r.get("variant")))
    return out,skipped


def weight_rare(examples,frequency):
    """1/sqrt(train frequency), scaled to mean 1: rare codes count more, every row counts."""
    raw=np.array([1/math.sqrt(frequency[(e["system"],e["code"])]) for e in examples])
    for e,w in zip(examples,raw/raw.mean()):e["weight"]=float(w)
    return examples


class AliasSampler:
    """Dictionary-name queries (clean or with realistic noise) for concepts outside the probe."""
    def __init__(self,cfg,indexes,probe_codes,settings):
        out=term_dir(cfg);self.indexes=indexes;self.kinds=list(settings["noise_kinds"])
        self.abbreviations=set(pd.read_parquet(out/"abbreviations.parquet").abbr)
        train=pd.read_parquet(out/"train.parquet");train=train.loc[train.valid_gold]
        tags=pd.read_parquet(out/"snomed_concepts.parquet",columns=["code","tag"]).set_index("code").tag
        train_tags=Counter(tags.reindex(train.loc[train.system.eq("SNOMED"),"code"]).fillna(""))
        self.codes={};self.queries={};self.p={}
        for system,index in indexes.items():
            sources=index.rows.source.to_numpy();codes=[];queries=[]
            for code,ids in index.by_code.items():
                if (system,code) in probe_codes or len(set(index.keys[ids]))<2:continue
                own=[i for i in ids if sources[i]!="dict_fsn" and index.keys[i]
                     and (index.alias_code[index.by_key[index.keys[i]]]==index.alias_code[i]).all()]
                if own:codes.append(code);queries.append(np.array(own))
            weight=np.ones(len(codes))
            if system=="SNOMED":
                tag=tags.reindex(codes).fillna("").to_numpy();size=Counter(tag)
                weight=np.array([(train_tags[t]+1)/size[t] for t in tag])
            self.codes[system]=codes;self.queries[system]=queries;self.p[system]=weight/weight.sum()

    def sample(self,epoch,n):
        rng=np.random.default_rng([13,epoch]);out=[]
        for s,system in enumerate(SYSTEMS):
            index=self.indexes[system];count=n//2 if s==0 else n-n//2;made=0;attempt=0
            while made<count:
                attempt+=1;j=rng.choice(len(self.codes[system]),p=self.p[system])
                code=self.codes[system][j];alias=int(rng.choice(self.queries[system][j]))
                text=index.rows.text.iat[alias];base=key(text);query=base;kind="none"
                if made%2:
                    kind=self.kinds[int(rng.integers(len(self.kinds)))]
                    query=clean(corrupt(text,kind,13,f"v2:{epoch}:{system}:{attempt}",self.abbreviations)[0])
                if not query:continue
                masked=index.masked(code,index.gold_keys(code)|{key(query)},{base,key(query)})
                if not index.usable(code,masked):continue
                out.append(dict(system=system,code=code,gold=index.code_index[code],texts=[query],masked=masked,weight=1.,
                                source="alias_noisy" if query!=base else "alias_clean",noise=kind,row_id=f"v2alias:{epoch}:{system}:{made}",
                                group_id=f"v2alias:{epoch}:{system}:{made}",subconcept=None,bucket=None,variant=None))
                made+=1
        return out


def encode(model,tokenizer,texts,max_length):
    """Same as Encoder.encode: fixed padding, bf16 autocast, normalized [CLS]."""
    import torch
    device=next(model.parameters()).device
    tokens=tokenizer(texts,padding="max_length",max_length=max_length,truncation=True,return_tensors="pt").to(device)
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):output=model(**tokens).last_hidden_state[:,0,:]
    return torch.nn.functional.normalize(output.float(),dim=-1)


def concept_scores(query,views,index,masked=None):
    """Dense-channel scores: max over views and over each concept's names (B x concepts)."""
    import torch
    g=index.tensors(query.device);q=query[views]
    alias=torch.einsum("bvd,nd->bvn",q,g["vectors"]).amax(1)
    if masked is not None:alias=alias.masked_fill(masked,float("-inf"))
    out=torch.full((len(alias),len(index.codes)),float("-inf"),device=query.device)
    return out.scatter_reduce(1,g["alias_code"].expand(len(alias),-1),alias,"amax",include_self=True)


def view_index(examples,position,device):
    """Rows with fewer views repeat their last view, which leaves the max unchanged."""
    import torch
    width=max(len(e["texts"]) for e in examples)
    return torch.tensor([[position[e["texts"][min(j,len(e["texts"])-1)]] for j in range(width)] for e in examples],device=device)


def mask_tensor(examples,n,device):
    import torch
    rows=np.concatenate([np.full(len(e["masked"]),i) for i,e in enumerate(examples)]).astype(np.int64)
    cols=np.concatenate([e["masked"] for e in examples]).astype(np.int64)
    mask=torch.zeros((len(examples),n),dtype=torch.bool,device=device)
    if len(rows):mask[torch.tensor(rows,device=device),torch.tensor(cols,device=device)]=True
    return mask


def info_nce(scores,gold,tau):
    import torch
    logits=scores/tau
    return torch.logsumexp(logits,1)-logits.gather(1,gold[:,None]).squeeze(1)


def gold_rank(scores,gold):
    """Pipeline rank: fp16 scores, ties broken by code string (codes are sorted)."""
    import torch
    s=scores.half().float();g=s.gather(1,gold[:,None])
    order=torch.arange(s.shape[1],device=s.device)[None,:]
    return ((s>g)|((s==g)&(order<gold[:,None]))).sum(1)+1


def batch_loss(model,tokenizer,examples,indexes,settings):
    """Backward per system; returns per-example losses for logging."""
    import torch
    total=sum(e["weight"] for e in examples);losses=[]
    for system in SYSTEMS:
        part=[e for e in examples if e["system"]==system]
        if not part:continue
        index=indexes[system];texts=sorted({t for e in part for t in e["texts"]});position={t:i for i,t in enumerate(texts)}
        query=encode(model,tokenizer,texts,settings["max_length"]);device=query.device
        scores=concept_scores(query,view_index(part,position,device),index,mask_tensor(part,len(index.rows),device))
        loss=info_nce(scores,torch.tensor([e["gold"] for e in part],device=device),settings["temperature"])
        weight=torch.tensor([e["weight"] for e in part],device=device)
        ((loss*weight).sum()/total).backward()
        losses.extend(zip(part,loss.detach().cpu().tolist()))
        del query,scores,loss
    return losses


def memory_rank(query,views,memory,gold,groups,n_codes):
    """Memory channel: best train mention per code, same chunk group excluded."""
    import torch
    scores=torch.einsum("bvd,nd->bvn",query[views],memory["vectors"]).amax(1).half().float()
    same=torch.tensor(np.array([memory["group"]==g for g in groups]),device=query.device)
    scores=scores.masked_fill(same,float("-inf"))
    per=torch.full((len(gold),n_codes),float("-inf"),device=query.device)
    per=per.scatter_reduce(1,memory["code"].expand(len(gold),-1),scores,"amax",include_self=True)
    found=torch.isfinite(per.gather(1,gold[:,None]).squeeze(1))
    return torch.where(found,gold_rank(per,gold),torch.full_like(gold,10**9))


def score_examples(vectors,position,part,index,tau,memory=None):
    """Exact gold rank (unmasked, like the pipeline), masked validation loss, memory rank."""
    import torch
    rank=[];loss=[];mrank=[];device=vectors.device
    for start in range(0,len(part),64):
        chunk=part[start:start+64];views=view_index(chunk,position,device)
        gold=torch.tensor([e["gold"] for e in chunk],device=device)
        rank.append(gold_rank(concept_scores(vectors,views,index),gold))
        loss.append(info_nce(concept_scores(vectors,views,index,mask_tensor(chunk,len(index.rows),device)),gold,tau))
        if memory is not None:mrank.append(memory_rank(vectors,views,memory,gold,[e["group_id"] for e in chunk],len(index.codes)))
    out=[torch.cat(x).cpu().numpy() for x in (rank,loss)]
    return out[0],out[1],torch.cat(mrank).cpu().numpy() if mrank else None


class Monitor:
    """Exact dense / memory recall and validation loss on fixed sets, without gradients."""
    def __init__(self,cfg,indexes,sets,settings):
        self.indexes=indexes;self.sets=sets;self.settings=settings
        root=directory(cfg)/"C1";self.memory={}
        for system,index in indexes.items():
            rows=pd.read_parquet(root/f"{system}_memory_rows.parquet")
            self.memory[system]=dict(texts=rows.text.tolist(),group=rows.group_id.astype(str).to_numpy(),
                                     code=rows.code.map(index.code_index).to_numpy(np.int64))
        rng=np.random.default_rng(13);self.drift={}
        for system,index in indexes.items():
            ids=np.sort(rng.choice(len(index.rows),settings["drift_names"],replace=False))
            self.drift[system]=(index.rows.text.iloc[ids].tolist(),ids)

    def __call__(self,model,tokenizer):
        import torch
        was=model.training;model.eval();result={};device=next(model.parameters()).device
        with torch.inference_mode():
            for system,index in self.indexes.items():
                memory=self.memory[system];drift_texts,drift_ids=self.drift[system]
                texts=sorted({t for examples in self.sets.values() for e in examples if e["system"]==system for t in e["texts"]}|set(memory["texts"])|set(drift_texts))
                position={t:i for i,t in enumerate(texts)}
                vectors=torch.cat([encode(model,tokenizer,texts[i:i+512],self.settings["max_length"]) for i in range(0,len(texts),512)])
                # Query-side drift on dictionary text: 1.0 means unchanged from the frozen index.
                original=index.tensors(device)["vectors"][torch.tensor(drift_ids,device=device)]
                result[f"{system}/drift_cos"]=float((vectors[[position[t] for t in drift_texts]]*original).sum(1).mean())
                mem=dict(memory,vectors=vectors[[position[t] for t in memory["texts"]]],code=torch.tensor(memory["code"],device=device))
                for name,examples in self.sets.items():
                    part=[e for e in examples if e["system"]==system]
                    if not part:continue
                    rank,loss,mrank=score_examples(vectors,position,part,index,self.settings["temperature"],None if name=="probe" else mem)
                    groups={"all":np.ones(len(part),dtype=bool)}
                    for field in ["bucket","variant"]:
                        for value in sorted({e[field] for e in part if e[field]}):groups[value]=np.array([e[field]==value for e in part])
                    for label,active in groups.items():
                        prefix=f"{name}/{system}" if label=="all" else f"{name}/{system}/{label}"
                        result[prefix+"/n"]=int(active.sum());result[prefix+"/loss"]=float(loss[active].mean())
                        for k in (1,10,64):result[prefix+f"/R@{k}"]=float((rank[active]<=k).mean()*100)
                        if mrank is not None:result[prefix+"/memory_R@10"]=float((mrank[active]<=10).mean()*100)
                del vectors,mem
        if device.type=="cuda":torch.cuda.empty_cache()
        model.train(was);return result


def schedule(step,total,warmup):
    return (step+1)/warmup if step<warmup else max(0.,(total-step)/max(1,total-warmup))


def parameter_groups(model,weight_decay):
    decay=[p for n,p in model.named_parameters() if p.requires_grad and not (n.endswith("bias") or "LayerNorm" in n)]
    other=[p for n,p in model.named_parameters() if p.requires_grad and (n.endswith("bias") or "LayerNorm" in n)]
    return [dict(params=decay,weight_decay=weight_decay),dict(params=other,weight_decay=0.)]


def baseline_check(cfg,metrics,tolerance=.5):
    """Step 0 is the off-the-shelf model: it must reproduce the C1 dense channel and index."""
    pools=joblib.load(path_for(cfg,"run_dir")/"C1"/"dev_pools.joblib");hits=defaultdict(list)
    for item in pools:
        row=item["row"]
        if row.get("valid_gold",True):hits[row["system"]].append(row["code"] in [c for c,_ in item["channels"]["dense"][:64]])
    report={}
    for s in SYSTEMS:
        pipeline=float(np.mean(hits[s])*100);monitor=metrics[f"dev/{s}/R@64"];report[s]=dict(pipeline=pipeline,monitor=monitor)
        if abs(pipeline-monitor)>tolerance:raise RuntimeError(f"Monitor disagrees with the C1 dense channel for {s}: {monitor} vs {pipeline}")
        if metrics[f"{s}/drift_cos"]<.995:raise RuntimeError(f"Frozen index vectors do not match the base model for {s}: {metrics[f'{s}/drift_cos']}")
    print("Baseline check passed",report,flush=True);return report


def load_data(cfg,settings,smoke=False):
    out=term_dir(cfg);probe=pd.read_parquet(out/"probe_set.parquet")
    held=set(probe.heldout_norm);probe_codes=set(zip(probe.system,probe.code));frequency=train_frequency(cfg)
    indexes={s:FrozenIndex.load(cfg,s) for s in SYSTEMS}
    train,skipped=mention_examples(cfg,indexes,"train",held,frequency);train=weight_rare(train,frequency)
    dev,dev_skipped=mention_examples(cfg,indexes,"dev",held,frequency)
    probe_rows,_=mention_examples(cfg,indexes,"probe",held,frequency)
    rng=np.random.default_rng(13);audit=[train[i] for i in np.sort(rng.choice(len(train),settings["train_audit_rows"],replace=False))]
    sets=dict(dev=dev,audit=audit,probe=probe_rows)
    if smoke:
        per_system=lambda rows,n:[e for s in SYSTEMS for e in [x for x in rows if x["system"]==s][:n]]
        sets={k:per_system(v,48) for k,v in sets.items()};train=per_system(train,128)
    sampler=AliasSampler(cfg,indexes,probe_codes,settings)
    return indexes,train,sets,sampler,dict(train_skipped=dict(skipped),dev_skipped=dict(dev_skipped))


def fit(cfg,smoke=False):
    import torch
    from transformers import AutoModel,AutoTokenizer
    from termmap.retrieval.dense import require_allocation
    require_allocation();seed_everything(13,True);cfg=pinned(cfg)
    settings=dict(cfg["training_v2"]);settings["max_length"]=cfg["training"]["max_length"]
    if smoke:settings.update(max_epochs=2,eval_every=4,patience=2,train_audit_rows=64,drift_names=64)
    suffix="_smoke" if smoke else ""
    base=Path(json.loads((path_for(cfg,"external_dir")/"model_snapshot.json").read_text())["path"])
    target=path_for(cfg,"model_dir")/("sapbert_v2"+suffix);run=path_for(cfg,"run_dir")/("pretrain_v2"+suffix)
    logs=path_for(cfg,"log_dir");step_log=logs/f"pretrain_v2{suffix}_steps.jsonl";val_log=logs/f"pretrain_v2{suffix}_val.jsonl"
    for path in [step_log,val_log]:path.unlink(missing_ok=True)
    if target.exists():shutil.rmtree(target)
    target.mkdir(parents=True);run.mkdir(parents=True,exist_ok=True)
    indexes,train,sets,sampler,audit=load_data(cfg,settings,smoke)
    n_alias=int(round(settings["alias_query_ratio"]*len(train)));per_epoch=math.ceil((len(train)+n_alias)/settings["batch_size"])
    total=per_epoch*settings["max_epochs"];warmup=max(1,int(round(settings["warmup_fraction"]*total)))
    write_json(target/"settings.json",dict(settings,base=str(base),objective="frozen-index InfoNCE over all concepts",
               frozen_index="C1 off-the-shelf alias vectors",train_examples=len(train),alias_examples_per_epoch=n_alias,
               steps_per_epoch=per_epoch,planned_steps=total,warmup_steps=warmup,seed=13,**audit))
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModel.from_pretrained(base,local_files_only=True).cuda().train()
    optimizer=torch.optim.AdamW(parameter_groups(model,settings["weight_decay"]),lr=settings["learning_rate"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:schedule(s,total,warmup))
    monitor=Monitor(cfg,indexes,sets,settings)
    best={s:dict(step=0,loss=None) for s in SYSTEMS};stale={s:0 for s in SYSTEMS};history=[];window=defaultdict(list)
    previous=None;streak={s:0 for s in SYSTEMS};first_overfit={s:None for s in SYSTEMS}

    def evaluate(step,epoch):
        nonlocal previous
        tick=time.perf_counter();metrics=monitor(model,tokenizer)
        record=dict(step=step,epoch=round(step/per_epoch,3),completed_epochs=epoch,lr=scheduler.get_last_lr()[0],eval_seconds=round(time.perf_counter()-tick,1),**metrics)
        improved=[]
        for s in SYSTEMS:
            loss=metrics[f"dev/{s}/loss"]
            if best[s]["loss"] is None or loss<best[s]["loss"]-settings["min_delta"]:
                best[s]=dict(step=step,loss=loss,**{k:metrics[f"dev/{s}/{k}"] for k in ["R@1","R@10","R@64"]})
                stale[s]=0;improved.append(s)
            elif step>warmup:stale[s]+=1
            if previous is not None:
                # Overfitting: the trained rows keep improving while dev gets worse.
                overfit=loss>previous[f"dev/{s}/loss"] and metrics[f"audit/{s}/loss"]<previous[f"audit/{s}/loss"]
                streak[s]=streak[s]+1 if overfit else 0
                if streak[s]>=2 and first_overfit[s] is None:first_overfit[s]=step
        record.update(improved=improved,stale=dict(stale),overfit_streak=dict(streak),first_overfit_step=dict(first_overfit))
        with val_log.open("a") as f:f.write(json.dumps(record)+"\n")
        history.append(record);previous=metrics
        if step>0:
            for s in improved:model.save_pretrained(target/f"best_{s}");tokenizer.save_pretrained(target/f"best_{s}")
        print(f"eval step {step} epoch {record['epoch']}: "+" ".join(f"{s} dev R@64 {metrics[f'dev/{s}/R@64']:.2f} loss {metrics[f'dev/{s}/loss']:.4f} audit loss {metrics[f'audit/{s}/loss']:.4f} probe R@64 {metrics[f'probe/{s}/R@64']:.2f} mem R@10 {metrics[f'dev/{s}/memory_R@10']:.2f} drift {metrics[f'{s}/drift_cos']:.4f}" for s in SYSTEMS)+f" improved {improved}",flush=True)
        return all(stale[s]>=settings["patience"] for s in SYSTEMS)

    evaluate(0,0)
    if not smoke:baseline_check(cfg,history[0])
    step=0;stop=False;started=time.perf_counter()
    for epoch in range(1,settings["max_epochs"]+1):
        examples=train+sampler.sample(epoch,n_alias)
        order=np.random.default_rng([13,epoch]).permutation(len(examples))
        for start in range(0,len(order),settings["batch_size"]):
            batch=[examples[i] for i in order[start:start+settings["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            losses=batch_loss(model,tokenizer,batch,indexes,settings)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings["grad_clip"])
            optimizer.step();scheduler.step();step+=1
            for e,value in losses:window[f"{e['source']}/{e['system']}"].append(value)
            window["grad_norm"].append(float(norm))
            if step%settings["log_every"]==0:
                record=dict(step=step,epoch=round(step/per_epoch,3),lr=scheduler.get_last_lr()[0],seconds=round(time.perf_counter()-started,1),
                            **{f"loss/{k}" if k!="grad_norm" else k:float(np.mean(v)) for k,v in window.items()},**{f"n/{k}":len(v) for k,v in window.items() if k!="grad_norm"})
                with step_log.open("a") as f:f.write(json.dumps(record)+"\n")
                window.clear()
            if step%settings["eval_every"]==0 and evaluate(step,epoch-1):stop=True;break
        if stop:break
        if step%settings["eval_every"]:
            stop=evaluate(step,epoch)
            if stop:break
    selection={}
    for s in SYSTEMS:
        selection[s]=dict(best[s],epochs=round(best[s]["step"]/per_epoch,3),
                          checkpoint=str(target/f"best_{s}") if best[s]["step"]>0 else str(base),adapted=best[s]["step"]>0)
    write_json(target/"selection.json",dict(selection=selection,steps_run=step,steps_per_epoch=per_epoch,planned_steps=total,
               stopped_early=stop,first_overfit_step=first_overfit,selection_rule="minimum dev validation loss per system; patience after warmup"))
    joblib.dump(history,run/"history.joblib")
    model.cpu();del model,optimizer;gc.collect();torch.cuda.empty_cache()
    print("P2-v2 training finished",json.dumps(selection),flush=True)
    return selection
