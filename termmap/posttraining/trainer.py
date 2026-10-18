"""Balanced off-the-shelf quizzes and one two-epoch LoRA run."""
from collections import Counter,defaultdict
import json
import math
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from termmap.common.io import load_config,path_for,term_dir,write_json,seed_everything
from termmap.preprocessing.noise import stable_rng
from termmap.reranking.model import Reranker
from termmap.preprocessing.normalize import clean
from termmap.terminology.hierarchy import strip_tag


def final_selected_epoch(cfg):
    """Carry DEV's epoch choice into the all-label run without test tuning."""
    frozen=json.loads((path_for(cfg,"run_dir")/"frozen/CONFIG_FROZEN.json").read_text())
    epoch=int(frozen["lora_selected_epoch"])
    if epoch not in {1,2}:raise RuntimeError("Frozen LoRA epoch is outside the fixed two-epoch recipe")
    return epoch


def quiz(item,cards,links,epoch=1):
    row=item["row"];gold=row["code"];pool=[c for c,_ in item["pool"] if c!=gold]
    rng=stable_rng(f"quiz:{epoch}:{row['row_id']}")
    selected=pool[:10]
    if row["system"]=="ICD":
        n=cards.icd[gold];p=cards.icd.get(n["parent"],{})
        neighbors=list(n["children"])+list(p.get("children",()))
    else:neighbors=links.get(gold,[])
    for c in sorted(neighbors,key=lambda c:(pool.index(c) if c in pool else 1000,c)):
        if c!=gold and c not in selected:selected.append(c)
        if len(selected)>=13:break
    remaining=[c for c in pool if c not in selected];rng.shuffle(remaining)
    selected.extend(remaining[:max(0,15-len(selected))])
    if len(selected)<15:
        universe=sorted(cards.icd if row["system"]=="ICD" else cards.sn)
        while len(selected)<15:
            c=rng.choice(universe)
            if c!=gold and c not in selected:selected.append(c)
    candidates=[gold]+selected[:15];rng.shuffle(candidates)
    return row,candidates,candidates.index(gold),rng.random()>=.3,rng.random()>=.5


def sampled_groups(cfg,train,aliases,epoch):
    valid=[x for x in train if x["row"].get("valid_gold",True)]
    freq=Counter((x["row"]["system"],x["row"]["code"]) for x in valid)
    weights=np.array([(2 if x["row"]["system"]=="ICD" else 1)/math.sqrt(freq[(x["row"]["system"],x["row"]["code"])]) for x in valid])
    weights/=weights.sum();rng=np.random.default_rng(13)
    n=len(valid)+sum(x["row"]["system"]=="ICD" for x in valid)
    selected=[];counts=Counter()
    while len(selected)<n:
        i=int(rng.choice(len(valid),p=weights));key=(valid[i]["row"]["system"],valid[i]["row"]["code"])
        if counts[key]>=10:continue
        counts[key]+=1;selected.append(valid[i])
    k=int(round(len(selected)/4));added=0
    for i in rng.permutation(len(aliases)):
        item=aliases[i];key=(item["row"]["system"],item["row"]["code"])
        if counts[key]>=10:continue
        selected.append(item);counts[key]+=1;added+=1
        if added>=k:break
    assert max(counts.values())<=10
    rng.shuffle(selected)
    return selected


def train_reranker(cfg,smoke=False,final=False):
    if smoke:cfg.setdefault("reranker",{})["cache_enabled"]=False
    fixed_final_epoch=final_selected_epoch(cfg) if final and not smoke else None
    run="final_reranker" if final else "posttrain";target=path_for(cfg,"model_dir")/("reranker_final" if final else "reranker_lora")
    if smoke:target=path_for(cfg,"run_dir")/(run+"_smoke")/"model"
    target.mkdir(parents=True,exist_ok=True)
    ranker=Reranker(cfg,train=True,profile="final" if final else "eval")
    parameters=[p for p in ranker.model.parameters() if p.requires_grad]
    trainable=sum(p.numel() for p in parameters);total_parameters=sum(p.numel() for p in ranker.model.parameters())
    write_json(target/"settings.json",dict(trainable_parameters=trainable,total_parameters=total_parameters,lora_r=16,lora_alpha=32,lora_dropout=.05,learning_rate=1e-4,effective_groups=8,max_tokens=512,epochs=2,seed=13))
    print("LoRA parameters",trainable,"/",total_parameters,flush=True)
    optimizer=torch.optim.AdamW(parameters,lr=1e-4)
    if smoke:
        row=dict(system="ICD",entity="hypertension",subconcept="DiagnosisCode",context={"window":"[M]hypertension[/M]"},views={})
        payloads=[ranker.cards.payload(row,c,seen=False) for c in ["I10","I50.9"]]
        ranker.model.train();loss=torch.nn.functional.cross_entropy(ranker.logits(payloads)[None,:],torch.tensor([0],device="cuda"));loss.backward();optimizer.step()
        ranker.model.save_pretrained(target);ranker.model.eval();print("LoRA smoke loss",float(loss.detach()));ranker.close()
        reloaded=Reranker(cfg,adapter=target,profile="final" if final else "eval")
        assert np.isfinite(reloaded.score(payloads)).all();reloaded.close();return
    base=path_for(cfg,"run_dir")/("final_pools" if final else "C1")
    train=joblib.load(base/("all_pools.joblib" if final else "train_pools.joblib"))
    train=[x for x in train if clean(strip_tag(x["row"]["entity"])) not in ranker.cards.held]
    aliases=joblib.load(base/"alias_pools.joblib")
    links=defaultdict(set)
    import pandas as pd
    for r in pd.read_parquet(term_dir(cfg)/"snomed_links.parquet").itertuples():links[r.src].add(r.dst);links[r.dst].add(r.src)
    dev=None if final else joblib.load(path_for(cfg,"run_dir")/"C1"/"dev_pools.joblib")
    epoch_groups=[sampled_groups(cfg,train,aliases,e) for e in [1,2]]
    total_steps=sum(math.ceil(len(g)/8) for g in epoch_groups)
    scheduler=get_cosine_schedule_with_warmup(optimizer,int(total_steps*.05),total_steps)
    state=target/"resume.pt";finished=0;best=-1.;best_epoch=0
    if state.exists():
        saved=torch.load(state,map_location="cpu",weights_only=False)
        ranker.model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"])
        finished=saved["epoch"];best=saved["best"];best_epoch=saved["best_epoch"]
    for epoch in [1,2]:
        epoch_dir=target/f"epoch_{epoch}"
        if epoch<=finished and (final or (epoch_dir/"dev_scores.joblib").exists()):continue
        if epoch>finished:
            seed_everything(13,True);ranker.model.train();optimizer.zero_grad(set_to_none=True);total=0.
            groups=epoch_groups[epoch-1]
            for i,item in enumerate(groups):
                row,codes,gold,use_type,use_seen=quiz(item,ranker.cards,links,epoch)
                payloads=[ranker.cards.payload(row,c,use_type,use_seen) for c in codes]
                scores=ranker.logits(payloads)
                loss=torch.nn.functional.cross_entropy(scores[None,:],torch.tensor([gold],device="cuda"))
                (loss/min(8,len(groups)-(i//8)*8)).backward();total+=float(loss.detach())
                if (i+1)%8==0 or i+1==len(groups):
                    torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                if i%100==0:print("LoRA epoch",epoch,i,"/",len(groups),"loss",total/(i+1),flush=True)
            ranker.model.save_pretrained(epoch_dir);ranker.tokenizer.save_pretrained(epoch_dir)
            # Save before validation, so an interrupted validation resumes without training again.
            torch.save(dict(model=ranker.model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch),state)
            write_json(epoch_dir/"training.json",dict(epoch=epoch,n_groups=len(groups),loss=total/max(1,len(groups)),seed=13))
        if final:score=0.;best_epoch=fixed_final_epoch
        else:
            from scripts.reranking.score import raw_metrics
            ranker.identity=f"epoch_{epoch}_{ranker.arch}";ranker.model.eval()
            ranker.connection.close();import sqlite3
            p=path_for(cfg,"run_dir")/"reranker_cache"/f"epoch_{epoch}_{ranker.arch}.sqlite"
            ranker.connection=sqlite3.connect(p);ranker.connection.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY,score REAL NOT NULL)")
            scored=ranker.score_items(dev);metrics=raw_metrics(cfg,scored,f"reranker_epoch_{epoch}")
            joblib.dump(scored,epoch_dir/"dev_scores.joblib")
            score=metrics["headline_min_acc"]
            if score>best:best=score;best_epoch=epoch
        finished=epoch;saved=torch.load(state,map_location="cpu",weights_only=False);saved.update(best=best,best_epoch=best_epoch);torch.save(saved,state)
    shutil.copytree(target/f"epoch_{best_epoch or 2}",target/"best",dirs_exist_ok=True)
    (target/"best"/"dev_scores.joblib").unlink(missing_ok=True)
    if not final:
        from termmap.common.io import sha256
        cache=path_for(cfg,"run_dir")/"reranker_cache"
        ranker.connection.commit()
        identity=sha256(target/"best"/"adapter_model.safetensors")[:20]+"_"+ranker.arch
        source=cache/f"epoch_{best_epoch}_{ranker.arch}.sqlite"
        if source.exists():shutil.copyfile(source,cache/f"{identity}.sqlite")
    write_json(target/"selection.json",dict(best_epoch=best_epoch,best_dev_min_acc=best if not final else None,
               selection_rule="frozen evaluation DEV epoch; both final epochs trained" if final else "best epoch on original DEV criterion"))
    ranker.close()
