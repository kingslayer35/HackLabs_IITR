"""P2-v2 evaluation: C2v2 index from the selected checkpoints, the real pipeline on dev and probe, pass rules.

The alias index stays the C1 (off-the-shelf) one; each system's queries and train-mention memory
are encoded with that system's checkpoint. A system that fails a pass rule falls back to the
off-the-shelf encoder and the run is rebuilt once.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from termmap.common.io import load_config,path_for,write_json,seed_everything
from termmap.retrieval.cache import directory
from termmap.evaluation.metrics import group_ci
from termmap.pretraining.trainer import SYSTEMS,bucket,train_frequency,pinned

RUN="C2v2";BASELINE="C1_v2ref"
SHARED=["query_embeddings.npy","query_embeddings_extra.npy"]  # type inference keeps the original vectors


def link(source,target):
    target.unlink(missing_ok=True);target.symlink_to(source.resolve())


def build_index(cfg,models,run=RUN):
    import torch
    from termmap.retrieval.dense import Encoder,DenseIndex
    seed_everything(13,True)
    root=directory(cfg);c1=root/"C1";target=root/run;target.mkdir(parents=True,exist_ok=True)
    queries=json.loads((root/"queries.json").read_text());rebuilt=False
    for name in SHARED:link(c1/name,target/name)
    for system,model in models.items():
        # Resumable: a system's files are reused only if they were completed for this exact model.
        marker=target/f"{system}_complete.json"
        if marker.exists() and json.loads(marker.read_text()).get("model")==str(model):continue
        marker.unlink(missing_ok=True);(target/"encoding_metadata.json").unlink(missing_ok=True);rebuilt=True
        for name in [f"{system}_aliases.npy",f"{system}_alias_rows.parquet",f"{system}_codes.npy",f"{system}_memory_rows.parquet"]:link(c1/name,target/name)
        encoder=Encoder(model,cfg["training"]["max_length"],cfg["retrieval"]["encode_batch"])
        for name in [f"{system}_query_embeddings.npy",f"{system}_memory.npy"]:(target/name).unlink(missing_ok=True)
        qvectors=encoder.encode(queries,target/f"{system}_query_embeddings.npy")
        a=pd.read_parquet(c1/f"{system}_alias_rows.parquet")
        index=DenseIndex(a,np.load(c1/f"{system}_aliases.npy",mmap_mode="r"),cfg["retrieval"]["dense_chunk"])
        assert (index.codes.astype(str)==np.load(c1/f"{system}_codes.npy")).all()
        for cdc in ([True,False] if system=="ICD" else [True]):
            allowed=None if cdc else ~a.source.isin(["icd_index","icd_inclusion"]).to_numpy()
            scores=np.lib.format.open_memmap(target/f"{system}_cdc{int(cdc)}_dense.npy",mode="w+",dtype=np.float16,shape=(len(queries),len(index.codes)))
            for start in range(0,len(queries),64):
                scores[start:start+64]=index.score_all(qvectors[start:start+64],allowed=allowed).astype(np.float16)
                if start%4096==0:print(system,"dense",start,"/",len(queries),flush=True)
            scores.flush();del scores
        del index
        memory=pd.read_parquet(c1/f"{system}_memory_rows.parquet")
        mvectors=encoder.encode(memory.text.tolist(),target/f"{system}_memory.npy")
        scores=np.lib.format.open_memmap(target/f"{system}_memory_scores.npy",mode="w+",dtype=np.float16,shape=(len(queries),len(memory)))
        m=torch.tensor(np.asarray(mvectors),device="cuda",dtype=torch.float16)
        for start in range(0,len(queries),64):
            q=torch.tensor(np.asarray(qvectors[start:start+64]),device="cuda",dtype=torch.float16)
            scores[start:start+64]=(q@m.T).cpu().numpy()
        scores.flush();del scores,m,encoder;torch.cuda.empty_cache()
        write_json(marker,dict(model=str(model),queries=len(queries)))
    write_json(target/"encoding_metadata.json",dict(models={s:str(m) for s,m in models.items()},queries=len(queries),seed=13,profile="eval",
               alias_index="C1 off-the-shelf alias vectors (frozen)",query_side="per-system checkpoint"))
    return rebuilt


def baseline(cfg,retrieve_datasets):
    """Off-the-shelf C1 index re-run through today's retrieval code: the like-for-like reference."""
    root=directory(cfg);target=root/BASELINE;target.mkdir(parents=True,exist_ok=True)
    for f in (root/"C1").iterdir():
        if f.is_file():link(f,target/f.name)
    retrieve_datasets(cfg,BASELINE,datasets={"dev","probe"})


def ranks(items,channel):
    out=[]
    for item in items:
        row=item["row"];codes=[c for c,_ in item["channels"][channel]]
        out.append(dict(system=row["system"],code=row["code"],group=str(row["group_id"]),rank=codes.index(row["code"])+1 if row["code"] in codes else 10**9))
    return pd.DataFrame(out)


def summary(cfg,run):
    folder=path_for(cfg,"run_dir")/run;frequency=train_frequency(cfg)
    dev=joblib.load(folder/"dev_pools.joblib");probe=joblib.load(folder/"probe_pools.joblib")
    metrics=json.loads((folder/"metrics.json").read_text());pm=json.loads((folder/"probe_metrics.json").read_text())
    dense=ranks(dev,"dense");memory=ranks(dev,"memory");pdense=ranks(probe,"dense")
    dense["bucket"]=[bucket(frequency[(s,c)]) for s,c in zip(dense.system,dense.code)]
    result={}
    for s in SYSTEMS:
        d=dense.loc[dense.system.eq(s)];r={f"dense_R@{k}":float((d["rank"]<=k).mean()*100) for k in (1,10,64)}
        for b in ["unseen","rare","common"]:r[f"dense_{b}_R@64"]=float((d.loc[d.bucket.eq(b),"rank"]<=64).mean()*100)
        r["memory_R@10"]=float((memory.loc[memory.system.eq(s),"rank"]<=10).mean()*100)
        p=pdense.loc[pdense.system.eq(s)];r["probe_dense_R@1"]=float((p["rank"]<=1).mean()*100);r["probe_dense_R@64"]=float((p["rank"]<=64).mean()*100)
        system=metrics["systems"][s]
        r.update({"pipeline_R@64":system["R@64"],"pipeline_R@10":system["R@10"],"pipeline_acc":system["acc@1"],
                  "pipeline_unseen_R@64":system["buckets"]["unseen"]["R@64"],"probe_pipeline_R@64":pm[s]["all"]["R@64"]})
        result[s]=r
    return result,dense


RULES=[("dense_R@64",1.0,"dev dense-only R@64 ≥ original + 1.0"),("dense_R@1",0.0,"dev dense-only R@1 not lower"),
       ("dense_unseen_R@64",-0.5,"unseen-code dense R@64 ≥ original − 0.5"),("probe_dense_R@64",-1.0,"probe dense R@64 ≥ original − 1.0"),
       ("pipeline_R@64",0.0,"full-pipeline dev R@64 not lower than C1"),("memory_R@10",0.0,"memory channel R@10 not lower")]


def gate(new,old):
    return {s:{k:dict(new=new[s][k],old=old[s][k],delta=new[s][k]-old[s][k],passed=new[s][k]>=old[s][k]+m) for k,m,_ in RULES} for s in SYSTEMS}


def curves(cfg,selection):
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[json.loads(l) for l in (path_for(cfg,"log_dir")/"pretrain_v2_val.jsonl").open()]
    steps=[r["step"] for r in rows];panels=[("loss","Validation loss (dev) vs training-sample loss",["dev/{s}/loss","audit/{s}/loss"]),
        ("R@64","Dense-only R@64",["dev/{s}/R@64","audit/{s}/R@64"]),("R@1","Dense-only R@1 / R@10 (dev)",["dev/{s}/R@1","dev/{s}/R@10"]),
        ("probe","Probe (held-out dictionary names) dense R@64",["probe/{s}/R@64"]),("memory","Memory channel R@10",["dev/{s}/memory_R@10","audit/{s}/memory_R@10"]),
        ("drift","Query drift on dictionary names (cosine to original)",["{s}/drift_cos"])]
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for ax,(_,title,keys) in zip(axes.flat,panels):
        for s,color in [("ICD","tab:blue"),("SNOMED","tab:orange")]:
            for key,style in zip(keys,["-","--"]):
                label=key.format(s=s);ax.plot(steps,[r[label] for r in rows],style,color=color,label=label)
            ax.axvline(selection[s]["step"],color=color,alpha=.3,lw=3)
        ax.set_title(title,fontsize=10);ax.set_xlabel("step");ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.suptitle("P2-v2 SapBERT query-side adaptation (shaded line = selected checkpoint per system)")
    fig.tight_layout();target=path_for(cfg,"report_dir")/"pretrain_v2_curves.png";fig.savefig(target,dpi=110);plt.close(fig)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--skip-build",action="store_true");args=ap.parse_args()
    cfg=pinned(load_config());from scripts.retrieval.baseline import retrieve_datasets
    model_dir=path_for(cfg,"model_dir")/"sapbert_v2";base=json.loads((path_for(cfg,"external_dir")/"model_snapshot.json").read_text())["path"]
    selection=json.loads((model_dir/"selection.json").read_text())["selection"]
    models={s:selection[s]["checkpoint"] if selection[s]["adapted"] else base for s in SYSTEMS}
    folder=path_for(cfg,"run_dir")/RUN;folder.mkdir(parents=True,exist_ok=True)
    baseline(cfg,retrieve_datasets);original,old_dense=summary(cfg,BASELINE);saved_c1,_=summary(cfg,"C1")
    for attempt in range(2):
        if not (args.skip_build and attempt==0):
            if build_index(cfg,models):
                for name in ["dev_pools.joblib","probe_pools.joblib"]:(folder/name).unlink(missing_ok=True)
            retrieve_datasets(cfg,RUN,datasets={"dev","probe"})
        new,new_dense=summary(cfg,RUN);checks=gate(new,original)
        failed=[s for s in SYSTEMS if models[s]!=base and not all(v["passed"] for v in checks[s].values())]
        if attempt==0:first=dict(models=dict(models),checks=checks,summary=new)
        if not failed:break
        print("Falling back to the off-the-shelf encoder for",failed,flush=True)
        for s in failed:models[s]=base
    paired={}
    for s in SYSTEMS:
        a=old_dense.loc[old_dense.system.eq(s)].reset_index(drop=True);b=new_dense.loc[new_dense.system.eq(s)].reset_index(drop=True)
        assert (a.code==b.code).all();diff=(b["rank"]<=64).astype(float)-(a["rank"]<=64).astype(float)
        paired[s]={"delta_R@64":float(diff.mean()*100),"ci95":group_ci(diff,a.group,1000,13)}
    v1,_=summary(cfg,"C2")
    result=dict(first_attempt=first,final_models={s:str(m) for s,m in models.items()},adapted={s:models[s]!=base for s in SYSTEMS},
                original=original,saved_c1=saved_c1,v1_epoch2=v1,final=new,checks=checks,paired_dense_R64=paired,rules=[r[2] for r in RULES])
    write_json(folder/"gate_v2.json",result)
    rows=curves(cfg,selection);report(cfg,result,selection,rows)
    print(json.dumps(dict(adapted=result["adapted"],paired=paired),indent=1),flush=True)


def report(cfg,result,selection,rows):
    settings=json.loads((path_for(cfg,"model_dir")/"sapbert_v2"/"settings.json").read_text())
    sel=json.loads((path_for(cfg,"model_dir")/"sapbert_v2"/"selection.json").read_text())
    o,v,n=result["original"],result["v1_epoch2"],result["final"]
    text="# P2-v2 pre-training result\n\nDev only; the sealed test is untouched. Dense-only numbers rank the gold inside the pipeline's own dense channel (top 100), exactly as the pipeline sees it.\n\n"
    text+="## Main table (dev)\n\n| Metric | System | Off-the-shelf (C1, same code) | v1 adapted (epoch 2) | **P2-v2** | P2-v2 − C1 |\n|---|---|---:|---:|---:|---:|\n"
    for k,label in [("dense_R@1","dense R@1"),("dense_R@10","dense R@10"),("dense_R@64","dense R@64"),("dense_unseen_R@64","dense R@64, unseen codes"),
                    ("dense_rare_R@64","dense R@64, rare codes"),("dense_common_R@64","dense R@64, common codes"),("memory_R@10","memory channel R@10"),
                    ("probe_dense_R@1","probe dense R@1"),("probe_dense_R@64","probe dense R@64"),("pipeline_R@10","full pipeline R@10"),
                    ("pipeline_R@64","full pipeline R@64"),("pipeline_unseen_R@64","full pipeline R@64, unseen"),("pipeline_acc","full pipeline top-1 (no reranker)")]:
        for s in SYSTEMS:text+=f"| {label} | {s} | {o[s][k]:.2f} | {v[s][k]:.2f} | **{n[s][k]:.2f}** | {n[s][k]-o[s][k]:+.2f} |\n"
    diff={s:{k:round(result['saved_c1'][s][k]-o[s][k],2) for k in o[s] if abs(result['saved_c1'][s][k]-o[s][k])>1e-9} for s in SYSTEMS}
    text+=f"\nThe off-the-shelf column re-runs the saved C1 index through today's retrieval code ({BASELINE}), so both columns differ only in the encoder. Saved C1 minus re-run: {diff or 'identical'}. The v1 column is the saved v1 run, for reference.\n"
    text+="\nPaired grouped bootstrap (1,000 resamples by chunk group) of the dense R@64 change: "+"; ".join(f"{s} {p['delta_R@64']:+.2f} (95% CI {p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f})" for s,p in result["paired_dense_R64"].items())+".\n"
    text+="\n## Pass rules (fixed before the run)\n\n| Rule | ICD | SNOMED |\n|---|---|---|\n"
    for (k,_,label) in RULES:
        text+=f"| {label} | "+" | ".join(f"{'pass' if result['checks'][s][k]['passed'] else 'FAIL'} ({result['checks'][s][k]['delta']:+.2f})" for s in SYSTEMS)+" |\n"
    text+=f"\nAdapted encoder used: {result['adapted']}.\n\n## Training and selection\n\n"
    text+=f"- {settings['train_examples']:,} train mentions per epoch (every valid row, weight 1/√frequency) + {settings['alias_examples_per_epoch']:,} dictionary-name queries (about 40% with realistic noise); {settings['steps_per_epoch']} steps per epoch of 64 queries.\n"
    text+=f"- Loss: InfoNCE over all {', '.join(s for s in SYSTEMS)} concepts against the frozen original index, τ = {settings['temperature']}; AdamW lr {settings['learning_rate']}, {int(settings['warmup_fraction']*100)}% warmup, linear decay, clip {settings['grad_clip']}.\n"
    text+=f"- Steps run: {sel['steps_run']} of {sel['planned_steps']} planned; stopped early: {sel['stopped_early']}. Validation every {settings['eval_every']} steps.\n"
    for s in SYSTEMS:
        b=selection[s];text+=f"- **{s}**: lowest dev validation loss at step {b['step']} (epoch {b['epochs']}); first overfitting signal (dev loss up while training-sample loss down, 2 checks in a row): step {sel['first_overfit_step'][s]}.\n"
    text+="\n![curves](pretrain_v2_curves.png)\n\n## Validation log (every check)\n\n| step | epoch | ICD dev loss | ICD train-sample loss | ICD dev R@64 | SNOMED dev loss | SNOMED train-sample loss | SNOMED dev R@64 | ICD probe R@64 | SNOMED probe R@64 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        text+=f"| {r['step']} | {r['epoch']} | {r['dev/ICD/loss']:.3f} | {r['audit/ICD/loss']:.3f} | {r['dev/ICD/R@64']:.2f} | {r['dev/SNOMED/loss']:.3f} | {r['audit/SNOMED/loss']:.3f} | {r['dev/SNOMED/R@64']:.2f} | {r['probe/ICD/R@64']:.2f} | {r['probe/SNOMED/R@64']:.2f} |\n"
    (path_for(cfg,"report_dir")/"pretrain_v2.md").write_text(text)


if __name__=="__main__":main()
