import argparse
import gc
import json
import time
from collections import Counter
import joblib
import pandas as pd
import torch
from termmap.common.io import load_config,path_for,term_dir,write_json,seed_everything
from termmap.retrieval.cache import encode_dense,directory
from termmap.retrieval.candidates import CachedRetriever
from termmap.evaluation.metrics import evaluate
from scripts.evaluation.checkpoints import save_checkpoint,probe_summary
from termmap.decision.coupling import other_row,infer_types

def evaluate_saved(cfg,retriever,items,name,dataset="dev"):
    rows=pd.DataFrame([x["row"] for x in items]);pools=[x["pool"] for x in items]
    if dataset=="probe":return probe_summary(rows,pools)
    freq=pd.read_parquet(term_dir(cfg)/"priors_train.parquet")
    f={(r.system,r.code):r.frequency for r in freq.itertuples()}
    metrics,details=evaluate(rows,pools,f,retriever.icd,retriever.sn,cfg["split"]["bootstrap_samples"],13,
                             [x.get("seconds",0) for x in items],torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,retriever.near)
    target=path_for(cfg,"run_dir")/name;target.mkdir(parents=True,exist_ok=True)
    details.drop(columns=[c for c in ["entity","chunk_text","display"] if c in details]).to_parquet(target/f"{dataset}_details.parquet",index=False)
    return metrics

def retrieve_datasets(cfg,run,profile="eval",smoke=False,checks=False,datasets=None):
    cache=directory(cfg,profile);retriever=CachedRetriever(cfg,run,profile)
    records=joblib.load(cache/"records.joblib");outputs={}
    infer_types(cfg,retriever,records,run)
    for dataset,rows in records.items():
        if datasets is not None and dataset not in datasets:continue
        if smoke:rows=rows[:2]
        path=path_for(cfg,"run_dir")/run/f"{dataset}_pools.joblib"
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and not smoke:outputs[dataset]=joblib.load(path);continue
        items=[]
        for i,row in enumerate(rows):
            start=time.perf_counter();item=retriever.retrieve(row)
            if dataset in {"dev","probe","test"}:item["other_system"]=retriever.retrieve(other_row(row,cfg))
            item["seconds"]=time.perf_counter()-start;items.append(item)
            if i%250==0:print(run,dataset,i,"/",len(rows),flush=True)
        outputs[dataset]=items
        if not smoke:joblib.dump(items,path)
    if smoke:return
    if "dev" in outputs:
        metrics=evaluate_saved(cfg,retriever,outputs["dev"],run)
        probe=evaluate_saved(cfg,retriever,outputs["probe"],run,"probe") if "probe" in outputs else None
        save_checkpoint(cfg,run,metrics,probe)
        misses=Counter()
        for item in outputs["dev"]:
            row=item["row"];gold=row["code"]
            if gold in [c for c,_ in item["pool"]]:continue
            if not row.get("valid_gold",True):cause="label noise"
            elif gold not in retriever.icd and row["system"]=="ICD":cause="alias gap"
            elif any(any(t!=row["entity"] for t in row["views"][v]) for v in ["V3","V4"]):cause="noise"
            elif any(t!=row["views"]["V1"][0] for t in row["views"]["V2"]):cause="abbreviation"
            elif row["system"]=="ICD" and any(c[:3]==gold[:3] for c,_ in item["pool"]):cause="specificity"
            else:cause="semantic distance"
            item["miss_cause"]=cause;misses[(row["system"],cause)]+=1
        write_json(path_for(cfg,"run_dir")/run/"miss_causes.json",{f"{s}/{c}":n for (s,c),n in misses.items()})
        joblib.dump(outputs["dev"],path_for(cfg,"run_dir")/run/"dev_pools.joblib")
        if checks:
            results={}
            for component in ["cdc","bridge","subtree","family","lookalike"]:
                flag=component+"_enabled";config=dict(cfg["retrieval"]);config[flag]=False
                affected="ICD" if component in {"cdc","bridge","subtree"} else "SNOMED"
                alternative=[retriever.retrieve(row,config) if row["system"]==affected else base for row,base in zip(records["dev"],outputs["dev"])]
                other=evaluate_saved(cfg,retriever,alternative,run+"_without_"+component)
                alternate_probe=[retriever.retrieve(row,config) if row["system"]==affected else base for row,base in zip(records["probe"],outputs["probe"])]
                other_probe=evaluate_saved(cfg,retriever,alternate_probe,run+"_without_"+component,"probe")
                old=metrics["overall"]["R@64"];new=other["overall"]["R@64"]
                keep=old>new and metrics["headline_min_acc"]>=other["headline_min_acc"]
                keep=keep and all(metrics["systems"][s]["buckets"]["unseen"]["acc@1"]>=other["systems"][s]["buckets"]["unseen"]["acc@1"]-.5 and probe[s]["all"]["acc@1"]>=other_probe[s]["all"]["acc@1"]-1 for s in ["ICD","SNOMED"])
                results[component]=dict(with_R64=old,without_R64=new,keep=keep,without_probe=other_probe)
            write_json(path_for(cfg,"run_dir")/run/"component_checks.json",results)
            dropped=[k for k,v in results.items() if not v["keep"]]
            if dropped:
                # One keep/drop pass, no tuning grid. Recompose saved channels using dropped flags.
                for k in dropped:cfg["retrieval"][k+"_enabled"]=False
                for dataset,rows in records.items():
                    outputs[dataset]=[retriever.retrieve(row) for row in rows]
                    joblib.dump(outputs[dataset],path_for(cfg,"run_dir")/run/f"{dataset}_pools.joblib")
                metrics=evaluate_saved(cfg,retriever,outputs["dev"],run)
                probe=evaluate_saved(cfg,retriever,outputs["probe"],run,"probe")
                save_checkpoint(cfg,run,metrics,probe)
            write_json(path_for(cfg,"run_dir")/"retrieval_policy.json",cfg["retrieval"])
            text="# Retrieval miss causes\n\n| System / cause | n |\n|---|---:|\n"
            for (s,c),n in sorted(misses.items()):text+=f"| {s}/{c} | {n} |\n"
            text+="\nHeuristic attribution is diagnostic; chart-specific coder-convention judgments are not asserted automatically.\n\n"
            text+=f"One-time source checks: {results}. Recall targets are reported without repeated tuning.\n"
            (path_for(cfg,"report_dir")/"miss_causes.md").write_text(text)
    return outputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");args=ap.parse_args();cfg=load_config()
    seed_everything(13,True)
    if args.smoke:
        from termmap.retrieval.dense import Encoder,DenseIndex
        from termmap.retrieval.lexical import CharIndex,BM25Index,rrf,expand_pool
        model=json.loads((path_for(cfg,"external_dir")/"model_snapshot.json").read_text())["path"]
        enc=Encoder(model,40,8)
        a=pd.DataFrame(dict(code=["I10","I50.9"],norm_text=["hypertension","heart failure"],text=["hypertension","heart failure"]))
        e=enc.encode(a.text.tolist());index=DenseIndex(a,e)
        char=CharIndex(a);word=BM25Index(a)
        channels={"dense":index.search(e,2)[0],"char":char.search(["hypertension"],2)[0],"bm25":word.search(["hypertension"],2)[0]}
        pool=rrf(channels,60,50)
        assert pool[0][0]=="I10"
        smoke=path_for(cfg,"run_dir")/"C1_smoke";smoke.mkdir(parents=True,exist_ok=True)
        joblib.dump([pool],smoke/"pools.joblib")
        assert len(joblib.load(smoke/"pools.joblib")[0])<=64
        print("C1 smoke passed: encoding, exact matmul, lexical channels, fusion, pool persistence");return
    model=json.loads((path_for(cfg,"external_dir")/"model_snapshot.json").read_text())["path"]
    encode_dense(cfg,model,"C1")
    retrieve_datasets(cfg,"C1",checks=True)

if __name__=="__main__":main()
