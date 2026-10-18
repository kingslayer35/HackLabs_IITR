"""Score the prescribed two-by-two experiment, reusing payload scores locally."""
import argparse
from collections import defaultdict
import json
import time
import joblib
import pandas as pd
import torch
from termmap.common.io import load_config,path_for,term_dir,write_json,sha256
from termmap.reranking.model import Reranker
from termmap.evaluation.metrics import evaluate
from scripts.evaluation.checkpoints import probe_summary,save_checkpoint


def metadata(cfg):
    out=term_dir(cfg)
    icd=pd.read_parquet(out/"icd_concepts.parquet").set_index("code").to_dict("index")
    sn=pd.read_parquet(out/"snomed_concepts.parquet").set_index("code").to_dict("index")
    links=defaultdict(set);children=defaultdict(set)
    for r in pd.read_parquet(out/"snomed_links.parquet").itertuples():
        if r.kind in {"specifier_child","qualifier_sibling"}:links[r.src].add(r.dst);links[r.dst].add(r.src)
        if r.kind=="specifier_child":children[r.src].add(r.dst)
    links["__specifier_children__"]=children
    freq={(r.system,r.code):r.frequency for r in pd.read_parquet(out/"priors_train.parquet").itertuples()}
    return icd,sn,links,freq


def ranked(items):
    return [[(c,round(float(s),5)) for c,s in sorted(x["reranker_scores"].items(),key=lambda kv:(-round(kv[1],5),kv[0]))] for x in items]


def raw_metrics(cfg,items,name,dataset="dev",structures=None):
    icd,sn,links,freq=structures or metadata(cfg)
    rows=pd.DataFrame([x["row"] for x in items]);pools=ranked(items)
    if "entity" not in rows:raise ValueError("Missing mention metadata")
    for col in ["chunk_text","subconcept"]:
        if col not in rows:rows[col]="" if col=="chunk_text" else "Unknown"
    metrics,details=evaluate(rows,pools,freq,icd,sn,cfg["split"]["bootstrap_samples"],13,
                             [x.get("rerank_seconds",0) for x in items],torch.cuda.max_memory_allocated(),links)
    dest=path_for(cfg,"run_dir")/name
    write_json(dest/f"{dataset}_raw_metrics.json",metrics)
    details.drop(columns=[c for c in ["entity","chunk_text","display","views","context"] if c in details]).to_parquet(dest/f"{dataset}_raw_details.parquet",index=False)
    if dataset=="probe":write_json(dest/"probe_metrics.json",probe_summary(rows,pools))
    return metrics


def score_flow(cfg,flow,source,adapter=None,profile="eval",smoke=False,datasets=None,force=False):
    if smoke:cfg.setdefault("reranker",{})["cache_enabled"]=False
    ranker=Reranker(cfg,adapter=adapter,profile="final" if profile=="final" else "eval")
    torch.cuda.reset_peak_memory_stats()
    dest=path_for(cfg,"run_dir")/flow;dest.mkdir(parents=True,exist_ok=True)
    structures=metadata(cfg)
    for dataset in datasets or ["dev","probe"]:
        target=dest/f"{dataset}_scores.joblib"
        provenance=dest/f"{dataset}_score_config.json"
        expected=dict(source=source,source_sha256=sha256(path_for(cfg,"run_dir")/source/f"{dataset}_pools.joblib"),
                      template_sha256=sha256(cfg["root"]/"termmap"/"reranking/model.py"),reranker=ranker.identity,
                      packing_sha256=sha256(cfg["root"]/"termmap"/"reranking/prompts.py"),target_instruction=cfg.get("reranker",{}).get("target_instruction",False),
                      seen=cfg.get("reranker",{}).get("seen_enabled",True),other_system=True)
        if target.exists() and provenance.exists() and json.loads(provenance.read_text())==expected and not smoke and not force:continue
        items=joblib.load(path_for(cfg,"run_dir")/source/f"{dataset}_pools.joblib")
        if not smoke:assert all("other_system" in x for x in items),"Complete-flow evaluation requires both systems per input"
        if smoke:
            items=[dict(x,pool=x["pool"][:2]) for x in items[:1]]
            for x in items:
                if "other_system" in x:x["other_system"]=dict(x["other_system"],pool=x["other_system"]["pool"][:2])
        tick=time.perf_counter();scored=ranker.score_items(items)
        elapsed=time.perf_counter()-tick
        if smoke:assert len(scored[0]["reranker_scores"])==len(items[0]["pool"]);continue
        joblib.dump(scored,target)
        write_json(provenance,expected)
        raw_metrics(cfg,scored,flow,dataset,structures)
        write_json(dest/f"{dataset}_timing.json",dict(seconds=elapsed,n=len(scored),peak_vram=torch.cuda.max_memory_allocated(),
                                                    definition="Synchronized per-input scoring, including the opposite system; local payload cache enabled. Final CLI cold-process latency is reported separately."))
    ranker.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--stage",choices=["zero","fine","adapted","test","drill"],default="zero");args=ap.parse_args();cfg=load_config()
    base_source=cfg.get("evaluation",{}).get("base_source","C1")
    adapted_source=cfg.get("evaluation",{}).get("adapted_source","C2")
    best=path_for(cfg,"model_dir")/"reranker_lora"/"best"
    if args.stage=="drill":score_flow(cfg,"F11","drill",best,smoke=args.smoke,datasets=["drill"])
    elif args.stage in {"zero","fine"}:score_flow(cfg,"F00" if args.stage=="zero" else "F01","C1",None if args.stage=="zero" else best,smoke=args.smoke,datasets=["dev"])
    else:
        if args.stage=="adapted" and not args.smoke:
            score_flow(cfg,"F00",base_source,None,datasets=["dev"])
            score_flow(cfg,"F01",base_source,best,datasets=["dev"])
            score_flow(cfg,"F00",base_source,None,datasets=["probe"])
            score_flow(cfg,"F01",base_source,best,datasets=["probe"])
        systems=[("F00","test_base",None),("F01","test_base",best),("F10","test_adapted",None),("F11","test_adapted",best)] if args.stage=="test" else [("F10",adapted_source,None),("F11",adapted_source,best)]
        for flow,source,adapter in systems:score_flow(cfg,flow,source,adapter,smoke=args.smoke,datasets=["test"] if args.stage=="test" else None)
        if args.stage=="adapted" and not args.smoke:
            check_seen(cfg,best)
            m=json.loads((path_for(cfg,"run_dir")/"F11"/"dev_raw_metrics.json").read_text())
            probe=json.loads((path_for(cfg,"run_dir")/"F11"/"probe_metrics.json").read_text())
            save_checkpoint(cfg,"C3",m,probe)
            zero=json.loads((path_for(cfg,"run_dir")/"F10"/"dev_raw_metrics.json").read_text())
            gate=m["headline_min_acc"]>zero["headline_min_acc"] and all(m["systems"][s]["buckets"]["unseen"]["acc@1"]>=zero["systems"][s]["buckets"]["unseen"]["acc@1"] for s in ["ICD","SNOMED"])
            write_json(path_for(cfg,"run_dir")/"C3"/"gate.json",dict(passed=gate,selection="best complete flow on dev",one_lora_run=True))


def check_seen(cfg,best,flow="F11",source="C2"):
    combinations=[("F00","C1",None),("F10","C2",None)]
    if best is not None:combinations.extend([("F01","C1",best),("F11","C2",best)])
    path=path_for(cfg,"run_dir")/"seen_as_check.json"
    if path.exists():
        check=json.loads(path.read_text())
        if not check.get("complete",False):
            cfg.setdefault("reranker",{})["seen_enabled"]=check["keep_seen"]
            if not check["keep_seen"]:
                for target,origin,adapter in combinations:score_flow(cfg,target,origin,adapter)
            check["complete"]=True;write_json(path,check)
        return
    ranker=Reranker(cfg,best);structures=metadata(cfg);metrics={};probes={}
    for dataset in ["dev","probe"]:
        items=joblib.load(path_for(cfg,"run_dir")/source/f"{dataset}_pools.joblib")
        scored=ranker.score_items(items,seen=False)
        metrics[dataset]=raw_metrics(cfg,scored,"seen_off",dataset,structures)
        joblib.dump(scored,path_for(cfg,"run_dir")/"seen_off"/f"{dataset}_scores.joblib")
    ranker.close()
    on=json.loads((path_for(cfg,"run_dir")/flow/"dev_raw_metrics.json").read_text());off=metrics["dev"]
    onp=json.loads((path_for(cfg,"run_dir")/flow/"probe_metrics.json").read_text());offp=json.loads((path_for(cfg,"run_dir")/"seen_off"/"probe_metrics.json").read_text())
    keep=on["headline_min_acc"]>=off["headline_min_acc"] and all(on["systems"][s]["buckets"]["unseen"]["acc@1"]>=off["systems"][s]["buckets"]["unseen"]["acc@1"]-.5 and onp[s]["all"]["acc@1"]>=offp[s]["all"]["acc@1"]-1 for s in ["ICD","SNOMED"])
    write_json(path,dict(keep_seen=keep,with_min_acc=on["headline_min_acc"],without_min_acc=off["headline_min_acc"],unseen_selection={s:dict(on=on["systems"][s]["buckets"]["unseen"]["selection_given_pool"],off=off["systems"][s]["buckets"]["unseen"]["selection_given_pool"]) for s in ["ICD","SNOMED"]}))
    check=json.loads(path.read_text());check.update(check_flow=flow,source=source,zero_shot=best is None);write_json(path,check)
    cfg.setdefault("reranker",{})["seen_enabled"]=keep;write_json(path_for(cfg,"run_dir")/"reranker_policy.json",cfg["reranker"])
    if not keep:
        for flow,source,adapter in combinations:score_flow(cfg,flow,source,adapter,force=True)
    check=json.loads(path.read_text());check["complete"]=True;write_json(path,check)

if __name__=="__main__":main()
