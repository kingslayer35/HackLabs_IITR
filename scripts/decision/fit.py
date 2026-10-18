"""Freeze four independently cross-fitted decision layers before any test access."""
import argparse
import fcntl
from functools import partial
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from termmap.common.io import load_config,path_for,term_dir,write_json,sha256,dictionary_hash
from termmap.decision.layer import Features,cross_fit,apply_decision
from termmap.evaluation.metrics import evaluate,paired_flow_delta
from scripts.reranking.score import metadata
from scripts.evaluation.checkpoints import save_checkpoint,probe_summary
from termmap.evaluation.export import export_predictions

FLOWS=["F00","F10","F01","F11"]
DELTAS=[("F00","F10"),("F01","F11"),("F00","F01"),("F10","F11"),("F00","F11")]


def matching_fit(marker,expected):
    if not marker.exists():return False
    try:saved=json.loads(marker.read_text())
    except (OSError,json.JSONDecodeError):return False
    return saved.get("provenance")==expected and bool(saved.get("outputs")) and all(
        Path(p).is_file() and sha256(Path(p))==h for p,h in saved["outputs"].items())


def fit_one(cfg,flow,dataset,structures,features,coupling_guard):
    """Serialize only the same flow; independent flows may fit concurrently."""
    started=time.perf_counter();folder=path_for(cfg,"run_dir")/flow
    folder.mkdir(parents=True,exist_ok=True)
    models=path_for(cfg,"model_dir")/"decision";models.mkdir(parents=True,exist_ok=True)
    lock=(folder/(dataset+"_fit.lock")).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    inputs=[folder/(dataset+"_scores.joblib")]
    if dataset=="dev":inputs.append(folder/"probe_scores.joblib")
    else:inputs.append(models/(flow+".joblib"))
    expected=dict(config_sha256=sha256(cfg["config_path"]),dataset=dataset,
                  inputs={str(p):sha256(p) for p in inputs},
                  implementation={name:sha256(cfg["root"]/"termmap"/name) for name in
                                  ["decision/layer.py","evaluation/metrics.py","evaluation/export.py","decision/coupling.py"]},
                  fit_script_sha256=sha256(Path(__file__)),
                  tables={name:sha256(term_dir(cfg)/name) for name in
                          ["priors_train.parquet","tag_priors_train.parquet","top_level_priors_train.parquet",
                           "seventh_priors_train.parquet","header_exceptions_train.parquet","cooccurrence_train.parquet",
                           "icd_concepts.parquet","snomed_concepts.parquet","snomed_links.parquet"]})
    marker=folder/(dataset+"_fit_complete.json")
    if matching_fit(marker,expected):
        print("Decision reuse",flow,dataset,"validated matching fit",flush=True)
        return json.loads((folder/(dataset+"_metrics.json")).read_text()),pd.read_parquet(folder/(dataset+"_details.parquet"))
    timing={};tick=time.perf_counter()
    items=joblib.load(inputs[0]);matrices=[features.matrix(x) for x in items]
    timing["dev_load_and_features_seconds"]=time.perf_counter()-tick
    thresholds=None
    if dataset=="dev":
        tick=time.perf_counter()
        probe=joblib.load(folder/"probe_scores.joblib");pm=[features.matrix(x) for x in probe]
        timing["probe_load_and_features_seconds"]=time.perf_counter()-tick
        tick=time.perf_counter()
        pools,conf,missing,model,choices,thresholds,without,auxiliary=cross_fit(items,matrices,features,probe,pm,coupling_guard)
        timing["nested_cv_and_full_dev_fit_seconds"]=time.perf_counter()-tick
        tick=time.perf_counter()
        pp,pc,pn,pa=apply_decision(model,probe,pm,features,return_aux=True)
        npools,nconf,nmissing=apply_decision(without["model"],probe,pm,features)
        check=coupling_guard(items,pools,without["pools"],probe,pp,npools,structures[3])
        timing["final_probe_application_and_guard_seconds"]=time.perf_counter()-tick
        write_json(folder/"coupling_check.json",dict(held_out_diagnostic=check,full_dev_fit=model["coupling_fit_check"],
            note="Each outer fold applies its keep/drop rule using only its fit groups plus probe; evaluated groups never choose their own parameters."))
        write_json(folder/"cv_manifest.json",model["cv_manifest"])
        joblib.dump(model,models/(flow+".joblib"));write_json(models/(flow+".json"),{k:model[k] for k in ["offset","w","tau","C","low_conf_threshold","features","objective"]}|dict(inner_C_choices=choices))
    else:
        tick=time.perf_counter();model=joblib.load(models/(flow+".joblib"))
        pools,conf,missing,auxiliary=apply_decision(model,items,matrices,features,return_aux=True)
        timing["frozen_decision_application_seconds"]=time.perf_counter()-tick
    tick=time.perf_counter()
    metrics,details=evaluate_flow(cfg,flow,dataset,items,pools,conf,missing,structures,thresholds,auxiliary,features)
    if dataset=="dev":evaluate_flow(cfg,flow,"probe",probe,pp,pc,pn,structures,auxiliary=pa,features=features)
    timing["metrics_bootstrap_export_validation_seconds"]=time.perf_counter()-tick
    timing["total_seconds"]=time.perf_counter()-started
    write_json(folder/(dataset+"_decision_timing.json"),timing)
    outputs=[models/(flow+".joblib")]
    for ds in (["dev","probe"] if dataset=="dev" else [dataset]):
        for suffix in ["metrics.json","details.parquet","decisions.joblib","validation.json","predictions.csv","diagnostics.jsonl","prediction_rows.parquet"]:
            outputs.append(folder/(ds+"_"+suffix))
        if not json.loads((folder/(ds+"_validation.json")).read_text())["passed"]:
            raise RuntimeError("Decision prediction validation failed for "+flow+"/"+ds)
    if dataset=="dev":outputs.extend(folder/name for name in ["probe_summary.json","cv_manifest.json","coupling_check.json"])
    write_json(marker,dict(provenance=expected,outputs={str(p):sha256(p) for p in outputs},timing=timing))
    return metrics,details


def keep_coupling(items,on,off,probe,pon,poff,freq,objective="diagnosis"):
    def acc(rows,pools,system,unseen=False):
        ids=[i for i,x in enumerate(rows) if x["row"]["system"]==system and
             (not unseen or freq.get((system,x["row"]["code"]),0)==0)]
        return float(np.mean([pools[i][0][0]==rows[i]["row"]["code"] for i in ids])*100) if ids else None
    from termmap.decision.layer import headline
    result=dict(with_min_acc=float(headline(items,[p[0][0] for p in on],objective)*100),
                without_min_acc=float(headline(items,[p[0][0] for p in off],objective)*100),systems={},objective=objective)
    keep=result["with_min_acc"]>=result["without_min_acc"]
    for system in ["ICD","SNOMED"]:
        values=dict(unseen_on=acc(items,on,system,True),unseen_off=acc(items,off,system,True),
                    probe_on=acc(probe,pon,system),probe_off=acc(probe,poff,system))
        result["systems"][system]=values
        if values["unseen_on"] is not None:keep=keep and values["unseen_on"]>=values["unseen_off"]-.5
        if values["probe_on"] is not None:keep=keep and values["probe_on"]>=values["probe_off"]-1
    return dict(result,keep=bool(keep))


def evaluate_flow(cfg,flow,dataset,items,pools,confidence,missing,structures,thresholds=None,auxiliary=None,features=None):
    icd,sn,links,freq=structures;rows=pd.DataFrame([x["row"] for x in items])
    timing_path=path_for(cfg,"run_dir")/flow/f"{dataset}_timing.json"
    timing=json.loads(timing_path.read_text()) if timing_path.exists() else {}
    # Ranking is fixed by the decision layer; its confidence drives risk/coverage.
    metric_pools=[[(c,float(confidence[i]) if j==0 else s) for j,(c,s) in enumerate(pool)] for i,pool in enumerate(pools)]
    metrics,details=evaluate(rows,metric_pools,freq,icd,sn,cfg["split"]["bootstrap_samples"],13,
                             [x.get("seconds",0)+x.get("rerank_seconds",0) for x in items],timing.get("peak_vram",0),links)
    metrics["latency"]["definition"]=timing.get("definition","Cached retrieval and reranking; final CLI reports uncached end-to-end latency.")
    details["prediction"]=[p[0][0] for p in pools];details["confidence"]=confidence;details["missing_probability"]=missing
    output=path_for(cfg,"run_dir")/flow
    model=joblib.load(path_for(cfg,"model_dir")/"decision"/(flow+".joblib"))
    details["low_conf"]=confidence<=(model["low_conf_threshold"] if thresholds is None else thresholds)
    metrics["low_conf_rate"]=float(details.low_conf.mean()*100)
    metrics["confidence_note"]="Nested grouped outer-fold predictions on dev; frozen full-dev fit on probe/test."
    # Preserve the retrieval ceiling independently of specificity-policy descent.
    metrics["original_pool_R64"]=float(np.mean([x["row"]["code"] in dict(x["pool"]) for x in items])*100)
    retrieval={}
    for system in ["ICD","SNOMED","SNOMED_Dx"]:
        ids=[i for i,x in enumerate(items) if x["row"]["system"]==("SNOMED" if system=="SNOMED_Dx" else system) and
             (system!="SNOMED_Dx" or x["row"].get("subconcept")=="DiagnosisCode")]
        retrieval[system]={}
        variants=sorted({x["row"]["variant"] for x in items if "variant" in x["row"]})
        for bucket in ["all","unseen","rare","common",*variants]:
            selected=[i for i in ids if bucket=="all" or (items[i]["row"].get("variant")==bucket if bucket in variants else details.iloc[i]["bucket"]==bucket)]
            retrieval[system][bucket]=dict(n=len(selected),**{f"R@{k}":float(np.mean([items[i]["row"]["code"] in [c for c,_ in items[i]["pool"][:k]] for i in selected])*100) if selected else None for k in [1,10,32,64]})
    metrics["retrieval"]=retrieval
    write_json(output/f"{dataset}_metrics.json",metrics)
    if dataset=="probe":write_json(output/"probe_summary.json",probe_summary(rows,pools))
    private=["entity","chunk_text","display","views","context","lexical_queries","dense_queries"]
    details.drop(columns=[c for c in private if c in details]).to_parquet(output/f"{dataset}_details.parquet",index=False)
    joblib.dump(dict(pools=pools,confidence=confidence,missing=missing,auxiliary=auxiliary),output/f"{dataset}_decisions.joblib")
    if auxiliary is not None:export_predictions(cfg,flow,dataset,items,pools,confidence,missing,
        model["low_conf_threshold"] if thresholds is None else thresholds,auxiliary,features)
    return metrics,details


def report(cfg,dataset,metrics,details):
    dest=path_for(cfg,"report_dir");deltas={}
    for a,b in DELTAS:
        if a in details and b in details:deltas[b+"-"+a]=paired_flow_delta(details[a],details[b],cfg["split"]["bootstrap_samples"],13)
    write_json(path_for(cfg,"run_dir")/"flows"/f"{dataset}_deltas.json",deltas)
    text=f"# Four-flow comparison: {dataset}\n\nPercentages. Dev is nested five-fold group cross-validation; test uses the frozen full-dev decision layers. Probe holds out an alias for each unseen dictionary code.\n\n"
    if cfg.get("generation"):
        lora="selected revised-input LoRA" if cfg.get("posttraining_revision") else "finalized original LoRA"
        text+=f"Generation: `{cfg['generation']}`. F00/F01 use new C1 pools; F10/F11 use the confirmed C2_v2 pools. F00/F10 use pretrained Qwen without an adapter; F01/F11 use the {lora}. The confirmation report records any per-system fallback. Each flow has an independently fitted decision layer.\n\n"
    text+="| Flow | ICD (95% CI) | SNOMED Dx (95% CI) | SNOMED all (95% CI) | PS all-type min proxy (95% CI) | Plan Dx min (95% CI) | Both correct (95% CI) |\n|---|---|---|---|---|---|---|\n"
    for flow in FLOWS:
        if flow not in metrics:text+=f"| {flow} | pending | pending | pending | pending | pending | pending |\n";continue
        m=metrics[flow];s=m["systems"]
        ps=m.get('headline_min_acc_all_types',min(s['ICD']['acc@1'],s['SNOMED']['acc@1']))
        text+=f"| {flow} | {s['ICD']['acc@1']:.3f} {s['ICD']['acc_ci']} | {s['SNOMED_Dx']['acc@1']:.3f} {s['SNOMED_Dx']['acc_ci']} | {s['SNOMED']['acc@1']:.3f} {s['SNOMED']['acc_ci']} | {ps:.3f} {m.get('headline_min_acc_all_types_ci','not recorded')} | {m['headline_min_acc']:.3f} {m['headline_min_acc_ci']} | {m['both_correct']['accuracy']} {m['both_correct']['ci']} |\n"
    text+="\nThe PS all-type minimum is a proxy on the supplied single-system labels, not an official two-gold hidden-evaluation score. The DiagnosisCode minimum is retained from plan §13.3. New decision fits and submission selection optimize the all-type minimum, as recorded in Review following the user's evaluation-metric clarification.\n"
    text+="\n| Flow / system | R@1 | R@10 | R@32 | R@64 | Selection given pool | min-acc 95% CI |\n|---|---:|---:|---:|---:|---:|---|\n"
    for flow in FLOWS:
        if flow not in metrics:continue
        for system in ["ICD","SNOMED_Dx","SNOMED"]:
            v=metrics[flow]["systems"][system]
            text+=f"| {flow}/{system} | {v['R@1']:.3f} | {v['R@10']:.3f} | {v['R@32']:.3f} | {v['R@64']:.3f} | {v['selection_given_pool']} | {metrics[flow].get('headline_min_acc_ci')} |\n"
    text+="\nFinal prediction accuracy above is separate from the incoming retrieval ceiling below. Specificity descent may add an unscored child after retrieval.\n\n| Flow / system / bucket | n | Incoming retrieval R@64 | Final accuracy |\n|---|---:|---:|---:|\n"
    for flow,m in metrics.items():
        for system in ["ICD","SNOMED_Dx","SNOMED"]:
            for bucket in ["unseen","rare","common"]:
                r=m["retrieval"][system][bucket];v=m["systems"][system]["buckets"][bucket]
                text+=f"| {flow}/{system}/{bucket} | {v['n']} | {r['R@64']} | {v['acc@1']} {v['acc_ci']} |\n"
    text+="\n| Delta | Bucket | Metric | points | Paired grouped 95% CI |\n|---|---|---|---:|---|\n"
    for delta,buckets in deltas.items():
        for bucket,values in buckets.items():
            for metric,value in values.items():text+=f"| {delta} | {bucket} | {metric} | {value['delta']:.3f} | {value['ci']} |\n"
    text+="\nProbe scores below use held-out dictionary aliases and noisy copies, with no chart context; these are distinct from dev chart accuracy.\n\n| Flow / system / variant | n | Final accuracy (95% CI) | Incoming retrieval R@64 |\n|---|---:|---|---:|\n"
    for flow in metrics:
        probe_file=path_for(cfg,"run_dir")/flow/"probe_summary.json"
        if not probe_file.exists():continue
        summaries=json.loads(probe_file.read_text());pm=json.loads(probe_file.with_name("probe_metrics.json").read_text())
        for system,variants in summaries.items():
            for variant,v in variants.items():
                recall=pm["retrieval"][system][variant]["R@64"]
                text+=f"| {flow}/{system}/{variant} | {v['n']} | {v['acc@1']:.3f} {v['acc_ci']} | {recall} |\n"
    text+="\nAll decision metrics, per-subconcept results, error classes, R@{1,10,32,64}, over/under-coding, header counts and risk/coverage are in the private run JSON files. Each input has both coding-system predictions for coupling; only its supplied gold system contributes an accuracy label. Unknown types are inferred from train-only SapBERT mention embeddings.\n"
    structure=path_for(cfg,"run_dir")/"priority_zero"/"probe_structure.json"
    if structure.exists():
        v=json.loads(structure.read_text())
        text+=f"\nThe fixed ICD probe contains {v['probe_headers']} headers among {v['icd_probe_concepts']} target codes. These codes are absent from all labels, so they cannot qualify for learned header exceptions; §9.2 prevents emitting them. All remain in the probe denominator. Incoming retrieval recall still counts them, while exact prediction accuracy has a policy ceiling of {100*v['probe_leaves']/v['icd_probe_concepts']:.3f}%. Dev has {v['dev_headers_outside_exception']} such ICD gold rows among {v['dev_icd_rows']}; none is removed.\n"
    provenance=path_for(cfg,"run_dir")/"priority_zero"/"alias_training_impact.json"
    if any(f in metrics for f in ["F01","F11"]) and provenance.exists():
        impact=json.loads(provenance.read_text());counts={e:v["union_affected_quizzes"] for e,v in impact.items()}
        if any(counts.values()):text+=f"\nTraining limitation for F01/F11: self-only abbreviation assignments affected selected evaluation-LoRA quizzes per epoch {counts}, out of 10,502 each. Three training rows' retrieval pools/features change under correct masking; the same two selected quizzes expose the aliases on candidate cards. No dev/test/probe text supplied them; held-out evaluation groups remain isolated. The running job was preserved as the user required. The final all-label rebuild masks these assignments. See training.md and the private provenance audits. F00/F10 have no LoRA training dependence.\n"
    target=dest/("flows.md" if dataset=="dev" else "flows_test.md");target.write_text(text)
    return deltas


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["dev","test","drill"],default="dev")
    p.add_argument("--flows",nargs="+",choices=FLOWS,default=FLOWS);p.add_argument("--no-freeze",action="store_true")
    p.add_argument("--report-only",action="store_true");args=p.parse_args()
    cfg=load_config();structures=metadata(cfg);features=Features(cfg,structures[0],structures[1]);metrics={};details={}
    coupling_guard=partial(keep_coupling,objective=cfg.get("decision",{}).get("objective","diagnosis"))
    models=path_for(cfg,"model_dir")/"decision";models.mkdir(parents=True,exist_ok=True)
    if args.dataset=="drill":
        items=joblib.load(path_for(cfg,"run_dir")/"F11"/"drill_scores.joblib")
        model=joblib.load(models/"F11.joblib");matrices=[features.matrix(x) for x in items]
        pools,conf,missing=apply_decision(model,items,matrices,features)
        assert all(x['row']['code'] not in dict(p) for x,p in zip(items,pools)),"Drill reintroduced removed code"
        result={}
        for system in ["ICD","SNOMED"]:
            ids=[i for i,x in enumerate(items) if x["row"]["system"]==system]
            near=[pools[i][0][0][:3]==items[i]["row"]["code"][:3] if system=="ICD" else pools[i][0][0] in structures[2].get(items[i]["row"]["code"],()) for i in ids]
            result[system]=dict(n=len(ids),same_category_or_near=float(np.mean(near)*100),detection_recall=float(np.mean(conf[ids]<=model["low_conf_threshold"])*100))
        write_json(path_for(cfg,"run_dir")/"drill"/"metrics.json",result);return
    for flow in args.flows:
        if args.report_only:
            folder=path_for(cfg,"run_dir")/flow
            metrics[flow]=json.loads((folder/f"{args.dataset}_metrics.json").read_text())
            details[flow]=pd.read_parquet(folder/f"{args.dataset}_details.parquet")
            continue
        metrics[flow],details[flow]=fit_one(cfg,flow,args.dataset,structures,features,coupling_guard)
        print(flow,args.dataset,"min-acc",metrics[flow]["headline_min_acc"],flush=True)
    # Independent flow fits can run concurrently; serialize their shared reports.
    # This descriptor is held until main returns (or this CLI process exits).
    report_dir=path_for(cfg,"run_dir")/"flows";report_dir.mkdir(parents=True,exist_ok=True)
    report_lock=(report_dir/"REPORT_LOCK").open("a")
    fcntl.flock(report_lock,fcntl.LOCK_EX)
    # Keep completed earlier arms visible when an independent worker finishes.
    # Validation is written last; partial exports are never merged as results.
    for flow in FLOWS:
        if flow in metrics:continue
        folder=path_for(cfg,"run_dir")/flow
        checks=[folder/(args.dataset+"_validation.json")]
        if args.dataset=="dev":checks.append(folder/"probe_validation.json")
        if all(p.exists() and json.loads(p.read_text()).get("passed") for p in checks):
            metrics[flow]=json.loads((folder/(args.dataset+"_metrics.json")).read_text())
            details[flow]=pd.read_parquet(folder/(args.dataset+"_details.parquet"))
    report(cfg,args.dataset,metrics,details)
    if set(args.flows)==set(FLOWS):
        from scripts.evaluation.diagnostics import build
        build(cfg,args.dataset)
    if args.dataset=="dev":
        probes={flow:pd.read_parquet(path_for(cfg,"run_dir")/flow/"probe_details.parquet") for flow in metrics}
        pdeltas={b+"-"+a:paired_flow_delta(probes[a],probes[b],cfg["split"]["bootstrap_samples"],13) for a,b in DELTAS if a in probes and b in probes}
        write_json(path_for(cfg,"run_dir")/"flows"/"probe_deltas.json",pdeltas)
        if args.no_freeze or set(args.flows)!=set(FLOWS):
            source=path_for(cfg,"run_dir")/cfg.get("evaluation",{}).get("adapted_source","C2_v2")
            checkpoint=json.loads((source/"metrics.json").read_text())
            save_checkpoint(cfg,"C2",checkpoint,json.loads((source/"probe_metrics.json").read_text()))
            return
        save_checkpoint(cfg,"C4",metrics["F11"],json.loads((path_for(cfg,"run_dir")/"F11"/"probe_summary.json").read_text()))
        confirmation=json.loads((path_for(cfg,"run_dir")/"C2_v2"/"confirmation.json").read_text())
        if not confirmation["ready"]:raise RuntimeError("V2 confirmation/fallback must complete before configuration freeze")
        adapted=confirmation["adapted"]
        # Independently confirmed systems can use v2 while a failing system
        # uses its original encoder. All four resulting flows remain eligible.
        eligible=FLOWS
        field="headline_min_acc_all_types" if cfg.get("decision",{}).get("objective")=="all_types" else "headline_min_acc"
        best=max(eligible,key=lambda f:(metrics[f][field],f=="F11"))
        frozen=dict(flow=best,dictionary_hash=dictionary_hash(cfg),retrieval=cfg["retrieval"],decision_settings=cfg.get("decision",{}),seed=13,
                    reranker=cfg.get("reranker",{}),decisions={f:sha256(models/(f+".joblib")) for f in FLOWS},splits_sha256=sha256(term_dir(cfg)/"splits.parquet"),adaptation_passed=all(adapted.values()),adapted_systems=adapted,
                    adapter_weights_sha256=sha256(path_for(cfg,"model_dir")/"reranker_lora"/"best"/"adapter_model.safetensors"))
        from termmap.retrieval.asymmetric import frozen_model_metadata
        frozen.update(frozen_model_metadata(cfg))
        from termmap.reranking.provenance import serving_metadata
        frozen.update(serving_metadata(cfg))
        write_json(path_for(cfg,"run_dir")/"frozen"/"CONFIG_FROZEN.json",frozen)
    else:
        (path_for(cfg,"run_dir")/"test").mkdir(parents=True,exist_ok=True)
        (path_for(cfg,"run_dir")/"test"/"STATUS").write_text("DONE\n")

if __name__=="__main__":main()
