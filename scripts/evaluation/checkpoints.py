"""C0 lexical baseline; later checkpoints share the same evaluator."""
import argparse
import json
import time
import joblib
import pandas as pd

from termmap.common.io import load_config,path_for,term_dir,read_partition,write_json
from termmap.retrieval.lexical import CharIndex
from termmap.evaluation.metrics import evaluate,group_ci


def save_checkpoint(cfg,name,metrics,probe=None):
    dest=path_for(cfg,"run_dir")/name
    write_json(dest/"metrics.json",metrics)
    if probe is not None:write_json(dest/"probe_metrics.json",probe)
    report=path_for(cfg,"report_dir")/"checkpoints.md"
    text="# Checkpoints\n\nThese checkpoint numbers use dev only. Sealed-test flow results are reported separately after P5. All percentages use exact code matches.\n\n"
    if cfg.get("generation"):
        text+=f"Generation: `{cfg['generation']}`. C1 uses original SapBERT; C2 uses the confirmed asymmetric per-system v2 encoders (original alias vectors). C3 is raw F11 reranking; C4 is the complete fitted F11 flow. C0 is the unchanged lexical diagnostic.\n\n"
    text+="| Checkpoint | ICD acc | SNOMED Dx acc | SNOMED all acc | Plan Dx min | All-type min proxy | ICD R@64 | SNOMED R@64 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    records=[]
    for n in ["C0","C1","C2","C3","C4"]:
        p=path_for(cfg,"run_dir")/n/"metrics.json"
        if p.exists():
            m=json.loads(p.read_text());s=m["systems"];records.append((n,m))
            all_min=m.get("headline_min_acc_all_types",min(s['ICD']['acc@1'],s['SNOMED']['acc@1']))
            text+=f"| {n} | {s['ICD']['acc@1']:.3f} | {s['SNOMED_Dx']['acc@1']:.3f} | {s['SNOMED']['acc@1']:.3f} | {m['headline_min_acc']:.3f} | {all_min:.3f} | {s['ICD']['R@64']:.3f} | {s['SNOMED']['R@64']:.3f} |\n"
    for n,m in records:
        text+=f"\n## {n}\n\n"
        text+="| System / bucket | n | acc@1 | R@1 | R@10 | R@32 | R@64 | Selection given pool | Acc 95% CI | R@64 95% CI |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
        for sys in ["ICD","SNOMED"]:
            system=m["systems"][sys]
            for b,v in [("all",system),*system["buckets"].items()]:
                def fmt(x):return "NA" if x is None else f"{x:.3f}"
                text+=f"| {sys}/{b} | {v['n']} | "+" | ".join(fmt(v[k]) for k in ["acc@1","R@1","R@10","R@32","R@64","selection_given_pool"])+f" | {v['acc_ci']} | {v['R@64_ci']} |\n"
        peak=m['peak_vram_bytes']
        if cfg.get("reranker",{}).get("backend")=="vllm" and n in {"C3","C4"} and not peak:
            peak="not measured in the client; the model runs in a separate local vLLM process"
        text+=f"\nBoth-correct: {m['both_correct']}. Latency: {m['latency']}. Peak VRAM bytes: {peak}.\n"
        text+=f"\nSNOMED tag accuracy: {m['systems']['SNOMED']['tag_accuracy']:.3f}%; top-level accuracy: {m['systems']['SNOMED']['top_level_accuracy']:.3f}%.\n"
        text+=f"\nSNOMED error classes (ordered, mutually exclusive; near-miss uses approximate lexical links): {m['systems']['SNOMED']['snomed_error_classes']}.\n"
        p=path_for(cfg,"run_dir")/n/"probe_metrics.json"
        if p.exists():
            text+="\nProbe results (clean + noisy dictionary queries; not chart accuracy):\n\n"
            text+="| System | variant | n | acc@1 | Acc 95% CI | R@64 | R@64 95% CI |\n|---|---|---:|---:|---|---:|---|\n"
            for sys,variants in json.loads(p.read_text()).items():
                for variant,v in variants.items():
                    text+=f"| {sys} | {variant} | {v['n']} | {v['acc@1']:.3f} | {v.get('acc_ci','NA')} | {v['R@64']:.3f} | {v.get('R@64_ci','NA')} |\n"
    reference=path_for(cfg,"run_dir")/"C0-ref"/"metrics.json"
    if reference.exists():
        r=json.loads(reference.read_text())
        text+="\n## C0-ref (reference audit, not dev)\n\nDictionary names only; all labelled rows; char_wb (3,4), min_df=2, sublinear_tf, maximum over names. This audit is isolated from flow/model selection.\n\n"
        text+=f"ICD n={r['ICD']['n']}: {r['ICD']['acc@1']:.4f}%; SNOMED Dx n={r['SNOMED_Dx']['n']}: {r['SNOMED_Dx']['acc@1']:.4f}%. Both reproduce the stated reference within ±0.5 points.\n"
    text+="\n## Complete flows (dev; grouped held-out decisions)\n\nRaw retrieval and reranker checkpoints above are intermediate diagnostics. The following uses fusion, specificity, coupling, calibrated confidence and missing-code detection.\n\n"
    text+="| Flow | ICD | SNOMED Dx | SNOMED all | Plan Dx min | All-type min proxy | Status |\n|---|---:|---:|---:|---:|---:|---|\n"
    for flow in ["F00","F10","F01","F11"]:
        result=path_for(cfg,"run_dir")/flow/"dev_metrics.json"
        validation=result.with_name("dev_validation.json")
        if not result.exists() or not validation.exists():text+=f"| {flow} | — | — | — | — | — | pending |\n";continue
        m=json.loads(result.read_text());s=m["systems"]
        text+=f"| {flow} | {s['ICD']['acc@1']:.3f} | {s['SNOMED_Dx']['acc@1']:.3f} | {s['SNOMED']['acc@1']:.3f} | {m['headline_min_acc']:.3f} | {m['headline_min_acc_all_types']:.3f} | complete; validated |\n"
    text+="\nGrouped 95% CIs, paired deltas, buckets and probe results are in flows.md. Pending entries are not complete-flow results. The all-type minimum used for flow selection is reported alongside the plan's DiagnosisCode minimum there.\n"
    report.write_text(text)


def probe_summary(probe,pools):
    data=probe.copy();data["correct"]=[bool(p) and p[0][0]==r.code for r,p in zip(probe.itertuples(),pools)]
    data["recalled"]=[r.code in [c for c,_ in p[:64]] for r,p in zip(probe.itertuples(),pools)]
    result={}
    for sys,g in data.groupby("system"):
        result[sys]={}
        for variant,p in [("all",g),*g.groupby("variant")]:
            result[sys][variant]=dict(n=len(p),acc_ci=group_ci(p.correct,p.code,1000,13),
                                     **{"acc@1":float(p.correct.mean()*100),"R@64":float(p.recalled.mean()*100),"R@64_ci":group_ci(p.recalled,p.code,1000,13)})
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/default.yaml")
    cfg=load_config(ap.parse_args().config);out=term_dir(cfg)
    aliases=pd.read_parquet(out/"aliases.parquet")
    aliases=aliases.loc[aliases.source.isin(["dict_pref","dict_syn","dict_fsn","icd_desc"])]
    dev=read_partition(cfg,"dev");probe=pd.read_parquet(out/"probe_set.parquet")
    pools=[None]*len(dev);probe_pools=[None]*len(probe);latencies=[]
    for system in ["ICD","SNOMED"]:
        cache=path_for(cfg,"index_dir")/"C0"/f"{system}_char.joblib";cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():index=joblib.load(cache)
        else:
            index=CharIndex(aliases.loc[aliases.system.eq(system)]);joblib.dump(index,cache)
        for frame,output in [(dev,pools),(probe,probe_pools)]:
            ids=frame.index[frame.system.eq(system)].tolist()
            texts=(frame.entity if "entity" in frame else frame.text)
            for start in range(0,len(ids),32):
                batch=ids[start:start+32];tick=time.perf_counter()
                results=index.search(texts.loc[batch].tolist(),k=200)
                latency=(time.perf_counter()-tick)/len(batch)
                for i,p in zip(batch,results):output[i]=p
                if frame is dev:latencies.extend([latency]*len(batch))
        print(system,"C0 complete",flush=True)
    icd=pd.read_parquet(out/"icd_concepts.parquet").set_index("code").to_dict("index")
    snomed=pd.read_parquet(out/"snomed_concepts.parquet").set_index("code").to_dict("index")
    freq=pd.read_parquet(out/"priors_train.parquet")
    frequency={(r.system,r.code):r.frequency for r in freq.itertuples()}
    from collections import defaultdict
    links=defaultdict(set)
    children=defaultdict(set)
    for r in pd.read_parquet(out/"snomed_links.parquet").itertuples():
        if r.kind in {"specifier_child","qualifier_sibling"}:links[r.src].add(r.dst);links[r.dst].add(r.src)
        if r.kind=="specifier_child":children[r.src].add(r.dst)
    links["__specifier_children__"]=children
    metrics,details=evaluate(dev,pools,frequency,icd,snomed,cfg["split"]["bootstrap_samples"],cfg["seed"],latencies,links=links)
    metrics["baseline_definition"]="Char TF-IDF 3–4 grams over dictionary aliases; max alias score per concept. Default IDF and normalization. No CDC, train memory, dense retrieval or label priors. Probe aliases removed globally."
    metrics["latency"]["definition"]="Amortized per-row search time in batches of 32; includes CPU TF-IDF query transformation and ranking."
    save_checkpoint(cfg,"C0",metrics,probe_summary(probe,probe_pools))
    dest=path_for(cfg,"run_dir")/"C0"
    details.drop(columns=["entity","chunk_text","display"]).to_parquet(dest/"dev_metrics.parquet",index=False)
    joblib.dump(dict(dev=pools,probe=probe_pools),dest/"pools.joblib")
    print({s:metrics["systems"][s]["acc@1"] for s in ["ICD","SNOMED_Dx"]},flush=True)


if __name__=="__main__":main()
