"""Stage-by-stage aggregate error attribution from the exact scored traces."""
from collections import Counter
import json
import joblib
import pandas as pd
from termmap.common.io import load_config, path_for, write_json


def build(cfg, dataset="dev"):
    runs=path_for(cfg,"run_dir")
    stages={};misses={}
    for flow in ["F00","F10","F01","F11"]:
        folder=runs/flow
        if not (folder/f"{dataset}_details.parquet").exists():continue
        details=pd.read_parquet(folder/f"{dataset}_details.parquet").set_index("row_id")
        items=joblib.load(folder/f"{dataset}_scores.joblib")
        counts=Counter();miss=Counter()
        for item in items:
            row=item["row"];system=row["system"];gold=row["code"]
            prefix=system+"/"
            incoming=gold in dict(item["pool"])
            raw=min(item["reranker_scores"],key=lambda c:(-round(item["reranker_scores"][c],5),c))==gold
            final=bool(details.loc[row["row_id"],"correct"])
            lexical_first=item["pool"][0][0]==gold
            for key, condition in [("rows",True),("gold_in_pool",incoming),("retrieval_top1_correct",lexical_first),
                                   ("raw_reranker_correct",raw),("final_correct",final),
                                   ("reranker_demoted_correct_retrieval",lexical_first and not raw),
                                   ("reranker_rescued_retrieval",not lexical_first and raw),
                                   ("decision_rescued_reranker",not raw and final),
                                   ("decision_lost_reranker_hit",raw and not final),
                                   ("final_error_gold_in_pool",not final and incoming),
                                   ("final_error_gold_absent",not final and not incoming),
                                   ("missing_context",bool(row.get("context",{}).get("no_context",False)))]:
                if condition:counts[prefix+key]+=1
            if incoming:continue
            enabled=cfg["retrieval"].get("enabled_by_system",{}).get(system,cfg["retrieval"]["enabled_views"])
            if not row.get("valid_gold",True):cause="label outside dictionary"
            elif any(v in enabled and row["views"][v]!=row["views"]["V1"] for v in ["V3","V4"]):cause="changed noise view"
            elif row["views"]["V2"]!=row["views"]["V1"]:cause="abbreviation ambiguity"
            elif system=="ICD" and any(c[:3]==gold[:3] for c,_ in item["pool"]):cause="specificity within category"
            else:cause="semantic distance or alias gap"
            miss[prefix+cause]+=1
        stages[flow]=dict(counts);misses[flow]=dict(miss)
    write_json(runs/"flows"/(dataset+"_stage_diagnostics.json"),dict(stages=stages,miss_causes=misses))
    text=f"# Flow stage diagnostics: {dataset}\n\nThese counts reconstruct each complete flow's exact scored traces. Raw reranker accuracy is an intermediate measurement; final accuracy includes the independently fitted decision layer. Auxiliary predictions do not add gold labels.\n\n"
    text+="| Flow / system | Rows | Gold in incoming pool | Retrieval top-1 correct | Raw reranker correct | Final correct | Reranker demotions / rescues | Decision rescues / losses | Final errors, gold present / absent | Missing context |\n|---|---:|---:|---:|---:|---:|---|---|---|---:|\n"
    for flow,v in stages.items():
        for s in ["ICD","SNOMED"]:
            get=lambda key:v.get(s+"/"+key,0)
            text+=f"| {flow}/{s} | {get('rows')} | {get('gold_in_pool')} | {get('retrieval_top1_correct')} | {get('raw_reranker_correct')} | {get('final_correct')} | {get('reranker_demoted_correct_retrieval')} / {get('reranker_rescued_retrieval')} | {get('decision_rescued_reranker')} / {get('decision_lost_reranker_hit')} | {get('final_error_gold_in_pool')} / {get('final_error_gold_absent')} | {get('missing_context')} |\n"
    text+="\nR@64 can be high while exact accuracy remains lower: gold availability does not identify which candidate wins. Demotions/rescues show where ranking changes the answer; decision rescues/losses show the net effect of fusion, specificity and coupling. These are observational counts, not additional component experiments.\n\n"
    text+="| Flow | System / heuristic miss cause | Count |\n|---|---|---:|\n"
    for flow,values in misses.items():
        for cause,count in sorted(values.items()):text+=f"| {flow} | {cause} | {count} |\n"
    text+="\nMiss causes are ordered heuristics for triage. A changed normalization view is not proof that normalization caused a miss, and the remaining semantic errors are not automatically annotation errors. Private row-level traces and two-code diagnostics remain in the ignored run artifacts. No extra training or prompt tuning was used for this report.\n"
    filename="miss_causes.md" if dataset=="dev" else "miss_causes_test.md"
    (path_for(cfg,"report_dir")/filename).write_text(text)
    return dict(stages=stages,miss_causes=misses)


if __name__=="__main__":build(load_config())
