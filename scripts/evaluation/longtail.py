"""Aggregate-only report; worked examples identify public codes, not chart text."""
import json
import numpy as np
import pandas as pd
from termmap.common.io import load_config,path_for,term_dir

def main():
    cfg=load_config();runs=path_for(cfg,"run_dir");text="# Long-tail results\n\nBuckets count train-split gold-code occurrences: unseen 0, rare 1–5, common >5. Probe queries are held-out dictionary names and their noisy copies, with no chart context.\n\n"
    text+="| Dataset / run | System / bucket | n | acc@1 | Acc 95% CI | R@64 | Selection given pool | Category / tag accuracy |\n|---|---|---:|---:|---|---:|---:|---:|\n"
    for dataset in ["dev","test"]:
        for flow in ["F00","F10","F01","F11"]:
            path=runs/flow/f"{dataset}_metrics.json"
            if not path.exists():continue
            m=json.loads(path.read_text())
            for system in ["ICD","SNOMED"]:
                for bucket,v in m["systems"][system]["buckets"].items():
                    text+=f"| {dataset}/{flow} | {system}/{bucket} | {v['n']} | {v['acc@1']} | {v['acc_ci']} | {v['R@64']} | {v['selection_given_pool']} | {v['category_accuracy'] if system=='ICD' else v['tag_accuracy']} |\n"
    text+="\n| Flow / system | Probe variant | n | acc@1 | Acc 95% CI | R@64 | R@64 95% CI |\n|---|---|---:|---:|---|---:|---|\n"
    for flow in ["F00","F10","F01","F11"]:
        p=runs/flow/"probe_summary.json"
        if p.exists():
            for system,variants in json.loads(p.read_text()).items():
                for variant,v in variants.items():text+=f"| {flow}/{system} | {variant} | {v['n']} | {v['acc@1']:.3f} | {v.get('acc_ci','NA')} | {v['R@64']:.3f} | {v.get('R@64_ci','NA')} |\n"
    delta=runs/"flows"/"probe_deltas.json"
    if delta.exists():
        text+="\n| Probe delta | System | points | Paired code-group 95% CI |\n|---|---|---:|---|\n"
        for name,buckets in json.loads(delta.read_text()).items():
            for system,v in buckets["all"].items():text+=f"| {name} | {system} | {v['delta']:.3f} | {v['ci']} |\n"
    text+="\n| Dataset / flow / system | Over-coding % | Under-coding % | Headers outside exception | SNOMED error classes |\n|---|---:|---:|---:|---|\n"
    for dataset in ["dev","test"]:
        for flow in ["F00","F10","F01","F11"]:
            p=runs/flow/f"{dataset}_metrics.json"
            if p.exists():
                for system in ["ICD","SNOMED"]:
                    v=json.loads(p.read_text())["systems"][system]
                    text+=f"| {dataset}/{flow}/{system} | {v['over_coding_rate']} | {v['under_coding_rate']} | {v['headers_outside_exception']} | {v['snomed_error_classes'] if system=='SNOMED' else 'NA'} |\n"
    details=runs/"F11"/"dev_details.parquet"
    if details.exists():
        frame=pd.read_parquet(details);text+="\n| F11 dev bucket | Not retrieved | Specificity errors | Other wrong picks |\n|---|---:|---:|---:|\n"
        for bucket,g in frame.groupby("bucket"):
            notretrieved=~g["R@64"];specific=(g.over_coding|g.under_coding)&~g.correct&~notretrieved
            other=~g.correct&~notretrieved&~specific
            text+=f"| {bucket} | {int(notretrieved.sum())} | {int(specific.sum())} | {int(other.sum())} |\n"
        text+="\nCoder-convention attribution requires chart review; the remaining wrong picks are not automatically declared annotation errors.\n\n"
        icd=pd.read_parquet(term_dir(cfg)/"icd_concepts.parquet").set_index("code").preferred.to_dict();sn=pd.read_parquet(term_dir(cfg)/"snomed_concepts.parquet").set_index("code").preferred.to_dict()
        rare=frame.loc[frame.bucket.isin(["unseen","rare"])].sort_values("row_id")
        text+="Public-code worked cases (chart wording is kept in private artifacts):\n\n"
        for success,n in [(True,3),(False,2)]:
            for _,r in rare.loc[rare.correct.eq(success)].head(n).iterrows():
                names=icd if r.system=="ICD" else sn
                text+=f"- {'Success' if success else 'Failure'}: {r.system} gold {r.code} ({names.get(r.code,'missing from dictionary')}), prediction {r.prediction} ({names.get(r.prediction,'')}), bucket {r.bucket}; {'gold was retrieved and selected' if success else 'gold absent from the pool' if not r['R@64'] else 'gold retrieved but a different code selected'}.\n"
    drill=runs/"drill"/"metrics.json"
    if drill.exists():text+="\nMissing-code drill: "+json.dumps(json.loads(drill.read_text()),sort_keys=True)+". Normal false-alarm rates are recorded as low_conf_rate in each flow's dev metrics.\n"
    text+="\nThe engine retrieves against the full supplied dictionaries and scores descriptions rather than classifying into a fixed list of code IDs. Uniform dictionary alias quizzes and dictionary-wide contrastive pairs cover codes with no labelled examples. A dictionary update requires rebuilding the indexes; the lexical SNOMED links remain approximate, and real chart unseen-code accuracy is the stronger stress test.\n"
    (path_for(cfg,"report_dir")/"longtail.md").write_text(text)

if __name__=="__main__":main()
