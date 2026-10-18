"""Aggregate P0 audit and seeded dictionary-link sample (no chart excerpts)."""
import argparse
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import simple_icd_10_cm as simple

from termmap.common.io import load_config,term_dir,path_for,write_json,sha256,read_strings,data_path


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/default.yaml")
    cfg=load_config(ap.parse_args().config);out=term_dir(cfg)
    icd=pd.read_parquet(out/"icd_concepts.parquet");sn=pd.read_parquet(out/"snomed_concepts.parquet")
    checks={name:json.loads((out/f"{name}_checks.json").read_text()) for name in ["icd","snomed","split","link"]}
    compared=mismatched=unknown=0;examples=[]
    supplied=set(icd.code)
    for r in icd.itertuples():
        if not simple.is_valid_item(r.code):unknown+=1;continue
        public=next((p for p in simple.get_ancestors(r.code) if p in supplied and p!=r.code),"")
        compared+=1
        if public!=r.parent:
            mismatched+=1
            if len(examples)<10:examples.append([r.code,r.parent,public])
    checks["simple_icd_crosscheck"]={"compared":compared,"parent_mismatches":mismatched,"not_in_package":unknown,"examples":examples}
    gaps=0;fragmented=0;multi=0
    import re
    trailing=Counter()
    for r in sn.itertuples():
        positions=sorted(map(int,r.source_rows));n=sum(b>a+1 for a,b in zip(positions,positions[1:]))
        gaps+=n;fragmented+=n>0
        tails={m.group(1) for text in r.descriptions if (m:=re.search(r"\(([^()]*)\)\s*$",text))}
        multi+=len(tails)>1
        trailing.update(tails)
    checks["snomed_storage"]={"fragmented_concepts":fragmented,"reopened_runs":gaps,
                              "multiple_trailing_parenthesis_strings":multi,"distinct_trailing_strings":len(trailing)}
    allergy=read_strings(data_path(cfg,"snomed_file"),"Allergy reaction labels")
    checks["allergy_labels"]={"rows":len(allergy),"all_present_in_main_concept_table":bool(allergy.snomed_code.isin(sn.code).all())}
    write_json(out/"terminology_audit.json",checks)
    sample_path=out/"link_audit_sample.json"
    if not sample_path.exists():
        file=pq.ParquetFile(out/"snomed_links.parquet")
        selected=set(np.random.default_rng(cfg["seed"]).choice(file.metadata.num_rows,100,replace=False).tolist())
        names=sn.set_index("code").preferred.to_dict();sample=[];offset=0
        for batch in file.iter_batches(batch_size=100000):
            for i in sorted(x-offset for x in selected if offset<=x<offset+len(batch)):
                row=batch.slice(i,1).to_pylist()[0]
                row.update(src_name=names[row["src"]],dst_name=names[row["dst"]],judgment="pending",reason="")
                sample.append(row)
            offset+=len(batch)
        write_json(sample_path,sample)
    sample=json.loads(sample_path.read_text())
    table="| Check | Result |\n|---|---|\n"
    for section,values in checks.items():
        for key,value in values.items():table+=f"| {section}: {key} | {value} |\n"
    text="# Terminology audit\n\n"+table
    text+="\nThe ICD ProcedureCode row (Z99.3, paired with SNOMED 105503008) is preserved. All ICD rows count in ICD metrics. Supplied subconcepts drive stratification and train-only priors. L6.0 is excluded from positive training pairs, retained in evaluation denominators, and reported as label noise.\n"
    text+="\nSNOMED top-level and tag are soft features only. Product uses the Pharmaceutical / biologic product primary domain; physical object uses Physical object. Metadata is the guide's explicit Model Component domain tag. The four unknowns are retained. The official two levels do not imply an IS-A graph. Lexical links are approximation-based diagnostics and candidate additions.\n"
    text+="\nOfficial reference: [SNOMED Editorial Guide, Semantic Tag](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-editorial-guide/readme/authoring/general-naming-conventions/descriptions/fully-specified-name/index). The education URL in the prompt returned HTTP 403; the official specifications route supplies the same table. Model/resource files, hashes and dates are in external/SOURCES.md.\n"
    text+="\nSplits were saved once, using 20 stratified grouped folds allocated 14/3/3. Singleton subconcepts cannot occur in all partitions; no row is relabelled or duplicated. Near-duplicate union-find uses character TF-IDF cosine ≥0.9. Test content is not read after construction.\n"
    text+="\nCounts with different definitions (unique fragmented concepts versus reopened runs, and all trailing parentheses versus official tags) are reported separately; arbitrary trailing text is never admitted as an official tag.\n"
    judgments=Counter(r["judgment"] for r in sample)
    text+=f"\nManual dictionary-link audit (100 links, seed 13): {dict(judgments)}.\n"
    if judgments.get("pending",0)==0:
        accepted=judgments.get("correct",0)
        text+=f"\nPrecision: {accepted/len(sample)*100:.1f}%. Every decision is recorded in the local artifact link_audit_sample.json; only dictionary concepts were examined.\n"
        text+="\n| src | dst | kind | Judgment | Reason |\n|---|---|---|---|---|\n"
        for r in sample:text+=f"| {r['src']} | {r['dst']} | {r['kind']} | {r['judgment']} | {r['reason']} |\n"
    (path_for(cfg,"report_dir")/"terminology.md").write_text(text)
    print(json.dumps(checks["simple_icd_crosscheck"]))


if __name__=="__main__":main()
