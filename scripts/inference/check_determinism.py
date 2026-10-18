"""Three separate cold processes, with genuine uncached reranker computation."""
import argparse
import json
import subprocess
import sys
import pandas as pd
from termmap.common.io import load_config,path_for,read_partition,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");a=p.parse_args();cfg=load_config();dest=path_for(cfg,"run_dir")/"determinism";dest.mkdir(parents=True,exist_ok=True)
    frame=read_partition(cfg,"dev").sample(n=12,random_state=13)
    inputs=pd.DataFrame(dict(chunk_id=["d:"+str(r) for r in frame.row_id],clinical_entity=frame.entity.tolist(),chunk_text=frame.chunk_text.tolist(),subconcept=frame.subconcept.tolist()))
    inputs.loc[inputs.index%2==0,"subconcept"]=""
    # Also exercise new-query encoding rather than only the final all-label cache.
    public=pd.DataFrame([
        dict(chunk_id="s:1",clinical_entity="mild intermittent left ankle discomfort without trauma",chunk_text="Assessment: mild intermittent left ankle discomfort without trauma.",subconcept=""),
        dict(chunk_id="s:2",clinical_entity="chronickidney disease stag 3",chunk_text="",subconcept="DiagnosisCode"),
        dict(chunk_id="s:3",clinical_entity="MS",chunk_text="Neurology follow-up: MS with demyelinating lesions.",subconcept="DiagnosisCode"),
        dict(chunk_id="s:4",clinical_entity="RA",chunk_text="Rheumatology assessment: RA involving several joints.",subconcept="DiagnosisCode"),
        dict(chunk_id="s:5",clinical_entity="PID",chunk_text="Gynecology assessment: PID with pelvic pain.",subconcept="DiagnosisCode"),
        dict(chunk_id="s:6",clinical_entity="type 2 diabetes mellitus",chunk_text="Assessment: type 2 diabetes mellitus without complications.",subconcept="DiagnosisCode"),
        dict(chunk_id="s:7",clinical_entity="diabetic neuropathy",chunk_text="Assessment: type 2 diabetes with diabetic neuropathy.",subconcept="DiagnosisCode"),
        dict(chunk_id="s:8",clinical_entity="no chest pain",chunk_text="Review of systems: no chest pain.",subconcept=""),
        dict(chunk_id="s:9",clinical_entity="history of stroke",chunk_text="Past medical history: history of stroke.",subconcept=""),
        dict(chunk_id="s:10",clinical_entity="right knee pain",chunk_text="",subconcept=""),
    ])
    inputs=pd.concat([inputs,public],ignore_index=True)
    if a.smoke:inputs=public.head(1)
    source=dest/"input.csv";inputs.to_csv(source,index=False)
    for name,rows in [("cold_a",inputs),("cold_b",inputs),("shuffled",inputs.sample(frac=1,random_state=13))]:
        folder=dest/name;folder.mkdir(parents=True,exist_ok=True);input_file=folder/"input.csv";rows.to_csv(input_file,index=False)
        cmd=[sys.executable,"-u","-m","scripts.inference.predict",str(input_file),str(folder/"predictions.csv"),"--run-id","determinism_"+name]
        subprocess.run(cmd,check=True)
    outputs=[pd.read_csv(dest/name/"predictions.csv",dtype=str,keep_default_na=False).sort_values("chunk_id").reset_index(drop=True) for name in ["cold_a","cold_b","shuffled"]]
    pd.testing.assert_frame_equal(outputs[0],outputs[1]);pd.testing.assert_frame_equal(outputs[0],outputs[2])
    diagnostics=[]
    for name in ["cold_a","cold_b","shuffled"]:
        rows=[json.loads(line) for line in (dest/name/"diagnostics.jsonl").read_text().splitlines()]
        for row in rows:row.pop("row",None)
        diagnostics.append(sorted(rows,key=lambda r:(r["chunk_id"],r["system"])))
    assert diagnostics[0]==diagnostics[1]==diagnostics[2],"Diagnostics differ after reordering"
    write_json(dest/"results.json",dict(n_rows=len(inputs),cold_runs=2,shuffled_run=1,prediction_diff_empty=True,diagnostic_diff_empty=True,reranker_score_cache=False,validator="passed",seed=13,
                                     hard_cases=["MS","RA","PID","joined and truncated words","diabetes with/without neuropathy","negation","history","laterality","missing context","missing type"]))
    print("Determinism passed: two cold processes plus shuffled input; predictions and diagnostics identical",flush=True)

if __name__=="__main__":main()
