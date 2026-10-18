"""Export and validate the same held-out decisions used by the metric tables."""
import json
import numpy as np
import pandas as pd
from termmap.common.io import path_for,write_json,sha256
from termmap.evaluation.metrics import validate_predictions


def export_predictions(cfg,flow,dataset,items,pools,conf,missing,thresholds,auxiliary,features):
    if auxiliary is None or any(x is None for x in auxiliary):raise ValueError("Both coding-system decisions are required for export")
    output=path_for(cfg,"run_dir")/flow;output.mkdir(parents=True,exist_ok=True)
    records=[];inputs=[];diagnostics=[];mapping=[]
    limits=np.broadcast_to(thresholds,len(items))
    for i,(item,pool,other) in enumerate(zip(items,pools,auxiliary)):
        row=item["row"];chunk_id=str(row.get("chunk_id",row["row_id"]))
        source=dict(chunk_id=chunk_id,clinical_entity=row["entity"]);inputs.append(source)
        prediction=dict(source);mapping.append(dict(row_id=row["row_id"],position=i,gold_system=row["system"]))
        results=[(item,pool,float(conf[i]),float(missing[i]),False),
                 (item["other_system"],other["pool"],other["confidence"],other["missing"],True)]
        for scored,ranking,probability,absent,aux in results:
            system=scored["row"]["system"];code=ranking[0][0]
            prediction["icd10cm_code" if system=="ICD" else "snomed_code"]=code
            diagnostics.append(dict(row_id=row["row_id"],chunk_id=chunk_id,system=system,is_auxiliary=aux,top3=ranking[:3],
                confidence=probability,low_conf=bool(probability<=limits[i]),gold_not_in_pool_probability=absent,
                views=scored["row"].get("views",{}),cue_flags=scored["row"].get("context",{}).get("cues",{}),
                winner_channels=[name for name,values in scored.get("channels",{}).items() if code in dict(values)],
                specificity_descended=code not in dict(scored["pool"])))
        records.append(prediction)
    frame=pd.DataFrame(records,columns=["chunk_id","clinical_entity","icd10cm_code","snomed_code"])
    if dataset=="dev":
        manifest=json.loads((output/"cv_manifest.json").read_text())
        for item in items:
            group=item["row"]["group_id"];folds=[f for f in manifest if group in f["evaluated_groups"]]
            if len(folds)!=1 or group in folds[0]["fit_groups"]:raise ValueError("Evaluated group used in its own decision fit")
    validate_predictions(pd.DataFrame(inputs),frame,{"ICD":set(features.icd),"SNOMED":set(features.sn)},features.icd,features.exceptions)
    prediction_file=output/f"{dataset}_predictions.csv";frame.to_csv(prediction_file,index=False)
    (output/f"{dataset}_diagnostics.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in diagnostics))
    pd.DataFrame(mapping).to_parquet(output/f"{dataset}_prediction_rows.parquet",index=False)
    write_json(output/f"{dataset}_validation.json",dict(passed=True,rows=len(items),codes=2*len(items),
        strings=True,dictionary_membership=True,legal_headers=True,row_count_and_order=True,
        auxiliary_labels_used=False,held_out_group_isolation=dataset=="dev",predictions_sha256=sha256(prediction_file),
        decision_protocol="outer held-out groups" if dataset=="dev" else "frozen full-dev fit"))
