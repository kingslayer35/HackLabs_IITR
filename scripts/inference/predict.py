"""Cold-load final artifacts and emit two valid string codes per input row."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from termmap.common.io import load_config,path_for,term_dir,write_json,seed_everything,verify_frozen
from termmap.retrieval.online import OnlineRetriever
from termmap.reranking.model import Reranker
from termmap.decision.layer import Features,apply_decision
from termmap.evaluation.metrics import validate_predictions
from termmap.common.release import verify_release

def main():
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("output");p.add_argument("--smoke",action="store_true");p.add_argument("--run-id",default="predict");a=p.parse_args()
    cfg=load_config();seed_everything(13,True);data=pd.read_csv(a.input,dtype=str,keep_default_na=False)
    destination=Path(a.output).resolve()
    if destination==Path(a.input).resolve() or path_for(cfg,"data_dir").resolve() in destination.parents:raise ValueError("Output must not overwrite input or the read-only data directory")
    mention="clinical_entity" if "clinical_entity" in data else "entity"
    if mention not in data:raise ValueError("Input needs entity or clinical_entity")
    if a.smoke:data=data.head(1)
    frozen=verify_release(cfg);flow=frozen["flow"]
    profile=cfg["deployment"]["profile"]
    retriever=OnlineRetriever(cfg,flow);adapter=path_for(cfg,"model_dir")/cfg["deployment"]["adapter_directory"] if flow[-1]=="1" else None
    cfg.setdefault("reranker",{})["cache_enabled"]=False
    terminology=dict(icd=retriever.icd,sn=retriever.sn,aliases=retriever.aliases)
    if cfg.get("reranker",{}).get("backend")=="vllm":
        from termmap.reranking.hosted import HostedFinalRanker
        ranker=HostedFinalRanker(cfg,adapter,terminology=terminology)
    else:
        ranker=Reranker(cfg,adapter,profile=profile,terminology=terminology)
    features=Features(cfg,retriever.icd,retriever.sn,profile);model=joblib.load(path_for(cfg,"model_dir")/"decision"/(flow+".joblib"))
    tick=time.perf_counter();prepared=retriever.prepare_inputs(data.to_dict("records"),mention);preparation=time.perf_counter()-tick
    predictions=[];diagnostics=[];latencies=[]
    for i,row in enumerate(data.to_dict("records")):
        tick=time.perf_counter();entity=row[mention];chunk=row.get("chunk_text","");chunk_id=row.get("chunk_id",str(i))
        items=[retriever.retrieve(r) for r in prepared[i]]
        scored=ranker.score_items(items);matrices=[features.matrix(x) for x in scored]
        pools,conf,missing=apply_decision(model,scored,matrices,features)
        output={"chunk_id":chunk_id,"clinical_entity":entity}
        for j,(item,pool) in enumerate(zip(scored,pools)):
            system=item["row"]["system"];code=pool[0][0];output["icd10cm_code" if system=="ICD" else "snomed_code"]=code
            diagnostics.append(dict(row=i,chunk_id=chunk_id,system=system,top3=pool[:3],confidence=float(conf[j]),low_conf=bool(conf[j]<=model["low_conf_threshold"]),
                                    gold_not_in_pool_probability=float(missing[j]),cue_flags=item["row"]["context"]["cues"],views=item["row"]["views"],
                                    winner_channels=[name for name,values in item["channels"].items() if code in dict(values)],
                                    specificity_descended=code not in dict(item["pool"])))
        predictions.append(output);latencies.append(time.perf_counter()-tick+preparation/max(1,len(data)))
    output=pd.DataFrame(predictions,columns=["chunk_id","clinical_entity","icd10cm_code","snomed_code"])
    validate_predictions(data,output,{"ICD":set(retriever.icd),"SNOMED":set(retriever.sn)},retriever.icd,features.exceptions)
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+".tmp");output.to_csv(temporary,index=False);temporary.replace(destination)
    diagnostic=destination.with_name("diagnostics.jsonl")
    diagnostic.write_text("".join(json.dumps(d,sort_keys=True)+"\n" for d in diagnostics))
    write_json(path_for(cfg,"run_dir")/a.run_id/"metrics.json",dict(n=len(data),flow=flow,validator="passed",latency_median_ms=float(np.median(latencies)*1000),latency_p95_ms=float(np.quantile(latencies,.95)*1000),peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_device_vram_bytes=getattr(ranker,"peak_device_vram_bytes",None),reranker_backend=cfg.get("reranker",{}).get("backend","transformers"),encoding_provenance=retriever.runtime_provenance))
    ranker.close();print("Predicted and validated",len(data),"rows",flush=True)

if __name__=="__main__":main()
