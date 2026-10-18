"""Supplement completed C1 pools without changing their primary retrieval ranks."""
import joblib
from termmap.common.io import load_config,path_for,write_json
from termmap.retrieval.cache import directory
from termmap.retrieval.candidates import CachedRetriever
from termmap.decision.coupling import other_row,infer_types

cfg=load_config();retriever=CachedRetriever(cfg);records=joblib.load(directory(cfg)/"records.joblib")
infer_types(cfg,retriever,records,"C1")
for dataset in ["train","dev","probe","alias"]:
    target=path_for(cfg,"run_dir")/"C1"/f"{dataset}_pools.joblib";items=joblib.load(target)
    by_id={r["row_id"]:r for r in records[dataset]}
    for i,item in enumerate(items):
        row=by_id[item["row"]["row_id"]]
        if "inferred_subconcept" in row:item["row"]["inferred_subconcept"]=row["inferred_subconcept"]
        if dataset in {"dev","probe"} and "other_system" not in item:item["other_system"]=retriever.retrieve(other_row(item["row"],cfg))
        if i%500==0:print("Joint pools",dataset,i,"/",len(items),flush=True)
    temporary=target.with_suffix(".tmp");joblib.dump(items,temporary);temporary.replace(target)
joblib.dump(records,directory(cfg)/"records.joblib")
write_json(path_for(cfg,"run_dir")/"joint_preparation.json",dict(complete=True,primary_pools_unchanged=True,unlabelled_auxiliary_system=True,type_classifier_train_only=True))
