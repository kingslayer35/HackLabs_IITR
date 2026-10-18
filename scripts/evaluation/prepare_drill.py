"""Remove each dev gold before retrieval; no label enters any query or feature."""
import json
import joblib
from termmap.common.io import load_config,path_for
from termmap.retrieval.cache import directory
from termmap.retrieval.candidates import CachedRetriever
from termmap.decision.coupling import other_row

cfg=load_config()
retriever=CachedRetriever(cfg,cfg.get("evaluation",{}).get("adapted_source","C2_v2"))
records=joblib.load(directory(cfg)/"records.joblib")["dev"];items=[]
for i,row in enumerate(records):
    row=dict(row,forbidden_code=row["code"])
    item=retriever.retrieve(row,forbidden=row["code"]);assert row["code"] not in dict(item["pool"]);items.append(item)
    item["other_system"]=retriever.retrieve(other_row(row,cfg),forbidden=row["code"])
    if i%250==0:print("Missing-code drill",i,"/",len(records),flush=True)
dest=path_for(cfg,"run_dir")/"drill";dest.mkdir(parents=True,exist_ok=True);joblib.dump(items,dest/"drill_pools.joblib")
