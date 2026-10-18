"""Unlabelled opposite-system queries and train-only missing-type inference."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import json
import hashlib
from sklearn.linear_model import LogisticRegression
from termmap.common.io import path_for,term_dir,write_json
from termmap.preprocessing.normalize import clean
from termmap.terminology.hierarchy import strip_tag


def effective_type(row):
    value=row.get("subconcept")
    return value if value not in {None,"","Unknown"} else row.get("inferred_subconcept","Unknown")


def other_row(row,cfg):
    system="SNOMED" if row["system"]=="ICD" else "ICD"
    result={k:v for k,v in row.items() if k not in {"code","display","valid_gold","dirty_label","display_matches"}}
    result.update(system=system,row_id="aux:"+str(row["row_id"]),code="",is_auxiliary=True,aux_for=row["row_id"],valid_gold=False)
    enabled=cfg["retrieval"].get("enabled_by_system",{}).get(system,cfg["retrieval"]["enabled_views"])
    result["lexical_queries"]=list(dict.fromkeys(t for v in enabled for t in row["views"][v]))
    result["dense_queries"]=list(dict.fromkeys(row["views"][v][0] for v in ["V1","V2","V3","V4"] if v=="V1" or v in enabled))
    return result


def type_cache_path(cfg,run,final=False):
    generation=(str(cfg["generation"])+"_") if cfg.get("generation") else ""
    return path_for(cfg,"model_dir")/(f"type_{generation}{run}.joblib" if final else f"type_eval_{generation}{run}.joblib")


def infer_types(cfg,retriever,records,run):
    """No dev/probe labels: fit only the provided train-row subconcepts."""
    training=records.get("train",records.get("all"))
    if training is None:
        # Test-only query caches do not contain train rows. Reuse the frozen,
        # new-generation original-query classifier rather than fitting on test
        # labels or silently leaving missing types unresolved.
        source=type_cache_path(cfg,cfg.get("evaluation",{}).get("base_source","C1"))
        audit=source.with_suffix(".json")
        if not source.exists() or not audit.exists():
            raise RuntimeError("Test type inference needs the train-only C1 classifier")
        metadata=json.loads(audit.read_text())
        if metadata.get("final_all_labels") or metadata.get("probe_exclusion_version")!=3:
            raise RuntimeError("Type classifier is not the confirmed train-only generation")
        model=joblib.load(source)
        for rows in records.values():
            unknown=[r for r in rows if r.get("subconcept") in {None,"","Unknown"}]
            if unknown:
                X=np.stack([retriever.qvectors[retriever.qid[r["views"]["V1"][0]]] for r in unknown]).astype(np.float32)
                for row,value in zip(unknown,model.predict(X)):row["inferred_subconcept"]=str(value)
        return
    final="all" in records
    path=type_cache_path(cfg,run,final)
    audit=path.with_suffix(".json")
    held=set() if final else set(pd.read_parquet(term_dir(cfg)/"probe_set.parquet").heldout_norm)
    rows=[r for r in training if clean(strip_tag(r["entity"])) not in held]
    X=np.stack([retriever.qvectors[retriever.qid[r["views"]["V1"][0]]] for r in rows]).astype(np.float32)
    expected=dict(probe_exclusion_version=3,train_rows=len(rows),excluded_rows=len(training)-len(rows),
                  final_all_labels=final,seed=13,
                  original_query_vectors_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                  targets_sha256=hashlib.sha256(json.dumps([r["subconcept"] for r in rows]).encode()).hexdigest())
    if path.exists() and audit.exists() and json.loads(audit.read_text())==expected:model=joblib.load(path)
    else:
        model=LogisticRegression(C=1,max_iter=1000,random_state=13).fit(X,[r["subconcept"] for r in rows]);path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path)
        write_json(audit,expected)
    for rows in records.values():
        unknown=[r for r in rows if r.get("subconcept") in {None,"","Unknown"}]
        if unknown:
            X=np.stack([retriever.qvectors[retriever.qid[r["views"]["V1"][0]]] for r in unknown]).astype(np.float32)
            for row,value in zip(unknown,model.predict(X)):row["inferred_subconcept"]=str(value)
