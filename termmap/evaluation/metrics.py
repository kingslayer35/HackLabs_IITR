"""Exact-match metrics and group-resampled confidence intervals."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd

KS=(1,10,32,64)


def validate_predictions(inputs,predictions,legal,icd_nodes=None,exceptions=None):
    expected=["chunk_id","clinical_entity","icd10cm_code","snomed_code"]
    if list(predictions.columns)!=expected:raise ValueError("Output schema mismatch")
    if len(inputs)!=len(predictions):raise ValueError("Row count changed")
    mention="clinical_entity" if "clinical_entity" in inputs else "entity"
    if predictions.clinical_entity.tolist()!=inputs[mention].tolist():raise ValueError("Mention order changed")
    if "chunk_id" in inputs and predictions.chunk_id.tolist()!=inputs.chunk_id.tolist():raise ValueError("Chunk order changed")
    for system,col in [("ICD","icd10cm_code"),("SNOMED","snomed_code")]:
        for value in predictions[col]:
            if not isinstance(value,str) or not value:raise ValueError("Codes must be nonempty strings")
            if value not in legal[system]:raise ValueError("Prediction outside supplied dictionary")
            if system=="ICD" and icd_nodes is not None and not icd_nodes[value]["is_leaf"] and value not in (exceptions or set()):
                raise ValueError("ICD header outside learned exception list")


def group_ci(values,groups,n=1000,seed=13):
    if len(values)==0:return [None,None]
    values=np.asarray(values,dtype=np.float64)
    groups=np.asarray(groups)
    keys,inverse=np.unique(groups,return_inverse=True)
    sums=np.bincount(inverse,weights=values,minlength=len(keys))
    counts=np.bincount(inverse,minlength=len(keys))
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(n):
        draw=rng.integers(len(keys),size=len(keys))
        samples.append(float(sums[draw].sum()/counts[draw].sum()))
    return [round(float(x)*100,4) for x in np.quantile(samples,[.025,.975])]


def evaluate(rows,pools,frequencies,icd,snomed,n_boot=1000,seed=13,latencies=None,peak_vram=0,links=None):
    frame=rows.reset_index(drop=True).copy()
    if {"entity","chunk_text"} <= set(frame.columns):
        # Preserve exact-input pairing when exported details omit private text.
        frame["pair_key"]=[hashlib.sha256(json.dumps([r.entity,r.chunk_text],ensure_ascii=False).encode()).hexdigest()
                           for r in frame.itertuples()]
    correct=[];recalls={k:[] for k in (1,5,10,20,32,50,64,200)};categories=[];tags=[];cores=[];confidence=[];tops=[];errors=[]
    for r,pool in zip(frame.itertuples(),pools):
        codes=[c for c,_ in pool];top=codes[0] if codes else ""
        correct.append(top==r.code)
        for k in recalls:recalls[k].append(r.code in codes[:k])
        categories.append(r.system=="ICD" and top[:3]==r.code[:3])
        tags.append(r.system=="SNOMED" and bool(snomed.get(r.code,{}).get("tag")) and snomed.get(top,{}).get("tag")==snomed.get(r.code,{}).get("tag"))
        cores.append(r.system=="SNOMED" and bool(snomed.get(r.code,{}).get("core_key")) and snomed.get(top,{}).get("core_key")==snomed.get(r.code,{}).get("core_key"))
        tops.append(r.system=="SNOMED" and snomed.get(top,{}).get("top_level","unknown")==snomed.get(r.code,{}).get("top_level","unknown"))
        if r.system!="SNOMED" or top==r.code:errors.append("correct" if top==r.code else "ICD_error")
        elif top in (links or {}).get(r.code,()):errors.append("near-miss")
        elif tags[-1]:errors.append("same tag, different concept")
        elif tops[-1]:errors.append("same top-level, different tag")
        else:errors.append("wrong top-level")
        confidence.append(pool[0][1] if pool else 0.0)
    frame["correct"]=correct;frame["category_correct"]=categories;frame["tag_correct"]=tags;frame["core_correct"]=cores
    frame["top_level_correct"]=tops;frame["error_class"]=errors
    over=[];under=[];headers=[]
    for r,pool in zip(frame.itertuples(),pools):
        prediction=pool[0][0] if pool else ""
        if r.system=="ICD":
            p,g=icd.get(prediction,{}),icd.get(r.code,{})
            descendant=(p and g and prediction!=r.code and "pre_in" in p and "pre_in" in g and g["pre_in"]<=p["pre_in"]<=g["pre_out"])
            ancestor=(p and g and prediction!=r.code and "pre_in" in p and "pre_in" in g and p["pre_in"]<=g["pre_in"]<=p["pre_out"])
            siblings=bool(p and g and p.get("parent") and p.get("parent")==g.get("parent") and prediction!=r.code)
            over.append(bool(descendant or (siblings and g.get("unspecified") and not p.get("unspecified"))))
            under.append(bool(ancestor or (siblings and p.get("unspecified") and not g.get("unspecified"))))
            headers.append(bool(p and not p["is_leaf"] and not p.get("legacy_gold",False)))
        else:
            # Directed trusted children are passed separately from undirected near-miss links.
            trusted=(links or {}).get("__specifier_children__",{})
            over.append(prediction in trusted.get(r.code,()))
            under.append(r.code in trusted.get(prediction,()))
            headers.append(False)
    frame["over_coding"]=over;frame["under_coding"]=under;frame["header_outside_exception"]=headers
    for k,v in recalls.items():frame[f"R@{k}"]=v
    counts=[frequencies.get((r.system,r.code),0) for r in frame.itertuples()]
    frame["bucket"]=["unseen" if n==0 else "rare" if n<=5 else "common" for n in counts]
    def summarize(part):
        n=len(part)
        output={"n":n,"acc@1":float(part.correct.mean()*100) if n else None,
                "acc_ci":group_ci(part.correct,part.group_id,n_boot,seed),
                "selection_given_pool":float(part.loc[part["R@64"],"correct"].mean()*100) if part["R@64"].any() else None}
        for k in recalls:
            output[f"R@{k}"]=float(part[f"R@{k}"].mean()*100) if n else None
        output["R@64_ci"]=group_ci(part["R@64"],part.group_id,n_boot,seed)
        for name,col in [("category_accuracy","category_correct"),("tag_accuracy","tag_correct"),("top_level_accuracy","top_level_correct"),("same_core_accuracy","core_correct")]:
            target=part.loc[part.system.eq("ICD" if name=="category_accuracy" else "SNOMED")]
            output[name]=float(target[col].mean()*100) if len(target) else None
        output["snomed_error_classes"]=part.loc[part.system.eq("SNOMED")&~part.correct,"error_class"].value_counts().to_dict()
        output["over_coding_rate"]=float(part.over_coding.mean()*100) if n else None
        output["under_coding_rate"]=float(part.under_coding.mean()*100) if n else None
        output["headers_outside_exception"]=int(part.header_outside_exception.sum())
        return output
    systems={}
    for s in ["ICD","SNOMED"]:
        p=frame.loc[frame.system.eq(s)]
        systems[s]=summarize(p)
        systems[s]["buckets"]={b:summarize(p.loc[p.bucket.eq(b)]) for b in ["unseen","rare","common"]}
        systems[s]["subconcepts"]={str(t):summarize(g) for t,g in p.groupby("subconcept")}
    dx=frame.loc[frame.system.eq("SNOMED")&frame.subconcept.eq("DiagnosisCode")]
    systems["SNOMED_Dx"]=summarize(dx)
    systems["SNOMED_Dx"]["buckets"]={b:summarize(dx.loc[dx.bucket.eq(b)]) for b in ["unseen","rare","common"]}
    paired=frame.loc[frame.system.eq("ICD")].merge(frame.loc[frame.system.eq("SNOMED")],on=["entity","chunk_text","group_id"],suffixes=("_icd","_snomed"))
    paired=paired.drop_duplicates(["row_id_icd","row_id_snomed"])
    both=(paired.correct_icd & paired.correct_snomed)
    headline=min(systems["ICD"]["acc@1"],systems["SNOMED_Dx"]["acc@1"]) if len(dx) and systems["ICD"]["n"] else None
    headline_all=min(systems["ICD"]["acc@1"],systems["SNOMED"]["acc@1"]) if systems["ICD"]["n"] and systems["SNOMED"]["n"] else None
    headline_ci=[None,None];headline_all_ci=[None,None]
    if headline is not None or headline_all is not None:
        keys,inverse=np.unique(frame.group_id,return_inverse=True);rng=np.random.default_rng(seed)
        draw=rng.integers(len(keys),size=(n_boot,len(keys)));values=[]
        for mask in [frame.system.eq("ICD"),frame.system.eq("SNOMED")&frame.subconcept.eq("DiagnosisCode"),frame.system.eq("SNOMED")]:
            den=np.bincount(inverse,weights=mask,minlength=len(keys));num=np.bincount(inverse,weights=mask*frame.correct,minlength=len(keys))
            with np.errstate(divide="ignore",invalid="ignore"):values.append(num[draw].sum(1)/den[draw].sum(1))
        if headline is not None:headline_ci=[float(v*100) for v in np.nanquantile(np.minimum(values[0],values[1]),[.025,.975])]
        if headline_all is not None:headline_all_ci=[float(v*100) for v in np.nanquantile(np.minimum(values[0],values[2]),[.025,.975])]
    order=np.lexsort((frame.row_id.astype(str),-np.array(confidence)))
    curve=[]
    for coverage in [.1,.25,.5,.75,.9,.95,1.0]:
        take=max(1,int(np.ceil(len(frame)*coverage)))
        curve.append(dict(coverage=coverage,risk=float(1-frame.iloc[order[:take]].correct.mean())))
    return dict(headline_min_acc=headline,headline_min_acc_ci=headline_ci,headline_min_acc_all_types=headline_all,headline_min_acc_all_types_ci=headline_all_ci,systems=systems,overall=summarize(frame),
                both_correct=dict(n=len(paired),accuracy=float(both.mean()*100) if len(paired) else None,
                                  ci=group_ci(both,paired.group_id,n_boot,seed)),
                risk_coverage=curve,confidence_note="Retrieval score only; calibration starts in P4",
                latency=dict(median_ms=float(np.median(latencies)*1000) if latencies else None,
                             p95_ms=float(np.quantile(latencies,.95)*1000) if latencies else None),
                peak_vram_bytes=peak_vram),frame


def paired_flow_delta(left,right,n_boot=1000,seed=13):
    """right minus left, resampling the same groups for both flows."""
    a=left.sort_values("row_id").reset_index(drop=True)
    b=right.sort_values("row_id").reset_index(drop=True)
    if a.row_id.tolist()!=b.row_id.tolist():raise ValueError("Flow rows are not paired")
    for column in ["group_id","system","code","subconcept","bucket","pair_key"]:
        if column in a and column in b and a[column].tolist()!=b[column].tolist():
            raise ValueError("Flow evaluation inputs differ: "+column)
    groups,inv=np.unique(a.group_id,return_inverse=True)
    masks={"ICD":a.system.eq("ICD").to_numpy(),"SNOMED_all":a.system.eq("SNOMED").to_numpy(),
           "SNOMED_Dx":(a.system.eq("SNOMED")&a.subconcept.eq("DiagnosisCode")).to_numpy()}
    output={};rng=np.random.default_rng(seed)
    draws=rng.integers(len(groups),size=(n_boot,len(groups)))
    for bucket in ["all","unseen","rare","common"]:
        active=np.ones(len(a),dtype=bool) if bucket=="all" else a.bucket.eq(bucket).to_numpy()
        summary={};boot={};point={}
        for name,mask in masks.items():
            mask=mask&active
            if not mask.any():continue
            den=np.bincount(inv,weights=mask,minlength=len(groups))
            l=np.bincount(inv,weights=a.correct.to_numpy()*mask,minlength=len(groups))
            r=np.bincount(inv,weights=b.correct.to_numpy()*mask,minlength=len(groups))
            with np.errstate(invalid="ignore",divide="ignore"):
                al=l[draws].sum(1)/den[draws].sum(1);br=r[draws].sum(1)/den[draws].sum(1)
            point[name]=(l.sum()/den.sum(),r.sum()/den.sum());boot[name]=(al,br)
            delta=(point[name][1]-point[name][0])*100
            summary[name]={"delta":float(delta),"ci":[float(x*100) for x in np.nanquantile(br-al,[.025,.975])]}
        if "ICD" in point and "SNOMED_Dx" in point:
            l=min(point["ICD"][0],point["SNOMED_Dx"][0]);r=min(point["ICD"][1],point["SNOMED_Dx"][1])
            lb=np.minimum(boot["ICD"][0],boot["SNOMED_Dx"][0]);rb=np.minimum(boot["ICD"][1],boot["SNOMED_Dx"][1])
            summary["min_acc"]={"delta":float((r-l)*100),"ci":[float(x*100) for x in np.nanquantile(rb-lb,[.025,.975])]}
        if "ICD" in point and "SNOMED_all" in point:
            l=min(point["ICD"][0],point["SNOMED_all"][0]);r=min(point["ICD"][1],point["SNOMED_all"][1])
            lb=np.minimum(boot["ICD"][0],boot["SNOMED_all"][0]);rb=np.minimum(boot["ICD"][1],boot["SNOMED_all"][1])
            summary["min_acc_all_types"]={"delta":float((r-l)*100),"ci":[float(x*100) for x in np.nanquantile(rb-lb,[.025,.975])]}
        if bucket=="all" and ("pair_key" in a or {"entity","chunk_text"}<=set(a.columns)):
            pair_columns=["pair_key","group_id"] if "pair_key" in a else ["entity","chunk_text","group_id"]
            def paired(frame):
                pair=frame.loc[frame.system.eq("ICD")].merge(frame.loc[frame.system.eq("SNOMED")],
                    on=pair_columns,suffixes=("_icd","_snomed"))
                pair=pair.drop_duplicates(["row_id_icd","row_id_snomed"]).sort_values(["row_id_icd","row_id_snomed"])
                return pair,(pair.correct_icd & pair.correct_snomed).to_numpy(dtype=float)
            lp,lv=paired(a);rp,rv=paired(b)
            if len(lp):
                assert lp[["row_id_icd","row_id_snomed"]].values.tolist()==rp[["row_id_icd","row_id_snomed"]].values.tolist()
                summary["both_correct"]={"delta":float(np.mean(rv-lv)*100),"ci":group_ci(rv-lv,lp.group_id,n_boot,seed)}
        output[bucket]=summary
    return output
