"""Single-pass M1–M4 measurements, with an explicit one-round tightening mode."""
import argparse
import json
from collections import Counter,defaultdict
import joblib
import numpy as np
import pandas as pd
from termmap.common.io import load_config,term_dir,path_for,read_partition,write_json
from termmap.preprocessing.normalize import Normalizer,clean,context_features
from termmap.preprocessing.noise import corrupt
from termmap.retrieval.pipeline import combine_views


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--tighten",action="store_true")
    args=ap.parse_args();cfg=load_config(args.config);out=term_dir(cfg)
    train=read_partition(cfg,"train");dev=read_partition(cfg,"dev")
    aliases=pd.read_parquet(out/"aliases.parquet")
    abbr=pd.read_parquet(out/"abbreviations.parquet")
    held=set(pd.read_parquet(out/"probe_set.parquet").heldout_norm)
    normalizer=Normalizer(aliases,train,abbr,alias_support=pd.read_parquet(out/"abbreviation_alias_support_train.parquet"),excluded_names=held)
    normalizer.tightened=True
    rng=np.random.default_rng(cfg["seed"])
    def views(row,training=False):
        context=context_features(row.entity,row.chunk_text,window=cfg["preprocessing"]["window_chars"])
        return normalizer.views(row.entity,"" if context["no_context"] else context["window"],row.group_id if training else None)
    train_views=[views(r,True) for r in train.itertuples()]
    dev_views=[views(r) for r in dev.itertuples()]
    names=sorted(set(aliases.norm_text))
    ids=rng.choice(len(names),cfg["preprocessing"]["noise_sample"],replace=False)
    m2=defaultdict(lambda:Counter(total=0,recovered=0,changed=0));damage=Counter();eligible=[]
    for i in ids:
        text=names[i]
        original=normalizer.views(text)
        noisy,kind=corrupt(text,seed=cfg["seed"],key=f"M2:{i}",abbreviations=normalizer.abbr)
        repaired=normalizer.views(noisy)
        m2[kind]["total"]+=1
        recovered=any(text in values for values in repaired.values())
        m2[kind]["recovered"]+=recovered
        changes={v:any(x!=text for x in original[v]) for v in ["V1","V2","V3","V4"]}
        for view,changed in changes.items():damage[view]+=changed
        damage["any"]+=any(changes.values())
    m1={}
    for view in ["V1","V2","V3","V4"]:
        changes=[any(t!=r.entity for t in vs[view]) for r,vs in zip(train.itertuples(),train_views)]
        top=Counter((clean(r.entity)," | ".join(vs[view])) for r,vs,c in zip(train.itertuples(),train_views,changes) if c)
        m1[view]={"changed_percent":sum(changes)/len(changes)*100,
                  "per_subconcept":{str(t):float(np.mean(np.array(changes)[g.index])*100) for t,g in train.groupby("subconcept")},
                  "top_changes":[dict(before=a,after=b,count=n) for (a,b),n in top.most_common(50)]}
    # Save private examples separately; report aggregate counts only.
    for pos,(row,views) in enumerate(zip(train.itertuples(),train_views)):
        meaningful=[v for v in ["V2","V3","V4"] if any(t!=clean(row.entity) for t in views[v])]
        if meaningful:
            eligible.append(dict(row_id=row.row_id,mention=row.entity,views={v:views[v] for v in meaningful},
                                 display=row.display,context=row.chunk_text[:0],judgment="pending",reason=""))
    sample_path=out/("preprocessing_audit_tightened.json" if args.tighten else "preprocessing_audit.json")
    if not sample_path.exists():
        if len(eligible)<100:raise RuntimeError("Fewer than 100 changed training mentions for M4")
        sample=[eligible[i] for i in sorted(rng.choice(len(eligible),100,replace=False))]
        write_json(sample_path,sample)
    else:
        sample=json.loads(sample_path.read_text());final_by_id={r.row_id:v for r,v in zip(train.itertuples(),train_views)};changes=[]
        for i,row in enumerate(sample):
            final=final_by_id[row["row_id"]];current={v:final[v] for v in ["V2","V3","V4"] if any(t!=clean(row["mention"]) for t in final[v])}
            if current!=row["views"]:changes.append(dict(audit_index=i,row_id=row["row_id"],old=row["views"],new=current))
        joblib.dump(changes,path_for(cfg,"run_dir")/"m4_final_view_changes.joblib")
        print("M4 frozen-sample entries needing final-view review",len(changes),flush=True)
    result={"tightened":args.tighten,"context_scope":"extracted window","train_group_mask":True,"M1":m1,"M2":dict(m2),
            "damage_percent":{v:n/cfg["preprocessing"]["noise_sample"]*100 for v,n in damage.items()},"M3":{}}
    for system in ["ICD","SNOMED"]:
        index=joblib.load(path_for(cfg,"index_dir")/"C0"/f"{system}_char.joblib")
        ids=dev.index[dev.system.eq(system)].tolist()
        texts=sorted({t for i in ids for values in dev_views[i].values() for t in values})
        cached={}
        for start in range(0,len(texts),32):cached.update(zip(texts[start:start+32],index.search(texts[start:start+32],100)))
        raw=[cached[t] for t in dev.loc[ids,"entity"]]
        for view in ["V1","V2","V3","V4"]:
            wins=losses=raw1=new1=raw64=new64=changed_n=changed_raw64=changed_new64=0
            for pos,i in enumerate(ids):
                base=raw[pos];r=dev.iloc[i];texts=dev_views[i][view]
                pooled=combine_views([base,*[cached[t] for t in texts]],100)
                before=r.code in [c for c,_ in base[:64]];after=r.code in [c for c,_ in pooled[:64]]
                wins+=after and not before;losses+=before and not after
                raw1+=base[0][0]==r.code;new1+=pooled[0][0]==r.code;raw64+=before;new64+=after
                if any(t!=r.entity for t in texts):
                    changed_n+=1;changed_raw64+=before;changed_new64+=after
            result["M3"][f"{system}/{view}"]=dict(n=len(ids),wins=wins,losses=losses,
                raw_R1=raw1/len(ids)*100,with_view_R1=new1/len(ids)*100,
                raw_R64=raw64/len(ids)*100,with_view_R64=new64/len(ids)*100,
                changed_n=changed_n,changed_raw_R64=changed_raw64/changed_n*100 if changed_n else None,
                changed_with_view_R64=changed_new64/changed_n*100 if changed_n else None)
        print("M3",system,"complete",flush=True)
    run="preprocessing_tightened" if args.tighten else "preprocessing"
    write_json(path_for(cfg,"run_dir")/run/"metrics.json",result)
    text="# Preprocessing measurements\n\nM1 examples and the 100 sampled training mentions stay in ignored local artifacts; no chart slices are printed here.\n\n"
    text+=f"Tightening round: {args.tighten}.\n\nM2 uncorrupted-name damage percentages: {result['damage_percent']}.\n\n"
    text+="M2 uses V1-canonical dictionary names; case/punctuation formatting is canonicalized before defining damage. Any subsequent change of an uncorrupted canonical name counts as damage.\n\n"
    text+="| Noise | n | Recovered | Recovery % |\n|---|---:|---:|---:|\n"
    for kind,v in result["M2"].items():text+=f"| {kind} | {v['total']} | {v['recovered']} | {v['recovered']/v['total']*100:.3f} |\n"
    text+="\n| View | Train mentions changed % |\n|---|---:|\n"
    for view,v in m1.items():text+=f"| {view} | {v['changed_percent']:.3f} |\n"
    text+="\n| System/view | Wins | Losses | Raw R@1 | With R@1 | Raw R@64 | With R@64 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for view,v in result["M3"].items():text+=f"| {view} | {v['wins']} | {v['losses']} | {v['raw_R1']:.3f} | {v['with_view_R1']:.3f} | {v['raw_R64']:.3f} | {v['with_view_R64']:.3f} |\n"
    text+="\nM4 judgments pending; no precision is claimed until every sampled mention has been reviewed.\n"
    (path_for(cfg,"report_dir")/"preprocessing.md").write_text(text)


if __name__=="__main__":main()
