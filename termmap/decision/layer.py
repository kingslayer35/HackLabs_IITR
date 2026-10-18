"""Group-cross-fitted fusion, specificity, local pairing and confidence."""
from collections import defaultdict,Counter
import math
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax,expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from termmap.common.io import term_dir,path_for
from termmap.preprocessing.normalize import clean
from termmap.decision.coupling import effective_type


FEATURES=["reranker_z","rrf_rank","dense","lexical","memory","frequency","leaf","unspecified","other","depth","tag_prior","top_prior","negation_match","history_match","family_match","literal","refinement_context","same_category","sibling","parent","tree_distance","category_vote","family_vote","agreement","seventh_context","seventh_prior"]


class Features:
    def __init__(self,cfg,icd,sn,profile="eval"):
        self.icd=icd;self.sn=sn;out=term_dir(cfg)
        suffix="final" if profile=="final" else "train"
        self.freq={(r.system,r.code):r.frequency for r in pd.read_parquet(out/f"priors_{suffix}.parquet").itertuples()}
        self.tags={(r.subconcept,r.tag):r.probability for r in pd.read_parquet(out/f"tag_priors_{suffix}.parquet").itertuples()}
        self.tops={(r.subconcept,r.top_level):r.probability for r in pd.read_parquet(out/f"top_level_priors_{suffix}.parquet").itertuples()}
        self.seventh={(r.category,r.seventh_char):r.probability for r in pd.read_parquet(out/f"seventh_priors_{suffix}.parquet").itertuples()}
        self.exceptions=set(pd.read_parquet(out/f"header_exceptions_{suffix}.parquet").code)
        co=pd.read_parquet(out/f"cooccurrence_{suffix}.parquet")
        self.co={(r.icd,r.snomed):float((r.count+1)/(r.total+2)) for r in co.itertuples()}
        self.children=defaultdict(set)
        for r in pd.read_parquet(out/"snomed_links.parquet").itertuples():
            if r.kind=="specifier_child":self.children[r.src].add(r.dst)
        self._sim={};self._vectors=None;self.cfg=cfg

    def node(self,system,code):return (self.icd if system=="ICD" else self.sn)[code]

    def evidence(self,row,code):
        if row["system"]!="ICD":return 0.,0.
        node=self.icd[code];text=clean(row.get("context",{}).get("window",row["entity"]))
        words=[w for w in re.findall(r"[a-z0-9]+",node.get("refinement_text","")) if w not in {"of","the","and","with","unspecified","other","nos","a","an","in","for"}]
        tokens=set(re.findall(r"[a-z0-9]+",text));ref=sum(w in tokens for w in words)/len(words) if words else 0.
        meaning=node.get("seventh_char_meaning","").lower()
        cues=row.get("context",{}).get("cues",{})
        seventh=float(sum((2 if w in {"routine healing","delayed healing","nonunion","malunion"} else 1) for w in ["initial","subsequent","sequela","routine healing","delayed healing","nonunion","malunion"] if w in text and w in meaning))
        return ref,seventh

    def matrix(self,item):
        row=item["row"];system=row["system"];codes=[c for c,_ in item["pool"]]
        raw=np.array([item["reranker_scores"][c] for c in codes]);z=(raw-raw.mean())/max(raw.std(),1e-5)
        probabilities=softmax(z);top=min(codes,key=lambda c:(-round(float(item["reranker_scores"][c]),5),c));tn=self.node(system,top)
        votes=defaultdict(float);family=defaultdict(float)
        for c,p in zip(codes,probabilities):
            n=self.node(system,c);votes[c[:3]]+=p
            if system=="SNOMED":family[n["family_id"]]+=p
        cues=row.get("context",{}).get("cues",{});result=[]
        for j,c in enumerate(codes):
            n=self.node(system,c);f=item["features"].get(c,{});ref,sev=self.evidence(row,c)
            same=system=="ICD" and c[:3]==top[:3];sibling=system=="ICD" and bool(n["parent"]) and n["parent"]==tn["parent"]
            parent=system=="ICD" and (n["parent"]==top or tn["parent"]==c)
            distance=abs(n.get("depth",0)-tn.get("depth",0)) if same else 10
            if same:
                ancestors={top};parent_code=tn["parent"]
                while parent_code:ancestors.add(parent_code);parent_code=self.icd[parent_code]["parent"]
                common=c
                while common not in ancestors and self.icd[common]["parent"]:common=self.icd[common]["parent"]
                distance=n["depth"]+tn["depth"]-2*self.icd[common]["depth"]
            unspecified=bool(n.get("unspecified")) or bool(re.search(r"\b(unspecified|nos)\b",n.get("preferred", ""),re.I))
            prior_tag=self.tags.get((effective_type(row),n.get("tag")),1e-6)
            prior_top=self.tops.get((effective_type(row),n.get("top_level")),1e-6)
            result.append([z[j],1/(j+1),f.get("dense",0),f.get("lexical",0),f.get("memory",0),min(math.log1p(self.freq.get((system,c),0)),5),
                           float(n.get("is_leaf",False)),float(unspecified),float(n.get("other",False)),n.get("depth",0),math.log(prior_tag+1e-6) if system=="SNOMED" else 0,math.log(prior_top+1e-6) if system=="SNOMED" else 0,
                           float(bool(n.get("absent")) and bool(cues.get("negation"))),float(bool(n.get("history")) and bool(cues.get("history"))),float(bool(n.get("family_history")) and bool(cues.get("family"))),
                           f.get("literal",0),ref,float(same),float(sibling),float(parent),distance,votes[c[:3]] if system=="ICD" else 0,family[n["family_id"]] if system=="SNOMED" else 0,f.get("agreement",0),sev,self.seventh.get((c[:3],c.replace(".","")[-1]),0) if len(c.replace(".",""))==7 and system=="ICD" else 0])
        return np.asarray(result,dtype=np.float64)

    def similarity(self,icd,sn):
        key=(icd,sn)
        if key in self._sim:return self._sim[key]
        if self._vectors is None:
            root=(path_for(self.cfg,"frozen_alias_index_dir") if "frozen_alias_index_dir" in self.cfg["paths"] else path_for(self.cfg,"index_dir")/"eval"/"C1")
            from termmap.retrieval.cache import directory
            revised=directory(self.cfg)/"C1/ICD_alias_rows.parquet"
            a=pd.read_parquet(revised if revised.exists() else root/"ICD_alias_rows.parquet");ids={}
            for r in a.iloc[::-1].itertuples():ids[r.code]=r.Index
            preferred={c:clean(n["preferred"]) for c,n in self.icd.items()}
            for r in a.itertuples():
                if r.norm_text==preferred.get(r.code):ids[r.code]=r.Index
            sv=np.load(root/"snomed_preferred.npy",mmap_mode="r")
            sc=np.load(root/"snomed_preferred_codes.npy")
            self._vectors=(np.load(root/"ICD_aliases.npy",mmap_mode="r"),ids,sv,{c:i for i,c in enumerate(sc)})
        iv,ii,sv,si=self._vectors
        value=float(np.dot(np.asarray(iv[ii[icd]],dtype=np.float32),np.asarray(sv[si[sn]],dtype=np.float32))) if icd in ii and sn in si else 0.
        self._sim[key]=round(value,5);return self._sim[key]

    def descend(self,row,codes,scores):
        if row["system"]!="ICD":return codes,scores
        chosen=min(range(len(codes)),key=lambda i:(-round(float(scores[i]),5),codes[i]));code=codes[chosen]
        observed={c:float(s) for c,s in zip(codes,scores)};visited=set()
        while not self.icd[code]["is_leaf"] and code not in visited:
            visited.add(code);children=[c for c in self.icd[code]["children"] if c!=row.get("forbidden_code")]
            supported=[c for c in children if self.evidence(row,c)[0]>=.999 and self.icd[c].get("refinement_text") and not self.icd[c].get("unspecified")]
            if code in self.exceptions and not supported:break
            if supported:choices=supported
            else:
                text=clean(row.get("context",{}).get("window",row["entity"]))
                details=re.findall(r"\b(?:stage|grade|type|class)\s+[0-9]+[a-z]?|\b(?:left|right|bilateral|acute|chronic|subacute|initial|subsequent|sequela)\b",text)
                parent_name=clean(self.icd[code].get("canonical_description",self.icd[code].get("preferred","")))
                child_names=[clean(self.icd[c].get("canonical_description",self.icd[c].get("preferred",""))) for c in children]
                uncovered=any(d not in parent_name and not any(d in n for n in child_names) for d in details)
                choices=[c for c in children if self.icd[c].get("other")] if uncovered or "other" in clean(row["entity"]) else []
                if not choices:choices=[c for c in children if self.icd[c].get("unspecified")]
                if not choices:choices=children
            if not choices:
                # Removing the sole gold child can exhaust this branch in the drill.
                choices=[c for c in codes if c!=code and c!=row.get("forbidden_code") and self.icd[c]["is_leaf"]]
                if not choices:raise ValueError("No legal fallback child or pooled leaf")
            evidence={c:self.evidence(row,c)[1] for c in choices}
            if max(evidence.values())>0:choices=[c for c in choices if evidence[c]==max(evidence.values())]
            def key(c):
                n=self.icd[c];prior=self.seventh.get((c[:3],c.replace(".","")[-1]),0) if n.get("seventh_char_meaning") else 0
                return (-round(observed.get(c,min(scores)-1)+prior,5),c)
            code=min(choices,key=key)
        if code!=codes[chosen]:
            # Policy can descend through an unscored child; make this explicit in diagnostics.
            if code not in observed:codes=list(codes)+[code];scores=np.append(scores,scores[chosen]+1e-4)
            else:scores=np.asarray(scores).copy();scores[codes.index(code)]=max(scores)+1e-4
        return codes,scores


def fit_lr(matrices,items,indices,C):
    X=np.concatenate([matrices[i] for i in indices]);y=np.concatenate([[c==items[i]["row"]["code"] for c,_ in items[i]["pool"]] for i in indices])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=500,random_state=13))
    model.fit(X,y);return model


def headline(items,predictions,objective="diagnosis"):
    if objective not in {"diagnosis","all_types"}:raise ValueError("Unknown decision objective")
    sn_label="SNOMED" if objective=="all_types" else "SNOMED_Dx"
    values={s:[] for s in ["ICD",sn_label]}
    for item,c in zip(items,predictions):
        r=item["row"];s="ICD" if r["system"]=="ICD" else sn_label if objective=="all_types" or r.get("subconcept")=="DiagnosisCode" else None
        if r.get("is_auxiliary"):continue
        if s:values[s].append(c==r["code"])
    return min(np.mean(x) for x in values.values() if x)


def base_scores(model,matrices):return [model.decision_function(x) for x in matrices]


def fit_offset(items,matrices,scores,features):
    # Evaluate only score-crossing breakpoints for a single unspecified offset.
    boundaries=[0.]
    for x,s in zip(matrices,scores):
        mask=x[:,7].astype(bool)
        if mask.any() and (~mask).any():boundaries.append(float(s[~mask].max()-s[mask].max()))
    boundaries=sorted(set(boundaries));candidates=sorted(set([boundaries[0]-1,boundaries[-1]+1,0.]+boundaries+[(a+b)/2 for a,b in zip(boundaries,boundaries[1:])]))
    best=None;alternatives=[]
    for item,x,s in zip(items,matrices,scores):
        codes=[c for c,_ in item["pool"]];mask=x[:,7].astype(bool)
        options=[]
        for flag in [False,True]:
            ids=np.flatnonzero(mask==flag)
            j=min(ids,key=lambda j:(-round(float(s[j]),5),codes[j])) if len(ids) else None
            code=codes[j] if j is not None else "";correct=code==item["row"]["code"]
            row=item["row"];over=under=False
            if code and row["system"]=="ICD":
                p=features.icd[code];g=features.icd.get(row["code"],{})
                if g and code!=row["code"]:
                    sibling=bool(p["parent"]) and p["parent"]==g["parent"]
                    over=g["pre_in"]<=p["pre_in"]<=g["pre_out"] or (sibling and g.get("unspecified") and not p.get("unspecified"))
                    under=p["pre_in"]<=g["pre_in"]<=p["pre_out"] or (sibling and p.get("unspecified") and not g.get("unspecified"))
            elif code and row["system"]=="SNOMED":
                over=code in features.children.get(row["code"],())
                under=row["code"] in features.children.get(code,())
            options.append((float(s[j]) if j is not None else -1e10,code,float(correct),float(bool(over)),float(bool(under))))
        alternatives.append(options)
    score0=np.array([a[0][0] for a in alternatives]);score1=np.array([a[1][0] for a in alternatives])
    codetie=np.array([a[1][1]<a[0][1] for a in alternatives])
    counts=np.array([[[o[2],o[3],o[4]] for o in a] for a in alternatives])
    for offset in candidates:
        a=np.round(score0,5);b=np.round(score1+offset,5);choose=(b>a)|((b==a)&codetie)
        correct,over,under=counts[np.arange(len(items)),choose.astype(int)].sum(0)
        value=(correct,-abs(over-under),-abs(offset))
        if best is None or value>best[0]:best=(value,offset)
    return float(best[1])


def independent(items,matrices,scores,features,offset):
    result=[]
    for item,x,s in zip(items,matrices,scores):
        codes=[c for c,_ in item["pool"]];s=s+offset*x[:,7]
        codes,s=features.descend(item["row"],codes,s)
        order=sorted(range(len(codes)),key=lambda j:(-round(float(s[j]),5),codes[j]))
        result.append([(codes[j],round(float(s[j]),5)) for j in order])
    return result


def legal_icd(row,code,features):
    if code==row.get("forbidden_code"):return False
    node=features.icd[code]
    return node["is_leaf"] or (code in features.exceptions and not any(
        child!=row.get("forbidden_code") and features.evidence(row,child)[0]>=.999
        and features.icd[child].get("refinement_text") and not features.icd[child].get("unspecified")
        for child in node["children"]))


def couple(items,pools,features,w,tau):
    result=[list(p) for p in pools];pairs=defaultdict(dict)
    for i,item in enumerate(items):
        r=item["row"];key=r.get("decision_pair_id",(r.get("group_id"),r.get("entity"),r.get("chunk_text","")));pairs[key][r["system"]]=i
    for pair in pairs.values():
        if "ICD" not in pair or "SNOMED" not in pair:continue
        a,b=pair["ICD"],pair["SNOMED"];pi,pj=pools[a][:5],pools[b][:5]
        row=items[a]["row"]
        pi=[(c,s) for c,s in pi if legal_icd(row,c,features)]
        options=[]
        for i,si in pi:
            for j,sj in pj:
                agreement=max(features.co.get((i,j),0),float(features.similarity(i,j)>=tau))
                options.append((round(si+sj+w*agreement,5),i,j))
        _,i,j=min(options,key=lambda q:(-q[0],q[1],q[2]))
        for k,c in [(a,i),(b,j)]:
            high=max(s for _,s in result[k]);result[k]=[(cc,high+1e-4 if cc==c else ss) for cc,ss in result[k]]
            result[k].sort(key=lambda t:(-round(t[1],5),t[0]))
    return result


def fit_coupling(items,pools,features):
    # Exact one-dimensional threshold fit: update only pairs whose indicator changes.
    # This visits observed breakpoints rather than introducing a threshold grid.
    bykey=defaultdict(dict)
    for k,x in enumerate(items):
        r=x["row"];key=r.get("decision_pair_id",(r.get("group_id"),r["entity"],r.get("chunk_text","")));bykey[key][r["system"]]=k
    pairs=[];events=defaultdict(list)
    for p in bykey.values():
        if "ICD" in p and "SNOMED" in p:
            a,b=p["ICD"],p["SNOMED"];row=items[a]["row"]
            legal=[(c,s) for c,s in pools[a][:5] if legal_icd(row,c,features)]
            options=[]
            for i,si in legal:
                for j,sj in pools[b][:5]:
                    index=len(options);sim=features.similarity(i,j);options.append((i,j,si+sj,features.co.get((i,j),0)))
                    events[sim].append((len(pairs),index))
            pairs.append((a,b,options))
    if not pairs:return 0.,1.
    objective=getattr(features,"cfg",{}).get("decision",{}).get("objective","diagnosis")
    sn_label="SNOMED" if objective=="all_types" else "SNOMED_Dx"
    systems=["auxiliary" if x["row"].get("is_auxiliary") else "ICD" if x["row"]["system"]=="ICD" else sn_label if objective=="all_types" or x["row"].get("subconcept")=="DiagnosisCode" else "other" for x in items]
    gold=[x["row"].get("code","") for x in items];den=Counter(systems)
    initial=[p[0][0] for p in pools];best=(-1,None)
    for w in [0.,.5,1.,2.]:
        predictions=list(initial);counts=Counter()
        for s,c,g in zip(systems,predictions,gold):counts[s]+=c==g
        scores=[];winners=[]
        def update(k,new):
            a,b,options=pairs[k];i,j,*_=options[new]
            for row,c in [(a,i),(b,j)]:
                counts[systems[row]]+=int(c==gold[row])-int(predictions[row]==gold[row]);predictions[row]=c
            winners[k]=new
        for k,(a,b,options) in enumerate(pairs):
            values=[round(base+w*co,5) for i,j,base,co in options];scores.append(values);winners.append(-1)
            winner=min(range(len(values)),key=lambda q:(-values[q],options[q][0],options[q][1]));update(k,winner)
        def value():return min(counts[s]/den[s] for s in ["ICD",sn_label] if den[s])
        score=value()
        if score>best[0]:best=(score,(w,max(events)+1e-5))
        if w==0:continue
        for tau in sorted(events,reverse=True):
            affected=set()
            for k,q in events[tau]:scores[k][q]=round(pairs[k][2][q][2]+w,5);affected.add(k)
            for k in sorted(affected):
                options=pairs[k][2];winner=min(range(len(options)),key=lambda q:(-scores[k][q],options[q][0],options[q][1]))
                if winner!=winners[k]:update(k,winner)
            score=value()
            if score>best[0]:best=(score,(w,tau))
    return best[1]


def confidence_features(items,pools,temperature=1.,features=None):
    paired=defaultdict(dict)
    for i,x in enumerate(items):
        r=x["row"];key=r.get("decision_pair_id",(r.get("group_id"),r["entity"],r.get("chunk_text","")));paired[key][r["system"]]=i
    agreement=np.zeros(len(items))
    if features is not None:
        for pair in paired.values():
            if "ICD" in pair and "SNOMED" in pair:
                a,b=pair["ICD"],pair["SNOMED"];i,j=pools[a][0][0],pools[b][0][0]
                agreement[a]=agreement[b]=max(features.co.get((i,j),0),features.similarity(i,j))
        for k,item in enumerate(items):
            if item.get("coupling_partner"):
                c=pools[k][0][0];partner=item["coupling_partner"]
                i,j=(c,partner) if item["row"]["system"]=="ICD" else (partner,c)
                agreement[k]=max(features.co.get((i,j),0),features.similarity(i,j))
    values=[]
    for k,(item,pool) in enumerate(zip(items,pools)):
        scores=np.asarray([s for _,s in pool]);p=softmax(scores/temperature);c=pool[0][0];f=item["features"].get(c,{})
        values.append([p[0],p[0]-(p[1] if len(p)>1 else 0),f.get("agreement",0),f.get("dense",0),f.get("lexical",0),float(c not in dict(item["pool"])),agreement[k]])
    return np.asarray(values)


def fit_confidence(items,pools,features):
    def nll(logt):
        temperature=math.exp(logt);values=[]
        for item,pool in zip(items,pools):
            codes=[c for c,_ in pool]
            if item["row"]["code"] in codes:values.append(-math.log(max(softmax(np.array([s for _,s in pool])/temperature)[codes.index(item["row"]["code"])],1e-12)))
        return np.mean(values) if values else 0
    temperature=math.exp(minimize_scalar(nll,bounds=(-3,3),method="bounded").x)
    X=confidence_features(items,pools,temperature,features);y=np.array([p[0][0]==x["row"]["code"] for x,p in zip(items,pools)])
    missing=np.array([x["row"]["code"] not in dict(x["pool"]) for x in items])
    def fit(y):
        if len(np.unique(y))<2:return float(y.mean())
        return make_pipeline(StandardScaler(),LogisticRegression(C=1,random_state=13,max_iter=300)).fit(X,y)
    return dict(temperature=temperature,correct=fit(y),missing=fit(missing))


def confidence(model,items,pools,features):
    X=confidence_features(items,pools,model["temperature"],features)
    def predict(estimator):return np.full(len(X),estimator) if isinstance(estimator,float) else estimator.predict_proba(X)[:,1]
    p=predict(model["correct"]);missing=predict(model["missing"])
    return np.round(np.clip(p*(1-missing),1e-5,1-1e-5),5),np.round(missing,5)


def joint_candidates(items,matrices,features):
    joint=[];values=[];positions=[];partners={}
    for k,(item,x) in enumerate(zip(items,matrices)):
        positions.append(len(joint));primary=item
        if "other_system" in item:primary=dict(item,row=dict(item["row"],decision_pair_id=item["row"]["row_id"]))
        joint.append(primary);values.append(x)
        if "other_system" in item:
            other=item["other_system"]
            other=dict(other,row=dict(other["row"],decision_pair_id=item["row"]["row_id"],is_auxiliary=True))
            partners[k]=len(joint);joint.append(other)
            if "_matrix" not in item["other_system"]:item["other_system"]["_matrix"]=features.matrix(other)
            values.append(item["other_system"]["_matrix"])
    return joint,values,positions,partners


def primary_decisions(items,pools,positions,partners):
    decorated=[dict(item,coupling_partner=pools[partners[k]][0][0]) if k in partners else item for k,item in enumerate(items)]
    return decorated,[pools[i] for i in positions]


def fit_decision(items,matrices,features,C):
    model=fit_lr(matrices,items,list(range(len(items))),C);scores=base_scores(model,matrices)
    offset=fit_offset(items,matrices,scores,features)
    joint,jx,positions,partners=joint_candidates(items,matrices,features)
    pools=independent(joint,jx,base_scores(model,jx),features,offset)
    w,tau=fit_coupling(joint,pools,features);pools=couple(joint,pools,features,w,tau)
    decorated,pools=primary_decisions(items,pools,positions,partners)
    calibration=fit_confidence(decorated,pools,features);conf,_=confidence(calibration,decorated,pools,features)
    return dict(model=model,offset=offset,w=w,tau=tau,calibration=calibration,low_conf_threshold=float(np.quantile(conf,.05)),C=C,features=FEATURES,
                objective=getattr(features,"cfg",{}).get("decision",{}).get("objective","diagnosis"))


def apply_decision(model,items,matrices,features,return_aux=False):
    joint,jx,positions,partners=joint_candidates(items,matrices,features)
    pools=independent(joint,jx,base_scores(model["model"],jx),features,model["offset"])
    pools=couple(joint,pools,features,model["w"],model["tau"])
    joint_pools=pools
    decorated,pools=primary_decisions(items,pools,positions,partners)
    conf,missing=confidence(model["calibration"],decorated,pools,features)
    if return_aux:
        auxiliary=[None]*len(items)
        if partners:
            keys=sorted(partners)
            others=[dict(joint[partners[k]],coupling_partner=pools[k][0][0]) for k in keys]
            # `pools` now contains primaries; the joint result retains the partners.
            other_pools=[joint_pools[partners[k]] for k in keys]
            oc,om=confidence(model["calibration"],others,other_pools,features)
            for i,k in enumerate(keys):auxiliary[k]=dict(pool=other_pools[i],confidence=float(oc[i]),missing=float(om[i]))
        return pools,conf,missing,auxiliary
    return pools,conf,missing


def uncoupled_model(model,items,matrices,features):
    result=dict(model,w=0.)
    joint,jx,positions,partners=joint_candidates(items,matrices,features)
    pools=independent(joint,jx,base_scores(model["model"],jx),features,model["offset"])
    pools=couple(joint,pools,features,0.,model["tau"])
    decorated,pools=primary_decisions(items,pools,positions,partners)
    result["calibration"]=fit_confidence(decorated,pools,features)
    conf,_=confidence(result["calibration"],decorated,pools,features)
    result["low_conf_threshold"]=float(np.quantile(conf,.05))
    return result


def guarded_coupling(model,items,matrices,features,probe,probe_matrices,guard):
    without=uncoupled_model(model,items,matrices,features)
    if guard is None:return model,without,None
    on,_,_=apply_decision(model,items,matrices,features)
    off,_,_=apply_decision(without,items,matrices,features)
    pon,_,_=apply_decision(model,probe,probe_matrices,features)
    poff,_,_=apply_decision(without,probe,probe_matrices,features)
    check=guard(items,on,off,probe,pon,poff,features.freq)
    selected=model if check["keep"] else without
    return dict(selected,coupling_fit_check=check),without,check


def cross_fit(items,matrices,features,probe=None,probe_matrices=None,guard=None):
    objective=getattr(features,"cfg",{}).get("decision",{}).get("objective","diagnosis")
    groups=np.array([x["row"]["group_id"] for x in items]);folds=GroupKFold(5)
    output=[None]*len(items);conf=np.zeros(len(items));missing=np.zeros(len(items));thresholds=np.zeros(len(items));choices=[]
    without=dict(pools=[None]*len(items),confidence=np.zeros(len(items)),missing=np.zeros(len(items)),thresholds=np.zeros(len(items)),auxiliary=[None]*len(items))
    auxiliary=[None]*len(items);manifest=[]
    for fold,(train,test) in enumerate(folds.split(items,groups=groups)):
        inner_groups=groups[train];inner=GroupKFold(5);best=(-1,None)
        for C in [.1,1.,10.]:
            predictions=[None]*len(train)
            for it,iv in inner.split(train,groups=inner_groups):
                model=fit_lr(matrices,items,train[it],C)
                for j in iv:
                    i=train[j];s=model.decision_function(matrices[i]);k=min(range(len(s)),key=lambda k:(-round(float(s[k]),5),items[i]["pool"][k][0]));predictions[j]=items[i]["pool"][k][0]
            score=headline([items[i] for i in train],predictions,objective)
            if score>best[0]:best=(score,C)
        model=fit_decision([items[i] for i in train],[matrices[i] for i in train],features,best[1])
        model,no_coupling,check=guarded_coupling(model,[items[i] for i in train],[matrices[i] for i in train],features,probe,probe_matrices,guard)
        npools,nconf,nmissing,naux=apply_decision(no_coupling,[items[i] for i in test],[matrices[i] for i in test],features,return_aux=True)
        pools,p,m,aux=apply_decision(model,[items[i] for i in test],[matrices[i] for i in test],features,return_aux=True)
        train_groups=sorted(set(groups[train]));test_groups=sorted(set(groups[test]));assert not set(train_groups)&set(test_groups)
        manifest.append(dict(fold=fold+1,fit_groups=train_groups,evaluated_groups=test_groups,C=best[1],coupling_check=check,objective=objective,
                             low_conf_threshold=model["low_conf_threshold"],w=model["w"],tau=model["tau"]))
        for j,i in enumerate(test):output[i]=pools[j];conf[i]=p[j];missing[i]=m[j];thresholds[i]=model["low_conf_threshold"];auxiliary[i]=aux[j]
        for j,i in enumerate(test):
            without["pools"][i]=npools[j];without["confidence"][i]=nconf[j];without["missing"][i]=nmissing[j];without["thresholds"][i]=no_coupling["low_conf_threshold"]
            without["auxiliary"][i]=naux[j]
        choices.append(best[1]);print("Fusion fold",fold+1,"C",best[1],"offset",model["offset"],"coupling",model["w"],flush=True)
    # Final fit uses the modal inner-C choice; all reported dev predictions remain outer-fold held out.
    C=min(set(choices),key=lambda c:(-choices.count(c),c))
    model=fit_decision(items,matrices,features,C)
    model,without["model"],_=guarded_coupling(model,items,matrices,features,probe,probe_matrices,guard)
    model["cv_manifest"]=manifest
    return output,conf,missing,model,choices,thresholds,without,auxiliary
