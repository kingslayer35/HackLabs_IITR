"""Single saved grouped split and dictionary-derived holdout probes."""
from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from termmap.preprocessing.noise import corrupt


def grouped_split(frame: pd.DataFrame, threshold: float=.9, seed: int=13):
    chunks = frame.chunk_text.unique().tolist()
    ids = {t:i for i,t in enumerate(chunks)}
    parent = list(range(len(chunks)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        parent[max(a,b)] = min(a,b)
    vectors = TfidfVectorizer(analyzer="char", ngram_range=(3,4), dtype=np.float32).fit_transform(chunks)
    matches = 0
    for start in range(0,len(chunks),128):
        scores = (vectors[start:start+128] @ vectors.T).tocoo()
        for i,j,s in zip(scores.row+start, scores.col, scores.data):
            if i<j and s>=threshold:
                union(i,j); matches+=1
    groups = np.array([find(ids[t]) for t in frame.chunk_text])
    folds = np.zeros(len(frame),dtype=np.int32)
    with warnings.catch_warnings(record=True) as ws:
        splitter = StratifiedGroupKFold(n_splits=20,shuffle=True,random_state=seed)
        for fold, (_, heldout) in enumerate(splitter.split(np.zeros(len(frame)),frame.subconcept,groups)):
            folds[heldout] = fold
    split = np.where(folds<14,"train",np.where(folds<17,"dev","test"))
    result = frame[["row_id"]].copy()
    result["group_id"] = [str(g) for g in groups]
    result["split"] = split
    result["fold"] = folds
    return result, dict(exact_chunks=len(chunks), merged_groups=len(set(groups)),
                        near_duplicate_edges=matches, split_rows=pd.Series(split).value_counts().to_dict(),
                        stratification_notes=[str(w.message) for w in ws])


def make_probe(aliases, icd, snomed, all_gold, size=1500, seed=13):
    rng = np.random.default_rng(seed)
    strata = {"ICD": icd.set_index("code").chapter.to_dict(), "SNOMED": snomed.set_index("code").tag.to_dict()}
    preferred = {"ICD":icd.set_index("code").preferred.to_dict(), "SNOMED":snomed.set_index("code").preferred.to_dict()}
    from termmap.preprocessing.normalize import clean
    probes=[]
    for system in ["ICD","SNOMED"]:
        buckets=defaultdict(list)
        names=defaultdict(set)
        excluded_codes=set(all_gold[system])
        for row in aliases.loc[aliases.system.eq(system)].itertuples(index=False):
            if row.code not in excluded_codes:
                names[row.code].add(row.norm_text)
        for code, variants in sorted(names.items()):
            if len(variants)<2:continue
            choices=sorted(variants-{clean(preferred[system][code])})
            if choices:buckets[strata[system][code]].append((code,choices))
        selected=[]
        # Proportional stratified allocation; one seeded choice per sampled code.
        population=sum(map(len,buckets.values()))
        quota={s:int(size*len(v)/population) for s,v in buckets.items()}
        remainder=size-sum(quota.values())
        order=sorted(buckets,key=lambda s:(-(size*len(buckets[s])/population-quota[s]),s))
        for s in order[:remainder]: quota[s]+=1
        for s, values in sorted(buckets.items()):
            chosen=rng.choice(len(values),size=quota[s],replace=False)
            for i in chosen:
                code,choices=values[i]
                alias=choices[rng.integers(len(choices))]
                selected.append((code,s,alias,alias))
        if len(selected)!=size:
            raise ValueError("Insufficient eligible dictionary probe codes")
        for code,s,text,norm in selected:
            noisy,kind=corrupt(norm,seed=seed,key=f"probe:{system}:{code}")
            probes.extend([dict(system=system,code=code,stratum=s,text=norm,heldout_norm=norm,variant="clean",noise="none"),
                           dict(system=system,code=code,stratum=s,text=noisy,heldout_norm=norm,variant="noisy",noise=kind)])
    return pd.DataFrame(probes)
