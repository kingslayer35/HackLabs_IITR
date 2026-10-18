"""Reconstruct the seeded quizzes' gold-insertion manifest without clinical text."""
import json
from collections import Counter
import joblib
import pandas as pd
from termmap.common.io import load_config,path_for,term_dir,write_json,sha256
from termmap.preprocessing.normalize import clean
from termmap.terminology.hierarchy import strip_tag
from termmap.posttraining.trainer import sampled_groups


def main():
    cfg=load_config();base=path_for(cfg,"run_dir")/"C1"
    held=set(pd.read_parquet(term_dir(cfg)/"probe_set.parquet").heldout_norm)
    train=joblib.load(base/"train_pools.joblib")
    train=[x for x in train if clean(strip_tag(x["row"]["entity"])) not in held]
    aliases=joblib.load(base/"alias_pools.joblib");result={}
    for epoch in [1,2]:
        groups=sampled_groups(cfg,train,aliases,epoch)
        counts=Counter((x["row"]["system"],x["row"]["code"]) for x in groups)
        inserted=[dict(row_id=x["row"]["row_id"],system=x["row"]["system"],code=x["row"]["code"]) for x in groups if x["row"]["code"] not in dict(x["pool"])]
        alias_ids={x["row"]["row_id"] for x in aliases}
        result[str(epoch)]=dict(n_groups=len(groups),alias_groups=sum(x["row"]["row_id"] in alias_ids for x in groups),max_groups_per_code=max(counts.values()),gold_insertions=len(inserted),insertions=inserted)
    result["source_sha256"]={name:sha256(base/name) for name in ["train_pools.joblib","alias_pools.joblib"]}
    write_json(path_for(cfg,"run_dir")/"posttrain"/"gold_insertion_manifest.json",result)
    print(json.dumps({e:{k:v for k,v in r.items() if k!="insertions"} for e,r in result.items() if e!="source_sha256"}))


if __name__=="__main__":main()
