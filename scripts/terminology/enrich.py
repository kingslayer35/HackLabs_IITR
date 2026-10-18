import json
from collections import Counter
import pandas as pd
from termmap.common.io import load_config,term_dir,path_for,read_partition,write_json
from termmap.terminology.structure import enrich_icd,snomed_links,family_index,save_sqlite

def main():
    cfg=load_config();out=term_dir(cfg);train=read_partition(cfg,"train")
    icd=pd.read_parquet(out/"icd_concepts.parquet")
    cdc=path_for(cfg,"external_dir")/cfg["resources"]["cdc_subdir"]/"xml"
    icd,seven,stats=enrich_icd(icd,next(cdc.rglob("*tabular*.xml")),train)
    icd.to_parquet(out/"icd_concepts.parquet",index=False);seven.to_parquet(out/"icd_seventh_def.parquet",index=False)
    print("ICD v2.1",stats,flush=True)
    sn=pd.read_parquet(out/"snomed_concepts.parquet")
    links=snomed_links(sn);links.to_parquet(out/"snomed_links.parquet",index=False)
    sn,families=family_index(sn,links)
    sn.to_parquet(out/"snomed_concepts.parquet",index=False);families.to_parquet(out/"snomed_families.parquet",index=False)
    multi=families.loc[families["size"].gt(1)]
    stats["links"]=links.kind.value_counts().to_dict();stats["families"]={"nontrivial":len(multi),"concepts":int(multi["size"].sum()),"median":float(multi["size"].median()),"max":int(multi["size"].max())}
    aliases=pd.read_parquet(out/"aliases.parquet")
    save_sqlite(out/"terminology.sqlite",icd,sn,links,families,seven,aliases)
    gold=train.loc[train.system.eq("ICD")&train.valid_gold,["code"]].merge(icd[["code","category","seventh_char","seventh_char_meaning"]],on="code")
    gold=gold.loc[gold.seventh_char_meaning.ne("")]
    prior=gold.groupby(["category","seventh_char"]).size().rename("count").reset_index()
    prior["probability"]=prior["count"]/prior.groupby("category")["count"].transform("sum")
    prior.to_parquet(out/"seventh_priors_train.parquet",index=False)
    icd.loc[icd.legacy_gold,["code","header_train_count","descendant_train_count"]].to_parquet(out/"header_exceptions_train.parquet",index=False)
    write_json(out/"v21_structure_checks.json",stats)
    print("SNOMED v2.1",stats["links"],stats["families"],flush=True)

if __name__=="__main__":main()
