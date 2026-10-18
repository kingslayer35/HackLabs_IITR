"""Build and audit P0 data. This command never reads a sealed partition."""
import argparse
import json
from pathlib import Path

import pandas as pd

from termmap.common.io import data_path, load_config, path_for, read_strings, term_dir, write_json
from termmap.terminology.hierarchy import build_icd, build_snomed, build_snomed_links
from termmap.terminology.tables import aliases_for, index_aliases, clean_training, abbreviation_table, abbreviation_alias_support, make_priors
from termmap.evaluation.splits import grouped_split, make_probe


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/default.yaml")
    cfg=load_config(ap.parse_args().config); out=term_dir(cfg)
    cdc=path_for(cfg,"external_dir")/cfg["resources"]["cdc_subdir"]/"xml"
    if not (out/"icd_concepts.parquet").exists():
        raw=read_strings(data_path(cfg,"icd_file"))
        tabular=next(cdc.rglob("*tabular*.xml"))
        icd,extra,checks=build_icd(raw,tabular)
        extra+=index_aliases(next(cdc.rglob("*index*.xml")),set(icd.code))
        icd.to_parquet(out/"icd_concepts.parquet",index=False)
        pd.DataFrame(extra).to_parquet(out/"cdc_aliases.parquet",index=False)
        write_json(out/"icd_checks.json",checks)
        print("ICD",checks,flush=True)
    icd=pd.read_parquet(out/"icd_concepts.parquet")
    import pyarrow.parquet as pq
    migrate_snomed=(out/"snomed_concepts.parquet").exists() and "top_level" not in pq.read_schema(out/"snomed_concepts.parquet").names
    if not (out/"snomed_concepts.parquet").exists() or migrate_snomed:
        raw=read_strings(data_path(cfg,"snomed_file"),cfg["resources"]["snomed_sheet"])
        snomed,checks=build_snomed(raw)
        snomed.to_parquet(out/"snomed_concepts.parquet",index=False)
        write_json(out/"snomed_checks.json",checks)
        if migrate_snomed:
            (out/"snomed_links.parquet").unlink(missing_ok=True)
            (out/"aliases.parquet").unlink(missing_ok=True)
        print("SNOMED",checks,flush=True)
    snomed=pd.read_parquet(out/"snomed_concepts.parquet")
    if not (out/"snomed_links.parquet").exists():
        write_json(out/"link_checks.json",build_snomed_links(snomed,out/"snomed_links.parquet"))
        print("SNOMED links built",flush=True)
    if not (out/"splits.parquet").exists():
        training=clean_training(read_strings(data_path(cfg,"train_file")),icd,snomed)
        splits,checks=grouped_split(training,cfg["split"]["near_duplicate_cosine"],cfg["seed"])
        splits.to_parquet(out/"splits.parquet",index=False)
        training=training.merge(splits,on="row_id",validate="one_to_one")
        # Only the construction stage writes sealed data; downstream loaders reject it.
        training.to_parquet(out/"train_clean.parquet",index=False)
        for split in ["train","dev","test"]:
            part=training.loc[training.split.eq(split)]
            part.to_parquet(out/("sealed_test.parquet" if split=="test" else f"{split}.parquet"),index=False)
        all_gold={s:sorted(set(training.loc[training.system.eq(s),"code"])) for s in ["ICD","SNOMED"]}
        write_json(out/"all_gold_codes.json",all_gold)
        checks["system_subconcept_counts"]={f"{s}/{t}":int(n) for (s,t),n in training.groupby(["system","subconcept"]).size().items()}
        checks["gold_coverage"]={s:float(training.loc[training.system.eq(s),"valid_gold"].mean()) for s in ["ICD","SNOMED"]}
        checks["invalid_gold"]={s:sorted(set(training.loc[training.system.eq(s)&~training.valid_gold,"code"])) for s in ["ICD","SNOMED"]}
        write_json(out/"split_checks.json",checks)
        print("Splits",checks,flush=True)
    train=pd.read_parquet(out/"train.parquet")
    if not (out/"aliases.parquet").exists():
        extra=pd.read_parquet(out/"cdc_aliases.parquet").to_dict("records")
        aliases=aliases_for(icd,snomed,extra)
        gold=json.loads((out/"all_gold_codes.json").read_text())
        if (out/"probe_set.parquet").exists():
            probe=pd.read_parquet(out/"probe_set.parquet")
        else:
            probe=make_probe(aliases,icd,snomed,gold,cfg["probe"]["codes_per_system"],cfg["seed"])
            probe.to_parquet(out/"probe_set.parquet",index=False)
        excluded=set(probe.heldout_norm)
        aliases=aliases.loc[~aliases.norm_text.isin(excluded)].reset_index(drop=True)
        ext=path_for(cfg,"external_dir")/cfg["resources"]["abbreviation_subdir"]
        abbreviations=abbreviation_table(aliases,next(ext.glob("*.csv")),train)
        abbreviations=abbreviations.loc[~abbreviations.expansion.isin(excluded)]
        unambiguous=abbreviations.loc[~abbreviations.ambiguous].drop_duplicates(["abbr","expansion"])
        # Attach only expansions actually present in a concept; no cross-code positives.
        variants=unambiguous.merge(aliases,left_on="expansion",right_on="norm_text")
        variants=variants[["system","code","abbr"]].rename(columns={"abbr":"text"})
        variants["norm_text"]=variants.text;variants["source"]="abbr_variant";variants["origin"]="derived";variants["weight"]=1.0
        aliases=pd.concat([aliases,variants],ignore_index=True).drop_duplicates(["system","code","norm_text","source"])
        aliases=aliases.loc[~aliases.norm_text.isin(excluded)]
        aliases.to_parquet(out/"aliases.parquet",index=False)
        abbreviations.to_parquet(out/"abbreviations.parquet",index=False)
        abbreviation_alias_support(aliases,abbreviations,train).to_parquet(out/"abbreviation_alias_support_train.parquet",index=False)
        print("Aliases",len(aliases),"abbreviation entries",len(abbreviations),flush=True)
    aliases=pd.read_parquet(out/"aliases.parquet")
    tables,legacy=make_priors(train,snomed,icd,aliases)
    excluded=set(pd.read_parquet(out/"probe_set.parquet").heldout_norm)
    for name,table in tables.items():
        if "norm_text" in table:
            table=table.loc[~table.norm_text.isin(excluded)]
        table.to_parquet(out/f"{name}.parquet",index=False)
    icd["legacy_gold"]=icd.code.isin(legacy)
    icd.to_parquet(out/"icd_concepts.parquet",index=False)
    print("Training-only priors built",flush=True)


if __name__=="__main__":main()
