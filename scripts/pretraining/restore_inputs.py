"""Restore pinned textual training inputs; no score/index cache is distributed."""
import json
import shutil
import joblib
from termmap.common.io import load_config, path_for
from termmap.retrieval.cache import directory
from termmap.pretraining.trainer import pinned


def main():
    cfg=load_config()
    source=path_for(cfg,"training_dataset_dir")/"pretraining"
    policy=path_for(cfg,"run_dir")/"pretrain_v2/retrieval_policy_C1.json"
    policy.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/"retrieval_policy_C1.json",policy)
    cfg=pinned(cfg)
    target=directory(cfg)
    records={}
    for dataset in ["train","dev","probe","alias"]:
        path=source/f"reference_{dataset}_records.jsonl"
        records[dataset]=[json.loads(line) for line in path.read_text().splitlines()]
    joblib.dump(records,target/"records.joblib")
    (target/"queries.json").write_bytes((source/"reference_query_texts.json").read_bytes())
    print("Restored pinned text inputs; dev/probe remain monitoring-only")


if __name__=="__main__":main()
