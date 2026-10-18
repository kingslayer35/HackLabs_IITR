"""Copy prepared current-generation pools into an isolated training run."""
import shutil
import joblib
from termmap.common.io import load_config,path_for,sha256,write_json


def main():
    cfg=load_config();source=path_for(cfg,"retrieval_run_dir")/"C1"
    target=path_for(cfg,"run_dir")/"C1";target.mkdir(parents=True,exist_ok=True)
    hashes={}
    for dataset in ["train","alias","dev"]:
        filename=dataset+"_pools.joblib";items=joblib.load(source/filename)
        if dataset in {"train","alias"}:
            field="excluded_group" if dataset=="train" else "excluded_alias"
            if any(not item["row"].get(field) for item in items):
                raise RuntimeError("Missing leakage exclusion in prepared training pools")
        shutil.copyfile(source/filename,target/filename);hashes[filename]=sha256(target/filename)
    write_json(target/"PREPARED.json",dict(source="original-SapBERT pools",files=hashes,test_read=False))
    print("Isolated original-SapBERT training pools prepared")


if __name__=="__main__":main()
