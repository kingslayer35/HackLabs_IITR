"""Optional private model-only Hugging Face upload; never uploads datasets.

Run deliberately after setting HF_TOKEN locally. This is not invoked by setup,
inference, packaging or tests. No credential is logged or written to the repo.
"""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys


def verify_account(api, namespace, expected_user=None):
    expected=expected_user or namespace
    actual=api.whoami()["name"]
    if actual.casefold()!=expected.casefold():
        raise RuntimeError(f"Account mismatch: expected {expected}, authenticated as {actual}; nothing uploaded")
    return actual


def verify_files(api, repo_id, folder):
    info=api.model_info(repo_id,files_metadata=True)
    if not info.private:
        raise RuntimeError("Model repository must remain private")
    remote={item.rfilename:item for item in info.siblings if item.rfilename!=".gitattributes"}
    local={path.relative_to(folder).as_posix():path for path in folder.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(folder).parts
        and (path.suffix in {".safetensors",".json",".txt",".jinja"} or path.name=="README.md")
        and "dev_scores" not in str(path.relative_to(folder)) and "resume" not in str(path.relative_to(folder))}
    if set(remote)!=set(local):
        raise RuntimeError(f"Uploaded file inventory differs for {repo_id}")
    verified=[]
    for name,path in sorted(local.items()):
        size=path.stat().st_size
        sha256=hashlib.sha256()
        blob=hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b""):
                sha256.update(chunk);blob.update(chunk)
        item=remote[name]
        lfs=item.lfs
        expected=(lfs.get("sha256") if isinstance(lfs,dict) else lfs.sha256) if lfs else item.blob_id
        actual=sha256.hexdigest() if lfs else blob.hexdigest()
        if actual!=expected:
            raise RuntimeError(f"Uploaded checksum differs for {repo_id}/{name}")
        verified.append({"file":name,"bytes":size,"sha256":sha256.hexdigest()})
    return {"repo_id":repo_id,"url":"https://huggingface.co/"+repo_id,
        "revision":info.sha,"private":True,"source":folder.relative_to(folder.parents[1]).as_posix(),
        "verified_files":verified}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--namespace",required=True)
    parser.add_argument("--expected-user",help="Explicit token owner when uploading to an organization")
    parser.add_argument("--prompt-token",action="store_true",help="Read a token through a hidden terminal prompt instead of HF_TOKEN")
    args=parser.parse_args()
    if args.prompt_token:
        if not sys.stdin.isatty():parser.error("A terminal is required for the hidden token prompt")
        token=getpass.getpass("Hugging Face token (hidden): ").strip()
        os.environ["HF_TOKEN"]=token
    else:token=os.environ.get("HF_TOKEN")
    if not token:parser.error("Set HF_TOKEN in your local environment")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
    from huggingface_hub import HfApi
    api=HfApi(token=token)
    owner=verify_account(api,args.namespace,args.expected_user)
    print("Verified token owner:",owner,"target namespace:",args.namespace,flush=True)
    root=Path(__file__).resolve().parents[1]/"private_assets/models"
    models={"termmap-sapbert-icd":root/"sapbert_v2/best_ICD",
        "termmap-sapbert-snomed":root/"sapbert_v2/best_SNOMED",
        "termmap-qwen-lora-epoch1":root/"reranker_lora/epoch_1",
        "termmap-qwen-lora-epoch2":root/"reranker_lora/epoch_2"}
    verified=[]
    for name,path in models.items():
        repo_id=args.namespace+"/"+name
        api.create_repo(repo_id,repo_type="model",private=True,exist_ok=True)
        if not api.model_info(repo_id).private:
            raise RuntimeError("Refusing to upload challenge-derived weights to a public model repository")
        api.upload_folder(repo_id=repo_id,repo_type="model",folder_path=str(path),
            allow_patterns=["*.safetensors","*.json","*.txt","*.jinja","README.md"],
            ignore_patterns=["*dev_scores*","*resume*","**/.cache/**"],commit_message="Upload fixed TermMap checkpoint")
        verified.append(verify_files(api,repo_id,path))
        print("Uploaded and verified private model:","https://huggingface.co/"+repo_id,flush=True)
    manifest={"verified_utc":datetime.now(timezone.utc).isoformat(),
        "credential_source":"Explicit hidden prompt; credential not recorded" if args.prompt_token else "Local environment; credential not recorded",
        "datasets_uploaded":False,"models":verified}
    target=root.parents[1]/"docs/MODEL_UPLOADS.json"
    target.write_text(json.dumps(manifest,indent=2)+"\n")
    print("All four private repositories verified; manifest:",target,flush=True)


if __name__=="__main__":main()
