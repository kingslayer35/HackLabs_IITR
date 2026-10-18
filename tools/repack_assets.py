"""Refresh the local private ZIP after an approved handoff; no source reads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(2**20),b""):value.update(part)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];private=root/"private_assets"
    manifest=private/"MANIFEST.json";value=json.loads(manifest.read_text())
    files=[p for p in sorted(private.rglob("*")) if p.is_file()]
    forbidden={"sealed_test.parquet","hackathon_train.csv","resume.pt"}
    if any(p.name in forbidden or p.suffix in {".sqlite",".log"} for p in files):
        raise RuntimeError("Refusing to package a sealed/raw dataset, optimizer state or log/cache")
    value["files"]=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in files if p!=manifest]
    manifest.write_text(json.dumps(value,indent=2)+"\n")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".zip.partial")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path in files:archive.write(path,path.relative_to(root))
    temp.replace(args.output)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:raise RuntimeError("ZIP integrity verification failed")
    digest=sha(args.output)
    args.output.with_suffix(".zip.sha256").write_text(digest+"  "+args.output.name+"\n")
    print("Verified private ZIP:",args.output,"bytes",args.output.stat().st_size,"sha256",digest)


if __name__=="__main__":main()
