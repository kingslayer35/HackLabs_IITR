"""Preserve dataset-specific GPU audits when the service is allocated again."""
import json
from pathlib import Path
import shutil

from termmap.common.io import path_for, sha256, write_json


def capture(cfg, dataset="dev", smoke=False, service_dir=None):
    service = Path(service_dir) if service_dir is not None else path_for(cfg,"run_dir")/"vllm_service"
    runtime = json.loads((service/"runtime.json").read_text())
    label = dataset + ("_smoke" if smoke else "")
    target = service/"evidence"/label/runtime["slurm_job_id"]
    target.mkdir(parents=True,exist_ok=True)
    files = {}
    names=["runtime.json","server.json","SMOKE.json"]
    if not smoke:names.append("BATCH_AUDIT.json")
    for name in names:
        source=service/name
        if not source.exists():continue
        dest=target/name
        if dest.exists() and sha256(dest)!=sha256(source):
            raise RuntimeError("An immutable serving audit would change: "+name)
        if not dest.exists():shutil.copyfile(source,dest)
        files[name]=dict(path=str(dest),sha256=sha256(dest))
    result=dict(dataset=dataset,smoke=smoke,slurm_job_id=runtime["slurm_job_id"],
                gpu=runtime["gpu"],architecture=runtime["architecture"],files=files)
    write_json(target/"manifest.json",result)
    if dataset=="dev" and not smoke:
        pointer=service/"DEV_EVIDENCE.json"
        if pointer.exists() and json.loads(pointer.read_text())!=result:
            raise RuntimeError("Cannot replace the frozen DEV serving evidence")
        write_json(pointer,result)
    return result
