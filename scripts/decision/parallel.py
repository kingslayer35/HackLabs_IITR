"""Fit independent complete flows as their scores become ready, on CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from termmap.common.io import load_config,path_for,write_json


def ready(root,flow,score_job):
    if not all((root/flow/(ds+"_timing.json")).exists() for ds in ["dev","probe"]):return False
    if score_job=="updated_flows":return True  # Original live job predates per-flow allocation markers.
    job=root/"jobs"/score_job/"JOB_ID";marker=root/flow/"SCORE_COMPLETE.json"
    return job.exists() and marker.exists() and json.loads(marker.read_text())["slurm_job_id"]==job.read_text().strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--watch-config")
    parser.add_argument("--flows",nargs="+",default=["F00","F10","F01","F11"])
    parser.add_argument("--score-job",default="updated_flows")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.watch_config:
        while not Path(args.watch_config).exists():time.sleep(10)
    cfg=load_config(args.watch_config);root=path_for(cfg,"run_dir");logs=path_for(cfg,"log_dir")
    env=dict(os.environ,TERMMAP_CONFIG=str(cfg["config_path"]),OMP_NUM_THREADS="6",OPENBLAS_NUM_THREADS="6",MKL_NUM_THREADS="6",PYTHONHASHSEED="13")
    pending=list(args.flows);running={};done=[]
    while pending or running:
        for flow,(process,handle) in list(running.items()):
            result=process.poll()
            if result is None:continue
            handle.close();del running[flow]
            if result:raise RuntimeError("Independent decision worker failed: "+flow)
            done.append(flow);print("Parallel decision complete",flow,flush=True)
        for flow in list(pending):
            if len(running)>=args.workers:break
            if not ready(root,flow,args.score_job):continue
            pending.remove(flow)
            log=logs/("parallel_"+root.name+"_"+flow+"_fit.log")
            handle=log.open("w")
            command=[sys.executable,"-u","-m","scripts.decision.fit","--flows",flow,"--no-freeze"]
            running[flow]=(subprocess.Popen(command,cwd=cfg["root"],env=env,stdout=handle,stderr=subprocess.STDOUT),handle)
            print("Parallel decision started",flow,flush=True)
        write_json(root/"parallel_decisions/STATUS.json",dict(running=list(running),waiting=pending,done=done,workers=args.workers))
        if pending or running:time.sleep(5)


if __name__=="__main__":main()
