"""Public CPU launcher: obtain an owned screen/Slurm allocation for inference."""
import argparse
import os
from pathlib import Path
from termmap.common.io import load_config,path_for
from termmap.common.workflow import Workflow
from termmap.common.release import verify_release

def main():
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("output");a=p.parse_args();cfg=load_config()
    source=Path(a.input).resolve();output=Path(a.output).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    if output==source or path_for(cfg,"data_dir").resolve() in output.parents:raise ValueError("Output must not overwrite input or the read-only data directory")
    verify_release(cfg)
    run=f"predict_{os.getpid()}"
    Workflow(cfg).gpu(run,"slurm/run_predict.sh",[str(source),str(output),run])
    print(str(output));print(str(output.with_name("diagnostics.jsonl")))

if __name__=="__main__":main()
