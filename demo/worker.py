"""Single warm GPU worker; runs only inside its own Slurm allocation."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import threading
import time
import traceback

from .store import JobStore, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--index-root")
    parser.add_argument("--frozen-alias-root")
    parser.add_argument("--vllm-python")
    parser.add_argument("--idle-timeout", type=int, default=1800)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Start the inference worker through Slurm.")
    os.umask(0o077)
    store = JobStore(args.runtime)
    lock = (store.root / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store.recover()
    state = dict(status="loading", message="Loading the released F11 models.", job_id=os.environ["SLURM_JOB_ID"])
    stopped = threading.Event()

    def heartbeat():
        while not stopped.is_set():
            write_json(store.root / "worker.json", dict(state, heartbeat=time.time()))
            stopped.wait(3)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    pipeline = None
    try:
        from .pipeline_adapter import Pipeline
        pipeline = Pipeline(args.pipeline_root, args.runtime, args.index_root, args.frozen_alias_root, args.vllm_python)
        state.update(status="ready", message="Ready for local inference.")
        last_activity = time.monotonic()
        while time.monotonic() - last_activity < args.idle_timeout:
            job = store.claim()
            if job is None:
                stopped.wait(0.5)
                continue
            state.update(status="busy", message="Mapping clinical terms.")
            try:
                elapsed = pipeline.predict(store.input_rows(job["id"]), store.folder(job["id"]),
                                           lambda stage, n: store.progress(job["id"], stage, n))
                store.finish(job["id"], "completed", elapsed=elapsed)
            except InterruptedError:
                store.finish(job["id"], "cancelled")
            except Exception:
                traceback.print_exc()
                store.finish(job["id"], "failed", error="The mapping run failed. Your file is retained; upload it again to retry. The operator can check the private worker log.")
            state.update(status="ready", message="Ready for local inference.")
            last_activity = time.monotonic()
    except BaseException:
        state.update(status="failed", message="Model startup failed. The operator can check the private worker log.")
        traceback.print_exc()
        raise
    finally:
        if pipeline is not None:
            pipeline.close()
        stopped.set()
        thread.join(timeout=5)
        if state["status"] != "failed":
            state.update(status="offline", message="Inference worker stopped. Start a new allocation to continue.")
        write_json(store.root / "worker.json", dict(state, heartbeat=time.time()))


if __name__ == "__main__":
    main()
