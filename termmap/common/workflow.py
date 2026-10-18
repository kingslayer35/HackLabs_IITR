"""Local CPU steps and owned screen/Slurm GPU jobs, with smoke gates."""
import getpass
import json
import os
import shlex
import subprocess
import time

from termmap.common.io import path_for, write_json, sha256


class Workflow:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg["root"]
        self.runs = path_for(cfg, "run_dir")
        self.logs = path_for(cfg, "log_dir")
        self.state_file = self.runs / "workflow.json"
        self.logs.mkdir(parents=True, exist_ok=True)

    def state(self, **changes):
        value = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        value.update(changes)
        write_json(self.state_file, value)

    def command(self, args):
        return subprocess.run(args, cwd=self.root, check=True, capture_output=True, text=True).stdout

    def queue(self):
        rows = self.command(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i %t"])
        return dict(row.split() for row in rows.splitlines() if row.strip())

    def cpu(self, name, args, env=None):
        marker = self.runs / "steps" / (name + ".json")
        expected = dict(command=[str(x) for x in args], config_sha256=sha256(self.cfg["config_path"]))
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise RuntimeError("A completed CPU step changed: " + name)
            return
        self.state(stage=name, status="RUNNING")
        environment = dict(os.environ, TERMMAP_CONFIG=str(self.cfg["config_path"]),
                           OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                           PYTHONHASHSEED="13", TOKENIZERS_PARALLELISM="false")
        environment.update(env or {})
        print("CPU", name, flush=True)
        with (self.logs / ("updated_" + name + ".log")).open("w") as log:
            subprocess.run(args, cwd=self.root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(marker, expected)

    def stuff(self, screen, command):
        self.command(["screen", "-S", screen, "-X", "stuff", command + "\n"])

    def wait_status(self, run, timeout=None):
        status = self.runs / run / "STATUS"
        deadline = time.monotonic() + timeout if timeout else None
        job_file = self.runs / "jobs" / run.removesuffix("_smoke") / "JOB_ID"
        next_queue_check = time.monotonic() + 60
        while not status.exists():
            if deadline and time.monotonic() > deadline:
                raise RuntimeError("GPU smoke did not complete within its time budget: " + run)
            if time.monotonic() >= next_queue_check and job_file.exists():
                if job_file.read_text().strip() not in self.queue():
                    raise RuntimeError("Owned Slurm job ended without a completion marker: " + run)
                next_queue_check = time.monotonic() + 60
            time.sleep(3 if timeout else 15)
        if status.read_text().strip() != "DONE":
            raise RuntimeError("GPU step failed; inspect its log before any retry: " + run)

    def start_gpu(self, run, script, args=()):
        status = self.runs / run / "STATUS"
        if status.exists() and status.read_text().strip() == "DONE":
            return
        screen = "termmap_" + run
        listing = subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
        if "." + screen + "\t" not in listing:
            self.command(["screen", "-dmS", screen])
        job_file = self.runs / "jobs" / run / "JOB_ID"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        if job_file.exists() and job_file.read_text().strip() in self.queue():
            raise RuntimeError("This workflow already owns an active allocation for " + run)
        job_file.unlink(missing_ok=True)
        self.state(stage=run, status="ALLOCATING", screen=screen)
        self.stuff(screen, "srun --gres=gpu:1 --cpus-per-task=6 --mem=48G --pty sh -i")
        # A scalar ID is written by this screen's allocated shell. This avoids
        # confusing another session's new job with our own allocation.
        # Screen expands dollar expressions in commands passed to `stuff`.
        # `printenv` obtains only this scalar from the allocated shell itself.
        self.stuff(screen, "printenv SLURM_JOB_ID > " + shlex.quote(str(job_file)))
        deadline = time.monotonic() + 7200
        next_queue_check = 0
        while not job_file.exists() or not job_file.read_text().strip():
            if time.monotonic() > deadline:
                self.stuff(screen, "\x03")
                raise RuntimeError("HARD STOP: Slurm still unavailable after two hours")
            if time.monotonic() >= next_queue_check:
                self.queue()
                next_queue_check = time.monotonic() + 300
            time.sleep(2)
        job = job_file.read_text().strip()
        if not job.isdigit() or self.queue().get(job) != "R":
            raise RuntimeError("The owned Slurm allocation is not running")
        smoke = self.runs / (run + "_smoke") / "STATUS"
        smoke.unlink(missing_ok=True)
        status.unlink(missing_ok=True)
        invocation = shlex.join(["env", "TERMMAP_CONFIG=" + str(self.cfg["config_path"]), "bash", script, *args])
        prefix = "cd " + shlex.quote(str(self.root)) + " && "
        self.state(stage=run, status="SMOKE", allocation=job, screen=screen)
        self.stuff(screen, prefix + invocation + " --smoke 2>&1 | tee " + shlex.quote(str(self.logs / ("updated_" + run + "_smoke.log"))))
        try:
            self.wait_status(run + "_smoke", timeout=240)
        except Exception:
            self.stuff(screen, "exit")
            raise
        self.state(stage=run, status="RUNNING", allocation=job, screen=screen)
        self.stuff(screen, invocation + " 2>&1 | tee " + shlex.quote(str(self.logs / ("updated_" + run + "_full.log"))) + "; exit")

    def gpu(self, run, script, args=()):
        self.start_gpu(run, script, args)
        self.wait_status(run)
