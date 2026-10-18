"""Use the frozen local scoring backend with final all-label artifacts."""
import atexit
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error

import torch

from termmap.common.io import path_for, sha256, write_json
from termmap.reranking.model import Cards
from termmap.reranking.vllm import request_json
from termmap.reranking.client import make_ranker


class HostedFinalRanker:
    """Own only the service spawned for this allocated prediction process."""
    def __init__(self, cfg, adapter=None, terminology=None):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Hosted prediction requires a Slurm allocation")
        self.server = self.ranker = self.log = None
        self.stopped = threading.Event()
        self.monitor = None
        self.peak_device_vram_bytes = 0
        self.cfg = copy.deepcopy(cfg)
        folder = path_for(cfg, "run_dir") / "hosted_predictions" / str(os.getpid())
        self.cfg["paths"]["run_dir"] = str(folder)
        dest = folder / "vllm_service"
        dest.mkdir(parents=True, exist_ok=True)
        settings = self.cfg["vllm"]
        if settings["host"] != "127.0.0.1":
            raise RuntimeError("Clinical inference must stay on the local machine")
        address = f"http://{settings['host']}:{settings['port']}"
        try:
            request_json(address + "/health", timeout=1)
            raise RuntimeError("The configured local prediction port is already occupied")
        except (urllib.error.URLError, TimeoutError):
            pass
        snapshot = json.loads((path_for(cfg, "external_dir") / "reranker_snapshot.json").read_text())
        # The service always registers the final adapter; the zero-shot flow
        # explicitly requests the base model and never activates that adapter.
        final_adapter = path_for(cfg, "model_dir") / cfg.get("deployment",{}).get("adapter_directory","reranker_final/best")
        if adapter is not None and adapter.resolve() != final_adapter.resolve():
            raise RuntimeError("The final prediction adapter differs from the requested model")
        base=Path(snapshot["path"])
        if not base.is_absolute():base=cfg["root"]/base
        manifest = dict(settings, base=str(base), adapter=str(final_adapter),
            adapter_sha256=sha256(final_adapter / "adapter_model.safetensors"),
            adapter_config_sha256=sha256(final_adapter / "adapter_config.json"),
            root=str(cfg["root"]), run_dir=str(folder))
        write_json(dest / "server.json", manifest)
        version_code = "import torch,importlib.metadata as m,json; print(json.dumps(dict(vllm_version=m.version('vllm'),torch_version=torch.__version__)))"
        versions = json.loads(subprocess.run([str(path_for(cfg, "vllm_python")), "-c", version_code],
            check=True, capture_output=True, text=True).stdout)
        if versions["vllm_version"] != settings["vllm_version"]:
            raise RuntimeError("The final serving environment differs from the frozen evaluation")
        runtime = dict(versions, gpu=torch.cuda.get_device_name(),
            architecture="sm" + "".join(map(str, torch.cuda.get_device_capability())),
            slurm_job_id=os.environ["SLURM_JOB_ID"], server_local_only=True)
        write_json(dest / "runtime.json", runtime)
        env = dict(os.environ, TERMMAP_VLLM_WORKER="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="13", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(cfg["root"] / "slurm/vllm_runtime"))
        self.log = (dest / "server.log").open("w")
        try:
            self.server = subprocess.Popen([str(path_for(cfg, "vllm_python")),
                str(cfg["root"] / "scripts/reranking/serve.py"), "--manifest", str(dest / "server.json")],
                cwd=cfg["root"], env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 140
            while True:
                if self.server.poll() is not None:
                    raise RuntimeError("Final local service failed; inspect its private server log")
                try:
                    models = request_json(address + "/v1/models", timeout=2)
                    if {settings["model_name"], settings["adapter_name"]} <= {r["id"] for r in models["data"]}:
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("Final local service exceeded its startup budget")
                time.sleep(1)
            cards = Cards(cfg, profile=cfg.get("deployment",{}).get("profile","final"), terminology=terminology)
            self.ranker = make_ranker(self.cfg, adapter=adapter is not None, cards=cards)
            self.monitor = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitor.start()
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def _monitor_memory(self):
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        if not uuid.startswith(("GPU-", "MIG-")):
            uuid = "GPU-" + uuid
        while not self.stopped.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--id=" + uuid,
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True)
                self.peak_device_vram_bytes = max(self.peak_device_vram_bytes, int(result.stdout.strip()) * 2**20)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stopped.wait(1)

    def score_items(self, items):
        # Determinism and deployment timing require real cold computations.
        return self.ranker.score_items(items, use_cache=False)

    def close(self):
        self.stopped.set()
        if self.monitor is not None:
            self.monitor.join(timeout=4)
            self.monitor = None
        if self.ranker is not None:
            self.ranker.close()
            self.ranker = None
        if self.server is not None and self.server.poll() is None:
            os.killpg(self.server.pid, signal.SIGTERM)
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.server.pid, signal.SIGKILL)
                self.server.wait()
        if self.log is not None:
            self.log.close()
            self.log = None
