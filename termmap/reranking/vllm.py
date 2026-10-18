"""Concurrent local vLLM requests using the reference reranker's token IDs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer

from termmap.common.io import path_for, sha256, write_json
from termmap.reranking.prompts import pack_payload
from termmap.reranking.model import Cards, PREFIX, SUFFIX


def raw_margin(values):
    if len(values) != 2 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected finite raw [no, yes] logits")
    return round(float(values[1]) - float(values[0]), 5)


def request_json(url, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read()
        # Do not put token IDs, clinical inputs or server-reflected text in logs.
        raise RuntimeError(f"Local vLLM HTTP {error.code}; response sha256="
                           + hashlib.sha256(detail).hexdigest()) from None


class VllmRanker:
    def __init__(self, cfg, adapter=False, cards=None):
        self.cfg = cfg
        self.settings = cfg["vllm"]
        self.runs = path_for(cfg, "run_dir")
        server = json.loads((self.runs / "vllm_service/server.json").read_text())
        runtime = json.loads((self.runs / "vllm_service/runtime.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(server["base"], padding_side="left", local_files_only=True)
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.cards = cards if cards is not None else Cards(cfg)
        self.adapter = adapter
        self.model_name = self.settings["adapter_name" if adapter else "model_name"]
        self.url = f"http://{self.settings['host']}:{self.settings['port']}"
        if self.settings["host"] != "127.0.0.1":
            raise RuntimeError("Clinical inference must use the local-only endpoint")
        snapshot = Path(server["base"])
        signature = dict(backend="vllm", version=runtime["vllm_version"],
            torch_version=runtime["torch_version"], gpu_architecture=runtime["architecture"],
            base_weights_sha256=hashlib.sha256("".join(sha256(p) for p in sorted(snapshot.glob("*.safetensors"))).encode()).hexdigest(),
            adapter_weights_sha256=server["adapter_sha256"] if adapter else None,
            adapter_config_sha256=server["adapter_config_sha256"] if adapter else None,
            implementation_sha256=sha256(Path(__file__)),
            server_implementation_sha256=sha256(cfg["root"] / "scripts/reranking/serve.py"),
            attention_backend=self.settings["attention_backend"],
            template_sha256=sha256(Path(__file__).with_name("model.py")),
            packing_sha256=sha256(Path(__file__).with_name("prompts.py")),
            tokenizer_sha256={name: sha256(snapshot/name) for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] if (snapshot/name).exists()},
            prefix=PREFIX, suffix=SUFFIX, max_tokens=512, raw_head_tokens=["no", "yes"],
            head_dtype="bfloat16", use_activation=False, batch_invariant=True,
            max_num_seqs=self.settings["max_num_seqs"],
            max_num_batched_tokens=self.settings["max_num_batched_tokens"])
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        self.identity = ("lora" if adapter else "zero") + "_vllm_" + runtime["architecture"] + "_" + digest
        folder = self.runs / "reranker_cache"; folder.mkdir(exist_ok=True)
        write_json(folder / (self.identity + ".json"), signature)
        self.connection = sqlite3.connect(folder / (self.identity + ".sqlite"))
        self.connection.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
        self.executor = ThreadPoolExecutor(max_workers=self.settings["workers"])
        self.last_timing = {}

    def token_ids(self, payloads):
        budget = 512 - len(self.prefix) - len(self.suffix)
        packed = [pack_payload(self.tokenizer, text, budget) for text in payloads]
        encoded = self.tokenizer(packed, padding=False, truncation=True,
                                 max_length=budget, add_special_tokens=False)["input_ids"]
        return [self.prefix + row + self.suffix for row in encoded]

    def request(self, token_lists):
        response = request_json(self.url + "/classify", dict(model=self.model_name,
            input=token_lists, add_special_tokens=False, use_activation=False))
        data = sorted(response["data"], key=lambda x: x["index"])
        if len(data) != len(token_lists): raise RuntimeError("vLLM returned the wrong row count")
        if response["usage"]["prompt_tokens"] != sum(map(len, token_lists)):
            raise RuntimeError("vLLM changed the supplied token sequences")
        return [raw_margin(row["probs"]) for row in data]

    def score(self, payloads, use_cache=True):
        tokens = self.token_ids(payloads)
        keys = [hashlib.sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest() for row in tokens]
        found = {}
        if use_cache:
            unique = list(dict.fromkeys(keys))
            for start in range(0, len(unique), 500):
                selected = unique[start:start+500]
                found.update(self.connection.execute(
                    "SELECT key,score FROM scores WHERE key IN (" + ",".join("?" for _ in selected) + ")", selected).fetchall())
        missing = {}
        for key, row in zip(keys, tokens):
            if key not in found: missing.setdefault(key, row)
        pending = list(missing.items())
        batch = self.settings["request_batch"]
        parts = [pending[i:i+batch] for i in range(0, len(pending), batch)]
        futures = [self.executor.submit(self.request, [row for _, row in part]) for part in parts]
        inserts = []
        for part, future in zip(parts, futures):
            for (key, _), score in zip(part, future.result()):
                found[key] = score; inserts.append((key, score))
        if use_cache and inserts:
            self.connection.executemany("INSERT OR REPLACE INTO scores VALUES (?,?)", inserts)
            self.connection.commit()
        return [found[key] for key in keys]

    def score_items(self, items, seen=None, use_cache=True):
        seen = self.cfg["reranker"].get("seen_enabled", True) if seen is None else seen
        outputs = []
        chunk = self.settings["item_batch"]
        started = time.perf_counter()
        for begin in range(0, len(items), chunk):
            tick = time.perf_counter()
            selected = [copy.copy(item) for item in items[begin:begin+chunk]]
            primary_ids = {id(item) for item in selected}
            targets = []
            for item in selected:
                targets.append(item)
                if "other_system" in item:
                    item["other_system"] = copy.copy(item["other_system"])
                    targets.append(item["other_system"])
            payloads = []
            ranges = []
            for item in targets:
                codes = [c for c, _ in item["pool"]]
                ranges.append((item, codes, len(payloads)))
                payloads.extend(self.cards.payload(item["row"], code, seen=seen) for code in codes)
            scores = self.score(payloads, use_cache=use_cache)
            seconds = (time.perf_counter()-tick) / len(selected)
            for item, codes, start in ranges:
                item["reranker_scores"] = dict(zip(codes, scores[start:start+len(codes)]))
                item["rerank_seconds"] = seconds if id(item) in primary_ids else 0.0
            outputs.extend(selected)
            print("vLLM", self.identity, min(begin+chunk, len(items)), "/", len(items), flush=True)
        self.last_timing = dict(seconds=time.perf_counter()-started, n=len(items),
            workers=self.settings["workers"], request_batch=self.settings["request_batch"],
            definition="Local vLLM concurrent offline batches; amortized time per input including both systems. Not single-request or cold-start latency.")
        return outputs

    def close(self):
        self.executor.shutdown(wait=True)
        self.connection.close()
