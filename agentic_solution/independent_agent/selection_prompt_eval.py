"""Prompt-only example-equivalence experiment on the existing 80 inputs."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import shutil
import sqlite3
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import FINISH_SCHEMA, OpenRouterClient, validate_finish
from .common import ARTIFACTS, SEED, SYSTEMS, digest, read_json, write_json
from .comparisons import PolicyTools, ReservedBudget, TASK_PROMPT, assert_flat, paired_delta, schemas_for
from .recovery_eval import BASE_POLICY

EXPERIMENT = ARTIFACTS / "selection_prompt_v1"
SOURCE = ARTIFACTS / "recovery_v1"
MODEL = "deepseek/deepseek-v4-flash-0731"
PROVIDER = "DeepInfra"
MODES = ("baseline", "example_equivalence")
MAX_TURNS = 5
SELECTION_CHECK = """Before using a retrieved training example's code as the deciding evidence for your final choice,
check whether its mention and context express the same concept, modifiers and status as the target.
Similar wording alone is insufficient. In the short final evidence statement, identify the decisive
target evidence, including any meaningful detail your selected candidate adds or discards.
A similar example is supporting evidence, not an override for a different target meaning.
This final selection check requires no additional model response or tool call."""


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    manifest_path = EXPERIMENT / "prepared.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    paths = [SOURCE / name for name in ("inputs.json", "labels.json", "examples.sqlite", "leakage_audit.json", "baseline_predictions.json")]
    paths += sorted((SOURCE / "initial_evidence").glob("*.json"))
    paths += sorted((SOURCE / "runs" / "baseline").glob("*.json"))
    for path in paths:
        target = EXPERIMENT / path.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    items = read_json(EXPERIMENT / "inputs.json")
    if len(items) != 80 or len(list((EXPERIMENT / "runs" / "baseline").glob("*.json"))) != 80:
        raise RuntimeError("Require exactly the original 80 inputs and saved baseline traces")
    retained = {str(p.relative_to(SOURCE)): file_hash(p) for p in paths}
    for relative, expected in retained.items():
        assert file_hash(EXPERIMENT / relative) == expected
    write_json(EXPERIMENT / "retained_inputs_and_baseline.json", retained)
    manifest = {"count": 80, "model": MODEL, "provider": PROVIDER, "modes": MODES,
                "source_experiment": str(SOURCE), "baseline_reused_without_API_calls": 80,
                "status": "Previously inspected development set, reused at user's request",
                "intervention": "Append one generic final example-equivalence check to the baseline system prompt; no other inference changes",
                "max_model_responses": MAX_TURNS}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


class GuardedClient(OpenRouterClient):
    """Reserve a conservative per-request token-price bound before spending."""
    def complete(self, messages, tools, force_finish=False, force_tool=None):
        # UTF-8 bytes upper-bound ordinary text tokens conservatively; padding
        # covers chat framing, schemas and request overhead. Prices exceed the
        # verified pinned-provider prices for both selected models.
        prompt_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + len(json.dumps(tools).encode()) + 10000
        request_upper = prompt_bound * .00000012 + 2048 * .00000020
        upper = 2 * request_upper  # reserve both possible HTTP attempts
        self.budget.reserve(upper)
        body = {"model": self.model, "messages": messages, "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "finish"}} if force_finish else "auto",
                "temperature": 0, "max_tokens": 2048, "reasoning": {"enabled": False}, "usage": {"include": True},
                "provider": {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True}}
        if force_tool:
            body["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        # Keep the same pinned provider and sampling seed as the baseline.
        if self.provider == "DeepInfra":
            body["seed"] = SEED
        attempts = []
        try:
            for attempt in range(2):
                started = time.monotonic()
                try:
                    response = requests.post(self.url, headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
                                             json=body, timeout=(15, 150))
                    if response.status_code != 200:
                        error = response.text.replace(self.key, "[REDACTED]")[:800]
                        attempts.append({"status": response.status_code, "seconds": round(time.monotonic() - started, 3), "error": error})
                        if attempt == 0 and (response.status_code in {408, 429} or response.status_code >= 500):
                            time.sleep(2)
                            continue
                        raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {error}")
                    data = response.json()
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]).replace(self.key, "[REDACTED]")[:800])
                    attempts.append({"status": 200, "seconds": round(time.monotonic() - started, 3)})
                    self.budget.add(float(data.get("usage", {}).get("cost") or 0))
                    write_json(EXPERIMENT / "usage_ledger" / (data["id"] + ".json"),
                               {"generation_id": data["id"], "model": self.model, "provider": data.get("provider"), "usage": data.get("usage", {}), "attempts": attempts})
                    result = data, attempts
                    break
                except (requests.Timeout, requests.ConnectionError) as exception:
                    attempts.append({"error": type(exception).__name__, "seconds": round(time.monotonic() - started, 3)})
                    if attempt:
                        raise RuntimeError(type(exception).__name__ + " after two attempts") from None
                    time.sleep(2)
            else:
                raise RuntimeError("API request did not complete")
            missing_cost = result[0].get("usage", {}).get("cost") is None
            failed = sum(a.get("status") != 200 for a in attempts)
            self.budget.release(upper, request_upper * (failed + int(missing_cost)))
            return result
        except Exception:
            # Timeout may have been billed; retain the conservative charge.
            self.budget.release(upper, upper)
            raise



def available_schemas(turn):
    return schemas_for("search_inspect", final_turn=(turn == MAX_TURNS - 1))


def run_one(item, mode, budget, client_factory=GuardedClient):
    if set(item) != {"input_id", "entity", "chunk_text"}:
        raise ValueError("Only unlabelled input fields are permitted")
    if mode != "example_equivalence":
        raise ValueError("Only the prompt experiment may call the API; reuse saved baseline traces")
    started = time.monotonic()
    evidence = read_json(EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
    # Use exactly the context window visible to the model for quote verification.
    tools = PolicyTools(EXPERIMENT / "examples.sqlite", graph=False)
    tools.seen = {s: set(evidence["seen"][s]) for s in SYSTEMS}
    policy = BASE_POLICY + "\n" + SELECTION_CHECK
    messages = [{"role": "system", "content": TASK_PROMPT + "\n" + policy},
                {"role": "user", "content": json.dumps(evidence["payload"], ensure_ascii=False)}]
    client = client_factory(MODEL, PROVIDER, budget)
    trace, usages = [], []
    prediction, error, status = None, None, "model"
    try:
        for turn in range(MAX_TURNS):
            final_turn = turn == MAX_TURNS - 1
            allowed = available_schemas(turn)
            forced = None
            response, attempts = client.complete(messages, allowed, force_finish=final_turn, force_tool=forced)
            usages.append({"usage": response.get("usage", {}), "provider": response.get("provider"), "model": response.get("model"),
                           "generation_id": response.get("id"), "attempts": attempts})
            choice = response["choices"][0]; message = choice["message"]
            messages.append({k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "reasoning_details"} and v is not None})
            record = {"turn": turn + 1, "finish_reason": choice.get("finish_reason"), "tool_calls": [], "assistant_content": message.get("content")}
            trace.append(record)
            calls = message.get("tool_calls", [])
            if not calls:
                try:
                    candidate = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", message.get("content") or "").strip())
                    if validate_finish(candidate, tools) is None:
                        prediction = candidate
                        break
                except (ValueError, TypeError, AttributeError):
                    pass
                messages.append({"role": "user", "content": "Use the permitted tools, then return both codes using finish within the remaining response budget."})
                continue
            if len(calls) > 12:
                raise RuntimeError("Per-response tool limit exceeded")
            names = {s["function"]["name"] for s in allowed}
            for call in calls:
                name = call["function"]["name"]; arguments = {}
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                    if name not in names:
                        raise ValueError("Tool disabled in this condition or phase")
                    if name == "finish":
                        problem = validate_finish(arguments, tools)
                        result = {"error": problem} if problem else {"accepted": True}
                        if not problem:
                            prediction = arguments
                    else:
                        result = tools.dispatch(name, arguments)
                        assert_flat(result)
                except (ValueError, TypeError, sqlite3.Error) as exception:
                    result = {"error": str(exception)}
                record["tool_calls"].append({"name": name, "arguments": arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            if prediction:
                break
    except Exception as exception:
        error = str(exception).replace(client.key, "[REDACTED]")[:1200]
        status = "fallback_error"
    if prediction is None:
        if status != "fallback_error":
            status = "fallback_budget_or_invalid"
        prediction = {"icd10cm_code": evidence["fallback"]["ICD10CM"], "snomed_code": evidence["fallback"]["SNOMEDCT_US"],
                      "confidence": 0.0, "evidence": "Deterministic initial-search fallback; no accepted model finish."}
    result = {"input_id": item["input_id"], "mode": mode, "model": MODEL, "provider_requested": PROVIDER,
              "prediction": prediction, "status": status, "error": error, "initial_evidence_sha256": evidence["sha256"],
              "seen_candidates": {s: sorted(tools.seen[s]) for s in SYSTEMS}, "trace": trace, "messages": messages,
              "model_calls": len(usages), "usage_records": usages,
              "cost_usd": sum(float(u["usage"].get("cost") or 0) for u in usages),
              "prompt_tokens": sum(u["usage"].get("prompt_tokens", 0) for u in usages),
              "completion_tokens": sum(u["usage"].get("completion_tokens", 0) for u in usages),
              "latency_seconds": round(time.monotonic() - started, 3)}
    tools.close()
    return result


def score():
    labels = read_json(EXPERIMENT / "labels.json")
    tools = PolicyTools(EXPERIMENT / "examples.sqlite")
    summary, vectors, detailed = {}, {}, []
    for mode in MODES:
        rows = [read_json(EXPERIMENT / "runs" / mode / (r["input_id"] + ".json")) for r in labels]
        if any(r["status"] == "fallback_error" for r in rows):
            raise RuntimeError("Resolve transport/infrastructure errors before scoring; preserve completed runs")
        scores, vecs = {}, {}
        for system, field in zip(SYSTEMS, ("icd10cm_code", "snomed_code")):
            hits, buckets = [], {}
            valid = initial_recall = ever_recall = missing = selection = 0
            for item, result in zip(labels, rows):
                gold = item["gold"][system]; prediction = result["prediction"][field]
                hit = int(gold == prediction); hits.append(hit)
                frequency = tools.example_db.execute("SELECT count(*) FROM examples WHERE system=? AND code=?", (system, gold)).fetchone()[0]
                bucket = "unseen" if frequency == 0 else "rare_1_to_5" if frequency <= 5 else "common_gt5"
                stat = buckets.setdefault(bucket, {"n": 0, "correct": 0}); stat["n"] += 1; stat["correct"] += hit
                valid += tools.valid(system, prediction)
                evidence = read_json(EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
                initial_recall += gold in evidence["seen"][system]
                retrieved = gold in result["seen_candidates"][system]
                ever_recall += retrieved
                missing += not hit and not retrieved; selection += not hit and retrieved
                detailed.append({"arm": mode, "input_id": item["input_id"], "entity": item["entity"], "system": system,
                                 "gold": gold, "predicted": prediction, "correct": bool(hit), "ever_retrieved": retrieved,
                                 "status": result["status"], "frequency_bucket": bucket,
                                 "evidence": result["prediction"]["evidence"]})
            scores[system] = {"correct": sum(hits), "n": len(hits), "accuracy": sum(hits) / len(hits),
                              "initial_evidence_recall": initial_recall / len(hits), "ever_retrieved_recall": ever_recall / len(hits),
                              "retrieval_misses": missing, "selection_errors": selection,
                              "valid_predictions": valid, "frequency_buckets": buckets}
            vecs[system] = hits
        vecs["both"] = [a * b for a, b in zip(vecs["ICD10CM"], vecs["SNOMEDCT_US"])]
        latencies = sorted(r["latency_seconds"] for r in rows)
        calls = Counter(c["name"] for r in rows for t in r["trace"] for c in t["tool_calls"] if c["name"] != "finish")
        summary[mode] = {"model": MODEL, "per_system": scores, "headline_min_accuracy": min(v["accuracy"] for v in scores.values()),
                         "both_correct": sum(vecs["both"]), "both_accuracy": sum(vecs["both"]) / len(labels),
                         "statuses": dict(Counter(r["status"] for r in rows)), "model_calls": sum(r["model_calls"] for r in rows),
                         "tool_calls_by_name": dict(calls), "cost_usd": sum(r["cost_usd"] for r in rows),
                         "prompt_tokens": sum(r["prompt_tokens"] for r in rows), "completion_tokens": sum(r["completion_tokens"] for r in rows),
                         "latency_median_seconds": statistics.median(latencies), "latency_p95_seconds": latencies[math.ceil(.95 * len(latencies)) - 1]}
        vectors[mode] = vecs
    pairs = {a + "_vs_" + b: {s: paired_delta(vectors[a][s], vectors[b][s]) for s in (*SYSTEMS, "both")}
             for a, b in [("example_equivalence", "baseline")]}
    tools.close()
    output = {"n_inputs": len(labels), "arms": summary, "paired_comparisons": pairs,
              "total_cost_usd": sum(r["cost_usd"] for r in summary.values()),
              "new_inference_cost_usd": summary["example_equivalence"]["cost_usd"], "total_model_calls": sum(r["model_calls"] for r in summary.values()),
              "leakage_audit": read_json(EXPERIMENT / "leakage_audit.json"),
              "limits": ["80-input development comparison; exploratory intervals, no multiple-comparison correction",
                         "Only an appended generic system-prompt check changes; no new tools, mandatory phases or label lookups",
                         "Cached baseline from prior run, not simultaneous repeated inference; hosted outputs can vary",
                         "Same schemas and response budget; the prompt can change subsequent searches and candidate exposure",
                         "lexical retrieval with model reformulation; no dense retrieval or official relationship graph",
                         "Same 80 previously inspected development inputs; this is a development benchmark, not independent confirmation"]}
    write_json(EXPERIMENT / "results.json", output)
    with (EXPERIMENT / "all_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)
    print(json.dumps({"n_inputs": len(labels), "arms": summary, "paired_comparisons": pairs, "total_cost_usd": output["total_cost_usd"]}, indent=2), flush=True)
    return output



def experiment_config(cap, workers):
    protected = [ARTIFACTS / 'runs' / 'smoke20_v1', ARTIFACTS / 'comparisons_v1', SOURCE, ARTIFACTS / 'selective_v1', ARTIFACTS / 'selection_audit_v1']
    hashes = {str(p.relative_to(ARTIFACTS)): file_hash(p) for root in protected for p in sorted(root.rglob('*')) if p.is_file()}
    return {'model': MODEL, 'provider': PROVIDER, 'modes': MODES, 'max_responses': MAX_TURNS,
            'seed': SEED, 'temperature': 0, 'reasoning_enabled': False, 'max_completion_tokens': 2048,
            'new_inference_cap_usd': cap, 'workers': workers,
            'input_sha256': file_hash(EXPERIMENT / 'inputs.json'), 'labels_sha256': file_hash(EXPERIMENT / 'labels.json'),
            'example_sha256': file_hash(EXPERIMENT / 'examples.sqlite'),
            'source_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'initial_evidence_sha256': {p.name: file_hash(p) for p in sorted((EXPERIMENT / 'initial_evidence').glob('*.json'))},
            'retained_inputs_and_baseline_sha256': read_json(EXPERIMENT / 'retained_inputs_and_baseline.json'),
            'protected_previous_artifacts_sha256': hashes,
            'prompt_sha256': digest(TASK_PROMPT + BASE_POLICY + SELECTION_CHECK)}


def run(workers=4, cap=.20):
    config = experiment_config(cap, workers)
    config_path = EXPERIMENT / 'config.json'
    if config_path.exists() and read_json(config_path) != json.loads(json.dumps(config)):
        raise RuntimeError('Frozen configuration changed; preserve completed results')
    write_json(config_path, config)
    items = read_json(EXPERIMENT / 'inputs.json')
    directory = EXPERIMENT / 'runs' / 'example_equivalence'
    directory.mkdir(parents=True, exist_ok=True)
    jobs, cached = [], []
    for item in items:
        path = directory / (item['input_id'] + '.json')
        if path.exists():
            previous = read_json(path)
            if previous['status'] != 'fallback_error':
                cached.append(previous)
                continue
            dest = EXPERIMENT / 'failed_attempts' / item['input_id']
            dest.mkdir(parents=True, exist_ok=True)
            path.rename(dest / f"attempt_{len(list(dest.glob('*.json')))+1}.json")
        jobs.append((item, path))
    jobs.sort(key=lambda job: digest('selection-prompt-schedule:' + job[0]['input_id']))
    budget = ReservedBudget(cap)
    ledger = [read_json(p) for p in (EXPERIMENT / 'usage_ledger').glob('*.json')]
    budget.add(sum(float(r.get('usage', {}).get('cost') or 0) for r in ledger))
    accounting = EXPERIMENT / 'execution.json'
    if accounting.exists():
        budget.uncertain_cost_bound = read_json(accounting)['uncertain_cost_upper_bound_usd']
    print(f'Example-equivalence prompt: {len(jobs)} pending, {len(cached)} cached; 80 baseline traces retained, no baseline API calls; cap ${cap:.2f}', flush=True)
    def work(job):
        item, path = job
        result = run_one(item, 'example_equivalence', budget)
        write_json(path, result)
        return result
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print(f"[{i}/{len(jobs)}] {row['input_id']} {row['status']} calls={row['model_calls']} cost=${row['cost_usd']:.5f}; total=${budget.cost:.5f}", flush=True)
    write_json(accounting, {'wall_seconds_this_invocation': round(time.monotonic()-started,3),
                            'new_inference_reported_cost_usd': budget.cost,
                            'uncertain_cost_upper_bound_usd': budget.uncertain_cost_bound, 'spending_limit_usd': cap,
                            'baseline_API_calls_this_experiment': 0, 'retained_baseline_traces': 80})
    if experiment_config(cap, workers) != config:
        raise RuntimeError('Frozen experiment or protected prior artifacts changed during inference')
    for relative, expected in config['retained_inputs_and_baseline_sha256'].items():
        if file_hash(EXPERIMENT / relative) != expected:
            raise RuntimeError('Retained baseline, data or starting evidence changed')
    for item in items:
        evidence = read_json(EXPERIMENT / 'initial_evidence' / (item['input_id']+'.json'))
        result = read_json(directory / (item['input_id']+'.json'))
        if result['initial_evidence_sha256'] != evidence['sha256']:
            raise RuntimeError('Starting evidence differs from baseline')
    return score()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','run','score'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-cost', type=float, default=.20)
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'run': run(args.workers, args.max_cost)
    else: score()


if __name__ == '__main__':
    main()
