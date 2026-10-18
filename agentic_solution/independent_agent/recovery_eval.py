"""Frozen DeepSeek-only ablation: baseline, query recovery, recovery + comparison."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .agent import FINISH_SCHEMA, OpenRouterClient, context_window, validate_finish
from .common import ARTIFACTS, ROOT, SEED, SYSTEMS, digest, input_id, jaccard, norm, read_json, shingles, write_json
from .comparisons import PolicyTools, ReservedBudget, TASK_PROMPT, assert_flat, paired_delta, schemas_for
from .recovery_tools import RecoveryTools, RECOVERY_SCHEMA, COMPARISON_SCHEMA

EXPERIMENT = ARTIFACTS / "recovery_v1"
MODEL = "deepseek/deepseek-v4-flash-0731"
PROVIDER = "DeepInfra"
MODES = ("baseline", "query_recovery", "recovery_comparison")
MAX_TURNS = 5

BASE_POLICY = "Use tools only when additional evidence is needed. You may finish immediately if the evidence is sufficient. Reformulate abbreviations/noisy mentions for search. Batch independent tool requests. Relationship tools and relationship metadata are not available. At most 5 model responses are allowed; the last is reserved for finish."
RECOVERY_POLICY = """First call recover_queries for BOTH systems. Interpret only the requested mention in its local context,
then propose one standard clinical name and at most two alternative phrasings per system.
Keep all numbers and clinically meaningful qualifiers; add no unsupported laterality, acuity, severity,
stage or complication. Supported details must quote the input exactly; use an empty list if none.
Do not use a neighbouring diagnosis as a replacement for the mention. Original-mention search is
preserved automatically. Generated interpretations and search names are hypotheses, not facts.
After recovery, inspect or search as needed and select dictionary-verified codes.
Batch independent tools. Relationships are unavailable. At most 5 model responses, last reserved for finish."""
COMPARISON_POLICY = """Before finishing, call compare_candidates for EACH system with its 2 to 6 most plausible retrieved codes
when at least two dictionary-valid candidates exist. Batch the two system comparisons in one response.
Use the side-by-side descriptions to identify which candidate distinction the mention/context supports.
Description-word differences are lexical clues, not clinical rules. Do not add specificity merely because
it occurs in one candidate. If no candidate fits the mention, search again. Final selected codes must have
been included in a comparison when two or more candidates were available."""


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(count=80):
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if (EXPERIMENT / "prepared.json").exists():
        manifest = read_json(EXPERIMENT / "prepared.json")
        if manifest["count"] != count:
            raise RuntimeError("Preserve the frozen experiment size")
        return manifest
    old = read_json(ARTIFACTS / "smoke20.json") + read_json(ARTIFACTS / "comparisons_v1" / "labels.json")
    old_ids = {r["input_id"] for r in old}
    old_groups = {r["group_id"] for r in old}
    old_mentions = {norm(r["entity"]) for r in old}
    old_shingles = [shingles(r["chunk_text"]) for r in old]
    assignments = read_json(ARTIFACTS / "split_manifest.json")["groups"]
    grouped = defaultdict(list)
    with (ROOT / "data" / "hackathon_train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subconcept"] != "DiagnosisCode":
                continue
            row["input_id"] = input_id(row["entity"], row["chunk_text"])
            row["group_id"] = digest(norm(row["chunk_text"]) or "empty:" + norm(row["entity"]))[:20]
            grouped[row["input_id"]].append(row)
    eligible = []
    for key, rows in grouped.items():
        row = rows[0]
        if assignments[row["group_id"]] != "test" or row["group_id"] in old_groups or key in old_ids or norm(row["entity"]) in old_mentions:
            continue
        if {r["coding_system"] for r in rows} != set(SYSTEMS):
            continue
        if any(len({r["code"] for r in rows if r["coding_system"] == s}) != 1 for s in SYSTEMS):
            continue
        if any(jaccard(shingles(row["chunk_text"]), previous) >= .85 for previous in old_shingles):
            continue
        eligible.append(key)
    eligible.sort(key=lambda k: digest(f"recovery-v1:{SEED}:{k}"))
    selected, used_groups, selected_shingles = [], set(), []
    for key in eligible:
        rows = grouped[key]; row = rows[0]; sh = shingles(row["chunk_text"])
        if row["group_id"] in used_groups or any(jaccard(sh, prior) >= .85 for prior in selected_shingles):
            continue
        used_groups.add(row["group_id"]); selected_shingles.append(sh)
        selected.append({"input_id": key, "entity": row["entity"], "chunk_text": row["chunk_text"], "group_id": row["group_id"],
                         "gold": {r["coding_system"]: r["code"] for r in rows}})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} fresh paired groups available; requested {count}")
    mentions = {norm(r["entity"]) for r in selected}
    source = sqlite3.connect(f"file:{ARTIFACTS / 'terminology.sqlite'}?mode=ro", uri=True); source.row_factory = sqlite3.Row
    original = [dict(r) for r in source.execute("SELECT * FROM examples ORDER BY id")]; source.close()
    group_shingles = {r["group_id"]: shingles(r["context"]) for r in original}
    excluded = {r["group_id"] for r in original if norm(r["entity"]) in mentions or r["group_id"] in used_groups}
    excluded.update(g for g, sh in group_shingles.items() if any(jaccard(sh, t) >= .85 for t in selected_shingles))
    examples = [r for r in original if r["group_id"] not in excluded]
    db_path = EXPERIMENT / "examples.sqlite"
    if db_path.exists():
        raise RuntimeError("Incomplete preparation exists; inspect before rebuilding")
    db = sqlite3.connect(db_path)
    db.executescript("""CREATE TABLE examples(id INTEGER PRIMARY KEY, system TEXT, code TEXT, entity TEXT, context TEXT, input_id TEXT, group_id TEXT);
       CREATE VIRTUAL TABLE example_fts USING fts5(entity,context,content='examples',content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');""")
    db.executemany("INSERT INTO examples VALUES(?,?,?,?,?,?,?)", [tuple(r[k] for k in ("id", "system", "code", "entity", "context", "input_id", "group_id")) for r in examples])
    db.execute("INSERT INTO example_fts(example_fts) VALUES('rebuild')"); db.commit(); db.close()
    max_similarity = max(jaccard(group_shingles[g], target) for g in {r["group_id"] for r in examples} for target in selected_shingles)
    audit = {"fresh_inputs": count, "distinct_context_groups": len(used_groups), "training_examples": len(examples),
             "training_groups_quarantined": len(excluded), "shared_input_ids": len({r["input_id"] for r in selected} & {r["input_id"] for r in examples}),
             "shared_normalized_mentions": len(mentions & {norm(r["entity"]) for r in examples}),
             "shared_context_groups": len(used_groups & {r["group_id"] for r in examples}),
             "shared_previous_eval_groups": len(used_groups & old_groups), "shared_previous_eval_mentions": len(mentions & old_mentions),
             "maximum_test_training_shingle_jaccard": max_similarity, "threshold": .85,
             "selection": "Seeded hash order; paired reserved-test inputs, one per context group; prior 60 eval inputs/groups/mentions/near-duplicates excluded; no within-set near duplicates."}
    assert not any(audit[k] for k in ("shared_input_ids", "shared_normalized_mentions", "shared_context_groups", "shared_previous_eval_groups", "shared_previous_eval_mentions"))
    assert max_similarity < .85
    write_json(EXPERIMENT / "labels.json", selected)
    write_json(EXPERIMENT / "inputs.json", [{k: r[k] for k in ("input_id", "entity", "chunk_text")} for r in selected])
    write_json(EXPERIMENT / "leakage_audit.json", audit)
    baselines = {}
    for row in selected:
        tools = PolicyTools(db_path)
        initial = {s: tools.search_concepts(s, [row["entity"]], limit=12) for s in SYSTEMS}
        demos = {s: tools.find_training_examples(row["entity"], s, limit=2) for s in SYSTEMS}
        fallback = {}
        for s in SYSTEMS:
            choices = [r["code"] for r in initial[s]["results"] if tools.valid(s, r["code"])]
            fallback[s] = choices[0] if choices else ("R69" if s == "ICD10CM" else "404684003")
        payload = {"mention": row["entity"], "context": context_window(row["entity"], row["chunk_text"]),
                   "initial_candidates": initial, "training_examples": demos}
        assert_flat(payload)
        record = {"input_id": row["input_id"], "payload": payload, "seen": {s: sorted(tools.seen[s]) for s in SYSTEMS},
                  "fallback": fallback, "sha256": digest(json.dumps(payload, sort_keys=True, ensure_ascii=False))}
        write_json(EXPERIMENT / "initial_evidence" / (row["input_id"] + ".json"), record)
        baselines[row["input_id"]] = fallback
        tools.close()
    write_json(EXPERIMENT / "baseline_predictions.json", baselines)
    manifest = {"count": count, "audit": audit, "model": MODEL, "provider": PROVIDER, "modes": MODES,
                "selection_seed": SEED, "max_model_responses_all_arms": MAX_TURNS,
                "interventions": "Mandatory first recovery call in both new arms; comparison arm additionally requires selected codes to have been compared when 2+ candidates exist. All calls count toward the common five-response limit."}
    write_json(EXPERIMENT / "prepared.json", manifest)
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



def available_schemas(mode, recovered=False, final_turn=False):
    if mode not in MODES:
        raise ValueError("Unknown experiment condition")
    if final_turn:
        return [FINISH_SCHEMA]
    if mode != "baseline" and not recovered:
        return [RECOVERY_SCHEMA]
    schemas = schemas_for("search_inspect")
    if mode != "baseline":
        schemas = schemas[:-1] + [RECOVERY_SCHEMA] + schemas[-1:]
    if mode == "recovery_comparison":
        schemas = schemas[:-1] + [COMPARISON_SCHEMA] + schemas[-1:]
    return copy.deepcopy(schemas)


def finish_problem(candidate, tools, mode):
    problem = validate_finish(candidate, tools)
    if problem:
        return problem
    if mode != "baseline" and not tools.recovery_count:
        return "Complete query recovery before finishing"
    if mode == "recovery_comparison":
        for s, field in zip(SYSTEMS, ("icd10cm_code", "snomed_code")):
            valid_seen = [c for c in tools.seen[s] if tools.valid(s, c)]
            if len(valid_seen) >= 2 and candidate[field] not in tools.compared[s]:
                return f"Compare the selected {s} code against plausible alternatives before finishing"
    return None


def run_one(item, mode, budget, client_factory=GuardedClient):
    if set(item) != {"input_id", "entity", "chunk_text"}:
        raise ValueError("Only unlabelled input fields are permitted")
    started = time.monotonic()
    evidence = read_json(EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
    # Use exactly the context window visible to the model for quote verification.
    tools = RecoveryTools(EXPERIMENT / "examples.sqlite", item["entity"], evidence["payload"]["context"], mode == "recovery_comparison")
    tools.seen = {s: set(evidence["seen"][s]) for s in SYSTEMS}
    policy = BASE_POLICY if mode == "baseline" else RECOVERY_POLICY + ("\n" + COMPARISON_POLICY if mode == "recovery_comparison" else "")
    messages = [{"role": "system", "content": TASK_PROMPT + "\n" + policy},
                {"role": "user", "content": json.dumps(evidence["payload"], ensure_ascii=False)}]
    client = client_factory(MODEL, PROVIDER, budget)
    trace, usages = [], []
    prediction, error, status = None, None, "model"
    try:
        for turn in range(MAX_TURNS):
            final_turn = turn == MAX_TURNS - 1
            allowed = available_schemas(mode, bool(tools.recovery_count), final_turn)
            forced = "recover_queries" if mode != "baseline" and not tools.recovery_count and not final_turn else None
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
                    if finish_problem(candidate, tools, mode) is None:
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
                        problem = finish_problem(arguments, tools, mode)
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
              "seen_candidates": {s: sorted(tools.seen[s]) for s in SYSTEMS}, "recovery_count": tools.recovery_count,
              "compared_codes": {s: sorted(tools.compared[s]) for s in SYSTEMS}, "trace": trace, "messages": messages,
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
            valid = initial_recall = ever_recall = recovery_recall = missing = selection = 0
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
                recovered_ids = {r["code"] for turn in result["trace"] for call in turn["tool_calls"]
                                 if call["name"] == "recover_queries" and "results" in call["result"]
                                 for r in call["result"]["results"][system]["results"]}
                recovery_recall += gold in recovered_ids
                detailed.append({"arm": mode, "input_id": item["input_id"], "entity": item["entity"], "system": system,
                                 "gold": gold, "predicted": prediction, "correct": bool(hit), "ever_retrieved": retrieved,
                                 "in_recovery_results": gold in recovered_ids, "status": result["status"], "frequency_bucket": bucket,
                                 "evidence": result["prediction"]["evidence"]})
            scores[system] = {"correct": sum(hits), "n": len(hits), "accuracy": sum(hits) / len(hits),
                              "initial_evidence_recall": initial_recall / len(hits), "ever_retrieved_recall": ever_recall / len(hits),
                              "recovery_result_recall": recovery_recall / len(hits) if mode != "baseline" else None,
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
             for a, b in [("query_recovery", "baseline"), ("recovery_comparison", "query_recovery"), ("recovery_comparison", "baseline")]}
    tools.close()
    output = {"n_inputs": len(labels), "arms": summary, "paired_comparisons": pairs,
              "total_cost_usd": sum(r["cost_usd"] for r in summary.values()), "total_model_calls": sum(r["model_calls"] for r in summary.values()),
              "leakage_audit": read_json(EXPERIMENT / "leakage_audit.json"),
              "limits": ["80-input development comparison; exploratory intervals, no multiple-comparison correction",
                         "mandatory recovery and comparison workflows, not selective routing",
                         "comparison exposes no extra resource beyond dictionary descriptions already accessible to inspection",
                         "same maximum responses, but differing minimum required actions and generated evidence",
                         "lexical retrieval with model reformulation; no dense retrieval or official relationship graph",
                         "new inputs, so compare arms within this experiment rather than against previous 40-case percentages"]}
    write_json(EXPERIMENT / "results.json", output)
    with (EXPERIMENT / "all_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)
    print(json.dumps({"n_inputs": len(labels), "arms": summary, "paired_comparisons": pairs, "total_cost_usd": output["total_cost_usd"]}, indent=2), flush=True)
    return output


def experiment_config(cap, workers):
    protected = [ARTIFACTS / "runs" / "smoke20_v1", ARTIFACTS / "comparisons_v1"]
    # Protect all original evaluation artifacts; the large shared dictionary is read-only.
    old_hashes = {str(p.relative_to(ARTIFACTS)): file_hash(p) for root in protected for p in sorted(root.rglob("*")) if p.is_file()}
    return {"model": MODEL, "provider": PROVIDER, "modes": MODES, "max_responses": MAX_TURNS, "seed": SEED,
            "temperature": 0, "reasoning_enabled": False, "max_completion_tokens": 2048, "cap_usd": cap, "workers": workers,
            "input_sha256": file_hash(EXPERIMENT / "inputs.json"), "labels_sha256": file_hash(EXPERIMENT / "labels.json"),
            "example_sha256": file_hash(EXPERIMENT / "examples.sqlite"),
            "source_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "initial_evidence_sha256": {p.name: file_hash(p) for p in sorted((EXPERIMENT / "initial_evidence").glob("*.json"))},
            "protected_previous_artifacts_sha256": old_hashes,
            "prompt_sha256": digest(TASK_PROMPT + BASE_POLICY + RECOVERY_POLICY + COMPARISON_POLICY)}


def run(workers=4, cap=.50):
    config = experiment_config(cap, workers)
    config_path = EXPERIMENT / "config.json"
    # Round-trip JSON converts tuples into lists, so compare canonical representations.
    if config_path.exists() and read_json(config_path) != json.loads(json.dumps(config)):
        raise RuntimeError("Frozen configuration changed; preserve completed results")
    write_json(config_path, config)
    items = read_json(EXPERIMENT / "inputs.json")
    jobs, cached = [], []
    for mode in MODES:
        directory = EXPERIMENT / "runs" / mode; directory.mkdir(parents=True, exist_ok=True)
        for item in items:
            path = directory / (item["input_id"] + ".json")
            if path.exists():
                previous = read_json(path)
                if previous["status"] != "fallback_error":
                    cached.append(previous)
                    continue
                # Preserve failed attempts; only infrastructure failures may be retried.
                attempt_dir = EXPERIMENT / "failed_attempts" / mode / item["input_id"]
                attempt_dir.mkdir(parents=True, exist_ok=True)
                path.rename(attempt_dir / (f"attempt_{len(list(attempt_dir.glob('*.json')))+1}.json"))
            jobs.append((item, mode, path))
    budget = ReservedBudget(cap)
    # Ledger includes abandoned responses, unlike summing just completed predictions.
    ledger = [read_json(p) for p in (EXPERIMENT / "usage_ledger").glob("*.json")]
    budget.add(sum(float(r.get("usage", {}).get("cost") or 0) for r in ledger))
    accounting = EXPERIMENT / "execution.json"
    if accounting.exists():
        budget.uncertain_cost_bound = read_json(accounting)["uncertain_cost_upper_bound_usd"]
    jobs.sort(key=lambda j: (digest("recovery-schedule:" + j[0]["input_id"]), MODES.index(j[1])))
    print(f"DeepSeek ablation: {len(jobs)} pending, {len(cached)} preserved, cap ${cap:.2f}", flush=True)
    def work(job):
        item, mode, path = job
        result = run_one(item, mode, budget)
        write_json(path, result)
        return result
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print(f"[{i}/{len(jobs)}] {row['mode']} {row['input_id']} {row['status']} calls={row['model_calls']} cost=${row['cost_usd']:.5f}; total=${budget.cost:.5f}", flush=True)
    write_json(accounting, {"wall_seconds_this_invocation": round(time.monotonic() - started, 3), "reported_cost_usd": budget.cost,
                            "uncertain_cost_upper_bound_usd": budget.uncertain_cost_bound, "spending_limit_usd": cap})
    # Verify source, data, old artifacts and initial evidence before reading gold for scores.
    if experiment_config(cap, workers) != config:
        raise RuntimeError("Frozen experiment or protected artifacts changed during inference")
    for mode in MODES:
        for item in items:
            row = read_json(EXPERIMENT / "runs" / mode / (item["input_id"] + ".json"))
            evidence = read_json(EXPERIMENT / "initial_evidence" / (item["input_id"] + ".json"))
            if row["initial_evidence_sha256"] != evidence["sha256"]:
                raise RuntimeError("Initial evidence differs between conditions")
    return score()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "score"])
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=.50)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.count)
    elif args.command == "run": run(args.workers, args.max_cost)
    else: score()


if __name__ == "__main__":
    main()
