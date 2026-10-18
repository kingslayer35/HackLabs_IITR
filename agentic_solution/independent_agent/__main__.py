from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import ARTIFACTS, ROOT, SYSTEMS, digest, read_json, write_json


def wilson(successes: int, total: int):
    if not total:
        return None
    z = 1.95996398454
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(center - half, 4), round(center + half, 4)]


def evaluate(run_dir: Path, artifact_dir: Path = ARTIFACTS):
    items = read_json(artifact_dir / "smoke20.json")
    results = {path.stem: read_json(path) for path in sorted((run_dir / "traces").glob("*.json"))}
    missing = [r["input_id"] for r in items if r["input_id"] not in results]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} smoke results")
    from .tools import TerminologyTools
    tools = TerminologyTools(artifact_dir / "terminology.sqlite")
    rows = []
    metrics = {}
    for system, field in [("ICD10CM", "icd10cm_code"), ("SNOMEDCT_US", "snomed_code")]:
        correct = baseline_correct = initial_recall = final_recall = valid = fixes = breaks = 0
        buckets = {}
        for item in items:
            result = results[item["input_id"]]
            gold = item["gold"][system]
            predicted = result["prediction"][field]
            hit = predicted == gold
            base_hit = result["baseline"][system] == gold
            correct += hit; baseline_correct += base_hit
            initial_recall += gold in result["initial_candidates"][system]
            final_recall += gold in result["seen_candidates"][system]
            valid += tools.valid(system, predicted)
            fixes += hit and not base_hit
            breaks += base_hit and not hit
            frequency = tools.db.execute("SELECT count(*) FROM examples WHERE system=? AND code=?", (system, gold)).fetchone()[0]
            bucket = "unseen" if frequency == 0 else "rare_1_to_5" if frequency <= 5 else "common_gt5"
            b = buckets.setdefault(bucket, {"n": 0, "correct": 0})
            b["n"] += 1; b["correct"] += hit
            gold_card = tools.card(system, gold)
            prediction_card = tools.card(system, predicted)
            rows.append({"input_id": item["input_id"], "entity": item["entity"], "system": system, "gold": gold,
                         "gold_description": gold_card["preferred"] if gold_card else "NOT IN DICTIONARY",
                         "predicted": predicted, "predicted_description": prediction_card["preferred"] if prediction_card else "INVALID",
                         "correct": hit, "baseline_correct": base_hit, "gold_initially_retrieved": gold in result["initial_candidates"][system],
                         "gold_ever_retrieved": gold in result["seen_candidates"][system], "train_frequency": frequency,
                         "status": result["status"], "evidence": result["prediction"]["evidence"]})
        n = len(items)
        metrics[system] = {"correct": correct, "n": n, "accuracy": correct / n, "wilson_95_ci": wilson(correct, n),
                           "baseline_accuracy": baseline_correct / n, "initial_recall": initial_recall / n,
                           "ever_retrieved_recall": final_recall / n, "valid_predictions": valid,
                           "fixes_vs_baseline": fixes, "regressions_vs_baseline": breaks, "frequency_buckets": buckets}
    tools.close()
    ordered = [results[r["input_id"]] for r in items]
    latency = sorted(r["latency_seconds"] for r in ordered)
    both = sum(all(r["prediction"][field] == item["gold"][s] for s, field in [("ICD10CM", "icd10cm_code"), ("SNOMEDCT_US", "snomed_code")]) for item in items for r in [results[item["input_id"]]])
    summary = {"inputs": len(items), "per_system": metrics, "headline_min_accuracy": min(m["accuracy"] for m in metrics.values()),
               "both_correct": both, "both_correct_accuracy": both / len(items),
               "statuses": dict(Counter(r["status"] for r in ordered)), "cost_usd": sum(r["cost_usd"] for r in ordered),
               "model_calls": sum(r["model_calls"] for r in ordered), "tool_calls": sum(r["tool_calls"] for r in ordered),
               "prompt_tokens": sum(r["prompt_tokens"] for r in ordered), "completion_tokens": sum(r["completion_tokens"] for r in ordered),
               "latency_median_seconds": statistics.median(latency), "latency_p95_seconds": latency[math.ceil(.95 * len(latency)) - 1],
               "leakage_audit": read_json(artifact_dir / "leakage_audit.json"),
               "scope": "20 fixed paired smoke inputs, not a performance claim on the full evaluation distribution",
               "confidence_note": "Model confidence is heuristic and uncalibrated. Wilson intervals are descriptive, not grouped-bootstrap intervals."}
    write_json(run_dir / "metrics.json", summary)
    with (run_dir / "errors_and_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (run_dir / "predictions.csv").open("w", newline="") as handle:
        fields = ["input_id", "clinical_entity", "icd10cm_code", "snomed_code"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in ordered:
            writer.writerow({"input_id": result["input_id"], "clinical_entity": result["entity"],
                             "icd10cm_code": result["prediction"]["icd10cm_code"], "snomed_code": result["prediction"]["snomed_code"]})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def smoke(args):
    from .agent import Budget, OpenRouterClient, PROMPT_VERSION, run_agent
    items = read_json(args.artifacts / "smoke20.json")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    traces = args.run_dir / "traces"; traces.mkdir(exist_ok=True)
    config = {"model": args.model, "provider": args.provider, "seed": 13, "temperature": 0,
              "max_turns": args.max_turns, "max_completion_tokens_per_request": 2048, "reasoning_enabled": False,
              "prompt_version": PROMPT_VERSION, "smoke_sha256": digest((args.artifacts / "smoke20.json").read_text()),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
              "workers": args.workers, "spending_stop_usd": args.max_cost,
              "dataset_sha256": read_json(args.artifacts / "terminology_stats.json")["sources"]}
    config_file = args.run_dir / "config.json"
    if config_file.exists() and read_json(config_file) != config:
        previous = read_json(config_file)
        old_settings = {k: v for k, v in previous.items() if k != "source_sha256"}
        new_settings = {k: v for k, v in config.items() if k != "source_sha256"}
        if not args.resume_after_repair or old_settings != new_settings:
            raise RuntimeError("Run config differs; choose a new --run-dir. An infrastructure-only source repair can be recorded with --resume-after-repair.")
        repair_path = args.run_dir / "repair_history.json"
        history = read_json(repair_path) if repair_path.exists() else []
        history.append({"previous_source_sha256": previous["source_sha256"], "new_source_sha256": config["source_sha256"],
                        "cached_predictions": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(traces.glob("*.json"))},
                        "note": "Infrastructure repair only; existing predictions retained. Model settings, prompt version and smoke split unchanged."})
        write_json(repair_path, history)
    write_json(config_file, config)
    pending = [r for r in items if not (traces / (r["input_id"] + ".json")).exists()]
    budget = Budget(args.max_cost)
    for path in traces.glob("*.json"):
        budget.add(float(read_json(path).get("cost_usd", 0)))
    print(f"Running {len(pending)} inputs; {len(items) - len(pending)} cached; provider={args.provider}", flush=True)
    started = time.monotonic()
    def work(item):
        client = OpenRouterClient(args.model, args.provider, budget)
        # Explicit whitelist ensures no gold labels enter the agent.
        unlabelled = {k: item[k] for k in ("input_id", "entity", "chunk_text")}
        result = run_agent(unlabelled, client, args.artifacts / "terminology.sqlite", args.max_turns)
        write_json(traces / (item["input_id"] + ".json"), result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(work, item): item["input_id"] for item in pending}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{len(pending)}] {result['input_id']} status={result['status']} calls={result['model_calls']} tools={result['tool_calls']} cost=${result['cost_usd']:.5f} seconds={result['latency_seconds']:.1f}", flush=True)
    summary = evaluate(args.run_dir, args.artifacts)
    summary["run_wall_seconds"] = round(time.monotonic() - started, 3)
    write_json(args.run_dir / "metrics.json", summary)


def main():
    parser = argparse.ArgumentParser(description="Independent TermMap terminology-search agent")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--data", type=Path, default=ROOT / "data")
    prepare_parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    smoke_parser.add_argument("--run-dir", type=Path, default=ARTIFACTS / "runs" / "smoke20_v1")
    smoke_parser.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    smoke_parser.add_argument("--provider", default="DeepInfra")
    smoke_parser.add_argument("--workers", type=int, default=2)
    smoke_parser.add_argument("--max-turns", type=int, default=6)
    smoke_parser.add_argument("--max-cost", type=float, default=2.0)
    smoke_parser.add_argument("--resume-after-repair", action="store_true")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    eval_parser.add_argument("--run-dir", type=Path, default=ARTIFACTS / "runs" / "smoke20_v1")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.data, args.artifacts)
    elif args.command == "smoke":
        if args.workers < 1 or args.workers > 8 or args.max_turns < 2 or args.max_turns > 12:
            parser.error("workers must be 1..8 and max-turns 2..12")
        smoke(args)
    else:
        evaluate(args.run_dir, args.artifacts)


if __name__ == "__main__":
    main()
