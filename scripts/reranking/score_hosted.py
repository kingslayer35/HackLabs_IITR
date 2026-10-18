"""Publish independently namespaced four-flow scores from the local service."""
import argparse
import copy
import json
import os
import time

import joblib
import numpy as np
import torch

from termmap.common.io import load_config, path_for, sha256, write_json, verify_frozen
from termmap.reranking.model import Reranker
from termmap.reranking.client import make_ranker
from termmap.evaluation.scope import inference_flows
from scripts.reranking.score import metadata, raw_metrics


def atomic_joblib(value, target):
    temp = target.with_name(target.name + ".tmp")
    joblib.dump(value, temp)
    temp.replace(target)


def match_reference_adapter_dtype(reference, dtype):
    """Match vLLM's auto LoRA dtype in the disposable smoke reference only."""
    if dtype is None:return 0
    if dtype!="bfloat16":raise ValueError("Only the configured bf16 LoRA reference is supported")
    changed=0
    for name,parameter in reference.model.named_parameters():
        if "lora_" in name:
            parameter.data=parameter.data.to(torch.bfloat16);changed+=1
    if not changed:raise RuntimeError("The smoke reference has no loaded LoRA parameters")
    return changed


def smoke(cfg):
    root = path_for(cfg, "run_dir")
    cfg.setdefault("reranker", {})["cache_enabled"] = False
    reference = Reranker(cfg, adapter=path_for(cfg, "model_dir") / "reranker_lora/best")
    payloads = []
    for source in ["C1", "C2_v2"]:
        items = joblib.load(root / "vllm_service" / (source + "_audit.joblib"))
        for item in items[:1]:
            for value in [item, item["other_system"]]:
                payloads.extend(reference.cards.payload(value["row"], code,
                    seen=cfg["reranker"].get("seen_enabled", True)) for code, _ in value["pool"][:2])
    audit = {}; role_scores = {}
    original_adapter_scores=None
    reference_dtype=cfg["vllm"].get("reference_adapter_dtype")
    if reference_dtype is not None:
        # PEFT promotes loaded adapters to float32 by default, while vLLM's
        # auto LoRA dtype follows its bf16 base. Compare matched arithmetic and
        # preserve the original mixed-precision discrepancy as a diagnostic.
        original_adapter_scores=reference.score(payloads)
        changed=match_reference_adapter_dtype(reference,reference_dtype)
        print("Smoke reference LoRA dtype",reference_dtype,"parameters",changed,flush=True)
    for adapter in [False, True]:
        ranker = make_ranker(cfg, adapter=adapter, cards=reference.cards)
        ids = ranker.token_ids(payloads)
        encoded = reference.tokenize(payloads)
        reference_ids = [row[mask.bool()].tolist() for row, mask in zip(encoded["input_ids"], encoded["attention_mask"])]
        assert ids == reference_ids, "The service input differs from the reference tokens"
        if adapter:
            expected = reference.score(payloads)
        else:
            with reference.model.disable_adapter(): expected = reference.score(payloads)
        observed = ranker.request(ids)
        role_scores["lora" if adapter else "zero"] = observed
        repeated = ranker.request(ids)
        shuffled = ranker.request(list(reversed(ids)))[::-1]
        error = float(np.max(np.abs(np.array(expected)-observed)))
        diagnostic=dict(role="lora" if adapter else "zero",identity=ranker.identity,
            reference_adapter_dtype=reference_dtype,reference_max_absolute_score_difference=error)
        if adapter and original_adapter_scores is not None:
            diagnostic["original_mixed_precision_max_absolute_difference"]=float(np.max(np.abs(np.array(original_adapter_scores)-observed)))
        print("Smoke score agreement",json.dumps(diagnostic),flush=True)
        if error>0.5:write_json(root/"vllm_service/SMOKE_FAILURE.json",diagnostic)
        assert error <= 0.5, "Backend/model mismatch exceeds the fixed bf16 smoke tolerance"
        assert observed == repeated == shuffled, "The new service is not deterministic at 1e-5"
        audit["lora" if adapter else "zero"] = dict(diagnostic,
            identity=ranker.identity, n=len(ids), exact_token_agreement=True,
            reference_max_absolute_score_difference=error,
            exact_repeated_and_shuffled_scores=True,
            bitwise_reference_scores=expected == observed)
        ranker.close()
    # An accidentally ignored adapter must not pass as the post-trained arm.
    adapter_effect = float(np.max(np.abs(np.array(role_scores["lora"])-role_scores["zero"])))
    assert adapter_effect > 1e-5, "The service appears to have ignored the LoRA adapter"
    write_json(root / "vllm_service/SMOKE.json", dict(passed=True, roles=audit,
        raw_scores="float(yes_logit)-float(no_logit)", use_activation=False,
        adapter_max_score_effect=adapter_effect,
        control_policy="Preserve old published controls; score all roles consistently in the new backend namespace.",
        training_untouched=True, test_opened=False))
    reference.close()
    print("vLLM adapter/raw-score/token/determinism smoke passed", json.dumps(audit), flush=True)


def score_dataset(cfg, ranker, structures, flow, source, dataset):
    print("vLLM scoring",flow,dataset,"from",source,flush=True)
    root = path_for(cfg, "run_dir")
    dest = root / flow; dest.mkdir(exist_ok=True)
    inputs = root / source / (dataset + "_pools.joblib")
    target = dest / (dataset + "_scores.joblib")
    provenance = dict(source=source, source_sha256=sha256(inputs),
        reranker=ranker.identity, template_sha256=sha256(cfg["root"] / "termmap/reranking/model.py"),
        packing_sha256=sha256(cfg["root"] / "termmap/reranking/prompts.py"),
        target_instruction=cfg["reranker"].get("target_instruction", False),
        seen=cfg["reranker"].get("seen_enabled", True), other_system=True)
    record = dest / (dataset + "_score_config.json")
    done = dest / (dataset + "_timing.json")
    if target.exists() and record.exists() and done.exists() and json.loads(record.read_text()) == provenance:
        print("vLLM score reuse", flow, dataset, flush=True)
        return
    items = joblib.load(inputs)
    assert all("other_system" in item for item in items)
    scored = ranker.score_items(items)
    atomic_joblib(scored, target)
    write_json(record, provenance)
    raw_metrics(cfg, scored, flow, dataset, structures)
    if dataset == "probe":
        # Preserve the raw summary before the independent CPU fitter publishes
        # complete probe metrics at the historical probe_metrics.json path.
        write_json(dest / "probe_raw_summary.json",
                   json.loads((dest / "probe_metrics.json").read_text()))
    write_json(done, dict(ranker.last_timing,
        backend=ranker.identity, peak_vram=0,
        memory_note="Model lives in the separate local vLLM server process; client allocator is not model VRAM."))
    print("vLLM completed", flow, dataset, flush=True)
    del items, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset",choices=["dev","test","drill"],default="dev")
    args = parser.parse_args()
    cfg = load_config()
    if args.smoke:
        smoke(cfg)
        return
    root = path_for(cfg, "run_dir")
    selected = inference_flows(cfg)
    if not json.loads((root / "vllm_service/SMOKE.json").read_text())["passed"]:
        raise RuntimeError("The exact-token/adapter smoke must pass first")
    structures = metadata(cfg)
    if args.dataset in {"test","drill"}:
        if args.dataset=="test":
            if cfg.get("fixed_test_scoring_only"):
                from termmap.evaluation.authorization import verify_fixed_test_scorer
                verify_fixed_test_scorer(cfg)
            else:verify_frozen(cfg)
        for adapter,flows in ([(True,[("F11","drill")])] if args.dataset=="drill" else
                [(True,[("F01","test_base"),("F11","test_adapted")]),
                 (False,[("F00","test_base"),("F10","test_adapted")])]):
            flows=[(flow,source) for flow,source in flows if flow in selected]
            if not flows:continue
            ranker=make_ranker(cfg,adapter=adapter)
            for flow,source in flows:
                if flow in selected:score_dataset(cfg,ranker,structures,flow,source,args.dataset)
            ranker.close()
        return
    cards = None
    # Prioritize the two requested adapter flows. Probe is needed for the fixed
    # component keep/drop rules; the GPU can score the next arm during CPU fits.
    for adapter, flows in [(True, [("F01", "C1"), ("F11", "C2_v2")]),
                           (False, [("F00", "C1"), ("F10", "C2_v2")])]:
        flows = [(flow, source) for flow, source in flows if flow in selected]
        if not flows:continue
        ranker = make_ranker(cfg, adapter=adapter, cards=cards)
        cards = ranker.cards
        for dataset in ["dev", "probe"]:
            for flow, source in flows:
                score_dataset(cfg, ranker, structures, flow, source, dataset)
                if dataset == "probe":
                    write_json(root / flow / "SCORE_COMPLETE.json", dict(flow=flow,
                        slurm_job_id=os.environ["SLURM_JOB_ID"], datasets=["dev", "probe"], backend=ranker.identity))
        ranker.close()
    from scripts.evaluation.checkpoints import save_checkpoint
    if "F11" in selected:
        metrics = json.loads((root / "F11/dev_raw_metrics.json").read_text())
        save_checkpoint(cfg, "C3", metrics,
                        json.loads((root / "F11/probe_raw_summary.json").read_text()))
        if "F10" in selected:
            base = json.loads((root / "F10/dev_raw_metrics.json").read_text())
            write_json(root / "C3/gate.json", dict(
                passed=metrics["headline_min_acc_all_types"] > base["headline_min_acc_all_types"],
                note="Raw reranker diagnostic; complete independently fitted flows select the submission."))
    write_json(root / "SCORES_READY.json", dict(flows=list(selected),
        backend="vllm", complete=True, test_opened=False))


if __name__ == "__main__": main()
