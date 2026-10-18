"""Local raw-two-logit Qwen service; run only inside the owned Slurm job."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def dispatch_pooling_adapter(original, serving, ctx, supports_default_mm_loras=False):
    """Fix vLLM 0.19's missing return after resolving a named pooling LoRA.

    The installed method sets ctx.lora_request and then rejects the same name
    for not being a base-model name. Keep the fix local to this service process;
    never edit the shared vLLM environment or bypass unknown-model validation.
    """
    name = ctx.request.model
    if not supports_default_mm_loras and name in serving.models.lora_requests:
        ctx.lora_request = serving.models.lora_requests[name]
        return None
    return original(serving, ctx, supports_default_mm_loras)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocate the GPU through Slurm before starting vLLM")
    settings = json.loads(Path(args.manifest).read_text())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_LOGGING_LEVEL="INFO",
        VLLM_BATCH_INVARIANT="1" if settings["batch_invariant"] else "0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="13")
    # The two original LM-head rows are retained separately in bf16. Subtract
    # after conversion to float, exactly as in the reference scorer. Do not
    # pre-subtract head weights or return a sigmoid-transformed probability.
    overrides = dict(architectures=["Qwen3ForSequenceClassification"],
        classifier_from_token=["no", "yes"], method="no_post_processing",
        num_labels=2, head_dtype=settings["head_dtype"])
    from vllm.entrypoints.pooling.base.serving import PoolingServing
    original_dispatch = PoolingServing._maybe_get_adapters
    def dispatch(serving, ctx, supports_default_mm_loras=False):
        return dispatch_pooling_adapter(original_dispatch, serving, ctx, supports_default_mm_loras)
    PoolingServing._maybe_get_adapters = dispatch
    command = ["vllm.entrypoints.openai.api_server", "--model", settings["base"],
        "--served-model-name", settings["model_name"], "--runner", "pooling",
        "--host", settings["host"], "--port", str(settings["port"]),
        "--dtype", settings["dtype"], "--seed", "13", "--max-model-len", "512",
        "--attention-backend", settings["attention_backend"],
        "--hf-overrides", json.dumps(overrides),
        "--pooler-config", json.dumps(dict(seq_pooling_type="LAST", use_activation=False)),
        "--enable-lora", "--max-lora-rank", "16", "--lora-modules",
        json.dumps(dict(name=settings["adapter_name"], path=settings["adapter"],
                        base_model_name=settings["model_name"])),
        "--max-num-seqs", str(settings["max_num_seqs"]),
        "--max-num-batched-tokens", str(settings["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(settings["gpu_memory_utilization"]),
        "--disable-uvicorn-access-log", "--no-enable-log-requests", "--no-enable-log-outputs"]
    if settings["enforce_eager"]: command.append("--enforce-eager")
    sys.argv = command
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__": main()
