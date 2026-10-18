#!/usr/bin/env bash
#SBATCH --job-name=termmap-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
REPO_ROOT="${TERMMAP_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=13
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
cd "$REPO_ROOT"
args=(--pipeline-root "$REPO_ROOT" --runtime "${TERMMAP_DEMO_RUNTIME:-$REPO_ROOT/runtime/demo}")
[[ -z "${TERMMAP_INDEX_ROOT:-}" ]] || args+=(--index-root "$TERMMAP_INDEX_ROOT")
[[ -z "${TERMMAP_FROZEN_ALIAS_ROOT:-}" ]] || args+=(--frozen-alias-root "$TERMMAP_FROZEN_ALIAS_ROOT")
[[ -z "${TERMMAP_VLLM_PYTHON:-}" ]] || args+=(--vllm-python "$TERMMAP_VLLM_PYTHON")
exec "${TERMMAP_PIPELINE_PYTHON:-python3}" -u -m demo.worker "${args[@]}"
