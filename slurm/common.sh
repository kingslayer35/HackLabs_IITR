#!/usr/bin/env bash
set -euo pipefail

job_init() {
    if [[ -n "${CONDA_SH:-}" ]]; then
        source "$CONDA_SH"
        conda activate "${TERMMAP_ENV:-termmap}"
    fi
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=13
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    : "${SLURM_JOB_ID:?Allocate the GPU through srun before this script}"
    RUN_ID="$1"; SMOKE_FLAG=""
    if [[ "${2:-}" == "--smoke" ]]; then RUN_ID="${RUN_ID}_smoke"; SMOKE_FLAG="--smoke"; fi
    RUN_PATH="$(python -c 'import sys; from termmap.common.io import load_config,path_for; print(path_for(load_config(),"run_dir")/sys.argv[1])' "$RUN_ID")"
    mkdir -p "$RUN_PATH"
    rm -f "$RUN_PATH/STATUS"
    trap 'code=$?; if [[ $code -eq 0 ]]; then echo DONE > "$RUN_PATH/STATUS"; else echo FAILED > "$RUN_PATH/STATUS"; fi' EXIT
    hostname
    nvidia-smi --query-gpu=name,memory.total --format=csv
    python -c 'import torch; print(torch.cuda.get_device_name())'
}
