#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
job_init posttrain "${1:-}"
if [[ -n "$SMOKE_FLAG" ]]; then
    timeout 175s python -u -m scripts.posttraining.train --smoke
else
    python -u -m scripts.posttraining.train
fi
