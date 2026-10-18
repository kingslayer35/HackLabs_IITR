#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
job_init indexes "${1:-}"
if [[ -n "$SMOKE_FLAG" ]]; then
    timeout 175s python -u -m scripts.retrieval.build_indexes --smoke
else
    python -u -m scripts.retrieval.build_indexes
fi
