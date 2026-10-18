#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
job_init "$3" "${4:-}"
if [[ -n "$SMOKE_FLAG" ]]; then
    timeout 175s python -u -m scripts.inference.predict "$1" "$2" --run-id "$3" --smoke
else
    python -u -m scripts.inference.predict "$1" "$2" --run-id "$3"
fi
