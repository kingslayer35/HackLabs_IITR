#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then printf 'Usage: bash run.sh <input.csv> <output.csv>\n' >&2; exit 2; fi
TERMMAP_REPO="$(cd "$(dirname "$0")" && pwd)"
TERMMAP_INPUT="$(realpath -- "$1")"
TERMMAP_OUTPUT="$(realpath -m -- "$2")"
cd "$TERMMAP_REPO"
export TERMMAP_CONFIG="${TERMMAP_CONFIG:-$TERMMAP_REPO/configs/default.yaml}"
python -m scripts.inference.launch "$TERMMAP_INPUT" "$TERMMAP_OUTPUT"
