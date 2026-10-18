#!/usr/bin/env bash
set -euo pipefail
umask 077
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${TERMMAP_WEB_PYTHON:-python3}" -u -m demo.server --runtime "${TERMMAP_DEMO_RUNTIME:-$REPO_ROOT/runtime/demo}" --port "${TERMMAP_DEMO_PORT:-8787}"
