# TermMap demo server

A minimal CSV upload interface around the team's **actual released F11 pipeline** in the repository root. Pipeline source, model weights, fitted decisions and inference settings are not edited. This is independent of the earlier agentic solution.

## Use the demo

Open **http://localhost:8787** while the SSH tunnel is active. Choose a CSV, inspect the input preview, then click **Run mapping**. The page shows actual preprocessing/retrieval/reranking/decision/validation stages, both code descriptions, review suggestions, context and top alternatives. Download the validated four-column prediction CSV when it completes.

An included synthetic sample contains two contexts for MS, an OCR-corrupted kidney-disease mention, and knee pain. These are sent through the live model; mappings are never hardcoded. In acceptance testing, the cardiac MS example still received the SNOMED multiple-sclerosis concept, exposing an existing model limitation. The demo preserves that output and its review suggestion. Results are model predictions, not an accuracy claim or clinical adjudication.

Input is UTF-8 CSV, at most 2 MB and 100 rows:

```csv
chunk_id,clinical_entity,chunk_text,subconcept
example-1,right knee pain,Assessment: right knee pain without trauma.,DiagnosisCode
```

`clinical_entity` (or `entity`) is required. `chunk_id`, `chunk_text` and `subconcept` are optional. IDs/codes remain strings and input order is preserved. The UI accepts extracted terms, not PDFs or raw-chart entity extraction.

## Start / restart

On the server:

```bash
cd /path/to/HackLabs_IITR
mkdir -p runtime/demo
# CPU web/API process; start in a dedicated terminal or named screen.
bash demo/scripts/start.sh
```

In a separate server terminal, obtain an owned Slurm GPU worker:

```bash
cd /path/to/HackLabs_IITR
sbatch --output=runtime/demo/worker-%j.log demo/scripts/worker.sbatch
```

The worker starts once, verifies the release, loads models, and handles queued uploads serially. It has a two-hour Slurm limit and exits after 30 minutes without a run to release an unused GPU. Startup can take several minutes; uploads can be checked while models load. `runtime/demo/worker.json`, `runtime/demo/worker-<job>.log` and the UI show readiness/failure. Do not start a second worker while the first is live; a lock prevents duplicate ownership. Do not cancel or reuse other jobs.

From your laptop, keep this tunnel running:

```bash
ssh -N -L 8787:127.0.0.1:8787 shivank_g@192.168.104.73
```

Then open http://localhost:8787. The API binds only to `127.0.0.1`; there is no public unauthenticated clinical-data endpoint. No passwords or API keys are stored here.

## Integration details

`pipeline_adapter.py` imports `OnlineRetriever`, `HostedFinalRanker`, `Features`, `apply_decision`, and `validate_predictions` directly from the curated checkout, following `scripts/inference/predict.py`. `verify_release` runs before loading the engine. Model objects stay resident between runs; output validation is the same pipeline validator.

This machine's curated checkout omits disposable indexes. On the current research host, explicitly configure the worker to reuse the finalized research indexes at `../HackLabs/artifacts/indexes/reviewed_v2` and the original frozen alias vectors at `../HackLabs/artifacts/indexes/eval/C1`. Checkpoint weights, query-manifest hash, original alias-vector hashes and available terminology tables are checked for compatibility. Legacy trusted index pickles are resolved through Python module aliases to the same curated lexical classes. No original index is overwritten and no model training occurs. This is a demo with verified existing indexes, not a claim of a fresh index-rebuild acceptance run.

All new requests, status, results, vLLM manifests/logs and job state are kept in the repository’s private `runtime/demo/`. vLLM uses a separate loopback port, 19089, and its frozen existing environment. Only path/port/cache-output overrides are applied. Set `TERMMAP_PIPELINE_PYTHON` and `TERMMAP_VLLM_PYTHON` for your two isolated environments. Optional `TERMMAP_INDEX_ROOT` and `TERMMAP_FROZEN_ALIAS_ROOT` select verified existing indexes. Without these overrides the curated configuration is used. With newly rebuilt curated indexes, omit the optional index overrides when invoking `python -m demo.worker` and use the curated config's paths.

The browser uses no external scripts/fonts or inference APIs. It keeps only the opaque current run ID in session storage, so refreshing the same tab reconnects to the run. There is no global job listing. Uploaded files/results remain under operator-controlled `runtime/demo/jobs/`; they are not silently deleted. Cancellation stops between model operations and never kills the shared worker or returns incomplete results.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Worker readiness, flow and upload limits |
| `GET /api/sample.csv` | Synthetic sample input |
| `POST /api/uploads` | Raw UTF-8 CSV body, `Content-Type: text/csv`; optional URL-encoded `X-Filename` |
| `GET /api/jobs/{id}` | Preview/status/stage/processed-row count |
| `POST /api/jobs/{id}/run` | Enqueue once; duplicate submissions are idempotent while active |
| `POST /api/jobs/{id}/cancel` | Cancel queued work or request cancellation of active work |
| `GET /api/jobs/{id}/results` | Validated rows, descriptions and diagnostics after completion |
| `GET /api/jobs/{id}/download` | Official-format predictions CSV after completion |

Runtime endpoints accept no server paths or shell commands. Run IDs are random, path-validated identifiers. CSV validation, upload size/row limits, queue bounds, state transitions and completion checks are enforced by the server.

## Verification

The API/UI server itself uses only Python's standard library. Inference uses the team's existing pinned environments; no global environment was changed.

```bash
python3 -m unittest demo.tests.test_demo -v
node --check demo/static/app.js
```

Browser acceptance uses Playwright and axe in a separate development environment:

```bash
# In your own isolated Node tooling environment, install playwright and @axe-core/playwright.
NODE_PATH=/path/to/node_modules node demo/scripts/browser.cjs
```

Optionally set `CHROMIUM_PATH` to an installed Chromium executable. The browser test uses real inference on the synthetic sample and writes screenshots/checks under `evidence/`. See [validation](../docs/VALIDATION.md) for completed checks and acceptance limits. Generated browser evidence and internal design notes are excluded from the submission.

## Files

- `server.py`: loopback HTTP API and static assets.
- `store.py`: CSV validation, private persistence and atomic queue transitions.
- `worker.py`: one warm Slurm GPU worker and heartbeat.
- `pipeline_adapter.py`: released pipeline integration.
- `static/`: offline HTML/CSS/JavaScript UI.
- `scripts/`: server launch, owned GPU job and browser acceptance.
- `tests/`: API, validation, state/cancellation and transport checks.

## Current research-machine worker configuration

These are paths, not credentials. Export them before submitting from the repository root. On this host use `/opt/slurm/bin/sbatch` if `sbatch` is not in PATH.

```bash
export TERMMAP_PIPELINE_PYTHON=/home/shivank_g/anaconda3/envs/termmap/bin/python
export TERMMAP_VLLM_PYTHON=/home/shivank_g/anaconda3/envs/vllm_serve/bin/python
export TERMMAP_INDEX_ROOT=/home/shivank_g/projects/pratham/entity_group/HackLabs/artifacts/indexes/reviewed_v2
export TERMMAP_FROZEN_ALIAS_ROOT=/home/shivank_g/projects/pratham/entity_group/HackLabs/artifacts/indexes/eval/C1
mkdir -p runtime/demo
/opt/slurm/bin/sbatch --output=runtime/demo/worker-%j.log demo/scripts/worker.sbatch
```

The web process and worker must use the same `TERMMAP_DEMO_RUNTIME` (default `runtime/demo` under this repository). Never submit another worker over an active one. Imports use the `demo` package; execute commands from the repository root, rather than running individual Python files directly.
