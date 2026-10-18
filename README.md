# TermMap — HiLabs / IIT Roorkee submission

TermMap maps an extracted clinical mention and its optional chart context to one ICD-10-CM code and one SNOMED CT US concept. It combines conservative preprocessing, terminology structure, hybrid candidate retrieval, a contextual Qwen reranker, and a fitted decision layer.

This repository contains the curated team pipeline, a CSV upload demo, and an independent agent research archive. The original pipeline structure and frozen model policies are preserved; research logs, datasets, weights and caches stay outside normal Git commits.

**Submission status:** source, final four-flow DEV/internal-TEST results, training datasets, selected models and frozen decision-layer handoff are prepared. The four trained model repositories are public and ungated, with verified anonymous weight access and checksums. Before submission, complete GPU deployment acceptance, the documentation PDF/demo, the restricted Drive link and repository/judge access. See [deliverables](docs/DELIVERABLES.md).

## Repository layout

| Component | Code | What it does |
|---|---|---|
| Preprocessing | [termmap/preprocessing](termmap/preprocessing/README.md) | Final reviewed mention views, context anchoring, abbreviation evidence, protected clinical tokens and noise |
| Hierarchy / terminology | [termmap/terminology](termmap/terminology/README.md) | ICD tree, official SNOMED semantic tags/top levels, conservative links, aliases and train-only priors |
| Candidate retrieval | [termmap/retrieval](termmap/retrieval/README.md) | Exact/lexical/BM25/dense/memory channels, RRF and bounded expansion; asymmetric adapted encoding |
| SapBERT pretraining | [termmap/pretraining](termmap/pretraining/README.md) | Final P2-v2 frozen-index InfoNCE method and per-system checkpoint selection |
| Qwen post-training | [termmap/posttraining](termmap/posttraining/README.md) | Revised-input, two-epoch LoRA training and seeded 16-candidate quizzes |
| Contextual reranking | [termmap/reranking](termmap/reranking/README.md) | Prompt packing, candidate cards, original/LoRA scoring and optional local vLLM serving |
| Decision layer | [termmap/decision](termmap/decision/README.md) | Fusion, supported specificity, joint decisions, confidence and missing-code detection |
| Evaluation | [termmap/evaluation](termmap/evaluation/README.md) | Grouped splits/CV/bootstrap, long-tail buckets, diagnostics and output validation |
| Stage entry points | [scripts](scripts/README.md) | Commands grouped by the same components |
| GPU jobs | [slurm](docs/OPERATIONS.md) | Screen + Slurm allocation, smoke gates and completion status |
| Tests | [tests](tests) | Synthetic regressions and private-dictionary integration checks |

## Demo and independent research

- [demo/](demo/README.md): HTTP server, minimal frontend, pipeline adapter, Slurm worker and demo tests. It runs the released F11 pipeline.
- [agentic_solution/](agentic_solution/README.md): independent agent implementation, experiment code and analyses, with its own environment.
- [Agent summary](agentic_solution/SUMMARY.md), [combined report](agentic_solution/FINAL_REPORT.md) and [code index](agentic_solution/CODE_INDEX.md): experimental results and inferences. These results are separate from the team metrics below.

From the repository root, start the CPU web server with `bash demo/scripts/start.sh`. A separately allocated GPU worker is required for inference; see the demo README. No external hosted API is used by this demo.

## Setup

Use Python 3.11 in an isolated environment. The core environment is pinned in `requirements.txt`; CUDA 12.8 supports the machines used for the work. The local vLLM serving environment has its own [lockfile](requirements-vllm.lock) and is separate from the core environment. The default final-flow configuration uses this backend.

```bash
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
# Separate server environment for the default hosted backend:
python -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -r requirements-vllm.lock
```

All operational paths and retained policies are in [configs/default.yaml](configs/default.yaml). Training configs write to separate `runtime/training/` directories. `.env.example` contains paths only; the engine requires no API key. It does not load `.env` implicitly. Set `TERMMAP_CONFIG` to choose a config. Optional `CONDA_SH` and `TERMMAP_ENV` let GPU scripts activate your environment; otherwise run them from an already activated environment.

Download the **private artifact ZIP** from the link below and extract it at this repository root. It restores ignored `private_assets/`, including training datasets, terminology tables, the original base models, selected SapBERT encoders and both LoRA epochs. Verify its published SHA256 first. Clinical text never goes into ordinary Git commits, Hugging Face model uploads, or external inference calls.

The refreshed bundle includes all four frozen decision models, the original train-fitted type classifier, validated DEV/probe/TEST predictions and `private_assets/models/RELEASE.json`. Disposable indexes are rebuilt locally, rather than distributed as caches. The curated checkout still requires a GPU deployment acceptance test after rebuilding those indexes. See [reproduction](docs/REPRODUCIBILITY.md).

## One-command prediction

After the verified artifact handoff and index build:

```bash
bash run.sh /absolute/path/input.csv /absolute/path/predictions.csv
```

On the target cluster this launches a named screen, obtains a Slurm GPU, runs smoke, predicts, validates and releases the allocation. For a GPU already allocated through Slurm:

```bash
python -m scripts.inference.predict /absolute/path/input.csv /absolute/path/predictions.csv
```

Input: `chunk_id, clinical_entity`; `chunk_text` and `subconcept` are optional. `entity` is accepted as an alternate mention column. Output, in the same order: `chunk_id, clinical_entity, icd10cm_code, snomed_code`. Codes remain strings. Private diagnostics beside the predictions contain top candidates, confidence, missing-code probability and cue/channel evidence. The input file and read-only dictionary directory cannot be overwritten. A missing final decision handoff fails before allocating a GPU.

## Models and four flows

| Flow | Retriever query / memory encoders | Reranker |
|---|---|---|
| F00 | Original SapBERT | Original pretrained Qwen, no LoRA |
| F10 | Selected per-system SapBERT P2-v2 | Original pretrained Qwen, no LoRA |
| F01 | Original SapBERT | Revised-input Qwen LoRA epoch 2 |
| F11 | Selected per-system SapBERT P2-v2 | Revised-input Qwen LoRA epoch 2 |

Every flow includes an independently fitted decision layer. **Alias/dictionary vectors and type-classifier vectors always use original SapBERT.** Adapted queries and memory use `best_ICD` (step 1431, epoch 9.0) and `best_SNOMED` (step 950, about epoch 6.0). The selected Qwen checkpoint is revised epoch 2, by explicit choice; epoch 1 is preserved. No all-label retrain is claimed.

The headline comparison is `min(ICD exact-code accuracy, SNOMED all-types exact-code accuracy)` on supplied gold labels. DiagnosisCode-only SNOMED accuracy is supplementary. The local saved split has 6,751 train, 1,368 DEV and 1,527 internal TEST rows; grouped splits make percentages approximate. DEV uses grouped cross-validation of decision parameters. TEST uses frozen DEV decisions. Dictionary-derived probes are separate. All four flows are complete on DEV and internal TEST; these are not the organizers' hidden evaluation results.

<!-- FINAL_RESULTS_START -->
| Set | Flow | ICD accuracy (95% CI) | SNOMED DiagnosisCode (95% CI) | SNOMED all-types (95% CI) | Primary minimum (95% CI) |
|---|---|---|---|---|---|
| DEV | F00 | 52.91 [47.54, 58.36] | 51.81 [46.84, 56.78] | 43.13 [40.04, 46.11] | 43.13 [40.18, 46.19] |
| DEV | F10 | 56.88 [51.46, 62.19] | 52.59 [47.53, 57.84] | 45.24 [42.08, 48.42] | 45.24 [41.96, 48.26] |
| DEV | F01 | 60.86 [55.66, 65.85] | 57.77 [52.44, 63.22] | 50.53 [47.57, 53.54] | 50.53 [47.60, 53.70] |
| DEV | F11 | 66.97 [61.81, 71.81] | 57.25 [52.03, 62.30] | 51.30 [48.23, 54.50] | 51.30 [48.12, 54.47] |
| TEST | F00 | 53.07 [47.38, 58.03] | 51.54 [46.81, 56.18] | 42.71 [39.83, 45.39] | 42.71 [39.91, 45.73] |
| TEST | F10 | 62.27 [56.88, 67.06] | 55.13 [49.74, 59.85] | 44.80 [41.69, 47.49] | 44.80 [42.03, 47.91] |
| TEST | F01 | 65.03 [60.00, 69.85] | 61.54 [56.34, 66.59] | 50.46 [47.39, 53.15] | 50.46 [47.58, 53.32] |
| TEST | F11 | 68.40 [63.38, 73.45] | 62.05 [56.92, 67.19] | 50.71 [47.87, 53.36] | 50.71 [48.12, 53.57] |
<!-- FINAL_RESULTS_END -->

The [final results analysis](results/README.md) and [complete stage/category tables](results/Final_all_flows_DEV_TEST.md) separate full retrieval, raw reranking and final decisions. They include grouped confidence intervals, paired deltas, unseen/rare/common buckets, separate probes, hierarchy/error analysis and measured F11 latency/footprint. The [664-row aggregate CSV](results/Final_all_flows_metrics.csv), [full metrics JSON](results/Final_all_flows_full_metrics.json), [judging diagnostics](results/Judging_metrics.json) and [provenance](docs/RESULTS_PROVENANCE.md) preserve the evidence.

F11 improves the primary minimum over F00 by **8.17 percentage points on DEV** and **7.99 on TEST** (TEST paired 95% CI **[5.61, 10.37]**). Its TEST gain over F01 is **0.25 points [−1.13, 1.92]**, so the additional overall benefit from encoder adaptation with LoRA is uncertain. The roughly 95% challenge target is not met. Warm F11 median latency is **1.346 seconds per input**, predicting both systems; this is not a cold-start or concurrent-throughput claim. Cross-reranker comparisons include the disclosed backend/GPU differences.

## Artifacts, public models and demo

<!-- ARTIFACT_LINKS_START -->
| Artifact | Link |
|---|---|
| Private artifact ZIP | Pending upload / access link |
| SapBERT ICD query encoder | [SapBERT ICD query encoder](https://huggingface.co/yobro4619/termmap-sapbert-icd) |
| SapBERT SNOMED query encoder | [SapBERT SNOMED query encoder](https://huggingface.co/yobro4619/termmap-sapbert-snomed) |
| Qwen LoRA epoch 1 | [Qwen LoRA epoch 1](https://huggingface.co/yobro4619/termmap-qwen-lora-epoch1) |
| Qwen LoRA epoch 2 (selected) | [Qwen LoRA epoch 2 (selected)](https://huggingface.co/yobro4619/termmap-qwen-lora-epoch2) |
| Demo video | Pending upload / access link |

ZIP SHA256: `ea3d2590050a4fdf0fdb72280d0e42700ed0920f9411b72c83570062f2bea420`.
<!-- ARTIFACT_LINKS_END -->

Drive access must stay restricted to the challenge team/judges. The four model links above point to **public, ungated** repositories under `yobro4619`; model downloads require no account or token. Both Qwen repositories contain LoRA adapters requiring the original `Qwen/Qwen3-Reranker-0.6B` base, and epoch 2 is selected. File checksums, uploaded revisions and anonymous-access verification are recorded in [the upload manifest](docs/MODEL_UPLOADS.json). Clinical datasets and per-input predictions remain in the restricted ZIP.

Optional [initial-upload tooling](tools/upload_models.py) stages allowlisted model/tokenizer files privately, verifies the authenticated account before creating repositories, and checks uploaded file inventories and hashes. The four submission repositories were subsequently made public at the owner's explicit request; their complete inventories and anonymous weight access were verified. The private staging uploader deliberately refuses an existing public repository and is not the command for changing visibility. For a new staging upload, run `python tools/upload_models.py --namespace YOUR_ACCOUNT --prompt-token`; alternatively set `HF_TOKEN` locally. Credentials are never recorded in the repository. For an organization, specify `--expected-user TOKEN_OWNER` explicitly. Clinical datasets are excluded from these uploads. Use [tools/update_links.py](tools/update_links.py) to update links after future uploads.

## Submission documentation

- [Submission deliverables and remaining handoff](docs/DELIVERABLES.md)
- [Architecture and design decisions](docs/ARCHITECTURE.md)
- [Reproduction and training data formats](docs/REPRODUCIBILITY.md)
- [Checkpoint inventory](docs/CHECKPOINTS.md)
- [Edge cases and limitations](docs/EDGE_CASES.md)
- [External sources and attribution](docs/RESOURCES.md)
- [Results and evaluation provenance](docs/RESULTS_PROVENANCE.md)
- [Validation and acceptance limits](docs/VALIDATION.md)
- [Demo usage](demo/README.md)
- [Independent agent experiment summary](agentic_solution/SUMMARY.md)

Keep this GitHub repository **private**, grant judges access, and share no proprietary chart material publicly. No remote operation or upload is performed by setup. Team members handle their own later pushes and attribution; the local history is organized by component.
