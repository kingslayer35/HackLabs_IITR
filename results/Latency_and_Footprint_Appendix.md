# Appendix — component latency, footprint and chart-volume assessment

This appendix reports a **new, fine-grained F11 timing measurement** using latest preprocessing, the fixed asymmetric SapBERT P2-v2 encoders and revised-input Qwen3-Reranker-0.6B LoRA **epoch 2** with the frozen F11 decision layer. It does not retrain a model, refit decisions or change the published DEV/TEST accuracies. One term means **one clinical-entity input with predictions for both ICD-10-CM and SNOMED**, rather than one candidate pair.

## Appendix table A — warm latency by component

All entries are milliseconds per input term. Each main component is exclusive; the last row measures the complete request directly.

| Component | Median ms/term | p95 ms/term |
| --- | --- | --- |
| Mention preprocessing, context and type handling | 10.01 | 12.24 |
| SapBERT query encoding — all three encoder roles | 59.17 | 61.39 |
| Character TF-IDF and BM25 scoring | 949.19 | 1009.16 |
| Dense alias search — both systems / configured variants | 25.18 | 29.26 |
| Mention-memory scoring and query bookkeeping (residual) | 10.81 | 11.63 |
| Candidate fusion, ranking and configured structural expansion | 27.99 | 477.99 |
| Contextual Qwen LoRA reranking — both systems | 461.12 | 579.25 |
| Decision features, fusion, specificity, coupling and confidence | 18.15 | 30.12 |
| Output assembly and code/schema validation | 1.45 | 1.67 |
| Complete warm request | 1567.07 | 1965.13 |

**Accounting:** component times add to the total for each individual input (maximum measured accounting error 0.00000000 ms). Component medians and p95s are independent quantiles, so their displayed values need not add to the total median or p95. “Mention-memory scoring and query bookkeeping” is the measured remainder of query preparation after excluding the separately timed encoders, lexical scoring and dense search; it is **not a separately isolated matmul-kernel latency**.

## Appendix table B — encoder and retrieval subcomponents

These rows further divide the encoding, lexical and dense rows in table A. They must **not** be added again to the table A total.

| Nested subcomponent | Median ms/term | p95 ms/term |
| --- | --- | --- |
| Original SapBERT: type-inference query vectors | 20.60 | 22.10 |
| Adapted ICD SapBERT: ICD query vectors | 19.18 | 21.16 |
| Adapted SNOMED SapBERT: SNOMED query vectors | 19.03 | 19.59 |
| Character TF-IDF scoring | 845.84 | 896.24 |
| BM25 scoring | 101.46 | 115.87 |
| ICD alias dense search, including prepared CDC variants | 6.66 | 9.80 |
| SNOMED alias dense search | 18.30 | 21.70 |

Original SapBERT produces type-inference vectors. The selected ICD and SNOMED encoders produce their respective query vectors. Dictionary/alias vectors remain original SapBERT and are precomputed. Dense timing includes the unchanged implementation’s prepared ICD CDC variants even when an optional retrieval variant is disabled by policy. Candidate fusion/expansion obeys the saved policy; no disabled bridge or expansion component is enabled for this measurement.

**Measured bottleneck:** Character TF-IDF and BM25 scoring accounts for 56.5% of mean request time; contextual reranking accounts for 30.8%, and all three encoder roles together account for 3.6%. These shares use mean component times, which are additive across the measured inputs. They describe this serial configuration and do not establish the result of any unmeasured optimization.

## Appendix table C — initialization and runtime footprint

| Measurement | Observed value | Scope |
| --- | --- | --- |
| Hosted reranker launch to healthy endpoint | 25.03 s | Model service startup; excludes Python launcher startup and first-request compilation |
| Retriever / features / decision initialization | 50.53 s | Loads three encoders, train-only indexes, classifier, frozen decisions and legal-code lookup sets |
| Allocated-device peak memory | 39.53 GiB | Whole allocated GPU; includes vLLM reservation, encoders and indexes |
| Observed process-tree RSS peak | 14.52 GiB | Owned client/server processes; shared pages may be counted more than once |
| Device/process telemetry | 130 samples | Sampled approximately once per second during client initialization and measurement |
| Reference allocation | 1 × RTX A6000; 6 CPU cores; 48 GiB requested RAM | Slurm allocation; GPU has 49,140 MiB as reported by nvidia-smi |

Startup values are separate observations and are excluded from warm per-term latency. They are not a complete cold-start-to-first-prediction benchmark. Device peaks are observed sampled values, not minimum memory requirements or a guarantee that a smaller GPU will fit. The GPU measurement includes reserved serving memory, rather than only the Python client’s PyTorch allocation.

## Appendix table D — selected disk assets

These are inventory sizes, not per-request costs or a complete distribution size.

| Asset group | Size MiB | Files |
| --- | --- | --- |
| Adapted ICD | 418.55 | 6 |
| Adapted SNOMED | 418.55 | 6 |
| Alias and mention-memory arrays | 1868.41 | 8 |
| Epoch-2 adapter | 93.28 | 12 |
| F11 decisions | 0.20 | 1 |
| Original SapBERT | 417.94 | 6 |
| Qwen base | 1151.55 | 7 |

Selected vector/row arrays exclude lexical indexes, terminology tables, temporary query/score caches, raw challenge data, logs and environment packages. Resolved files are counted once within each asset group. These sizes do not establish the full deployment bundle footprint.

## Measurement protocol and boundaries

- **Sample:** 50 unique saved DEV groups, sorted by row ID then sampled with seed 13; three public synthetic warm-ups. A separate public reference input is run once through original methods and once through wrapped methods to check instrumentation equivalence. No TEST examples are read.
- **Execution:** warm, sequential, one input request at a time; both systems predicted. Candidate scoring within a request uses the existing local vLLM client batching/concurrency. This is not a multi-request load test.
- **Software/model:** vLLM 0.19.0, bf16 Qwen margins and selected epoch-2 adapter; original alias vectors, per-system adapted query/memory encoders; fixed train-only evaluation assets and decision parameters. Actual GPU architecture: `sm86`.
- **Fresh work:** query IDs are cleared per request so query vectors, lexical/dense scores and memory similarities are recomputed. Client score cache and token-input cache are disabled. Normal vLLM prefix/KV reuse remains enabled, and loaded model/index objects stay warm.
- **No input labels:** only mention and chart context are supplied to inference; absent subconcepts use the trained type classifier. These input-information conditions differ from chart evaluations that have supplied subconcepts.
- **Timers:** monotonic wall-clock spans, with CUDA synchronization around encoder/dense work and stage boundaries. Wrappers call the unchanged methods; original and instrumented public-input reranker scores, ranked outputs, confidences and missing-code estimates match exactly. Synchronization and profiling overhead are part of this instrumented run.
- **Included in total:** context/preprocessing, all encoders and channel scores, candidate fusion/ranking, local hosted reranking, frozen decisions, output assembly and schema/code validation.
- **Excluded:** Slurm queue wait, service/asset initialization, offline dictionary/index construction, model training, file upload, remote/UI transport, CSV/JSON file writing, and performance on other flows or hardware. No per-component minimum-GPU-memory claims are made.
- **Statistics:** sample median and pandas 95th percentile (linear interpolation), computed from the 50 per-input timings. No latency confidence intervals, repeated-trial variance, or sustained/concurrent throughput benchmark is claimed.

## Chart-processing-volume assessment

The observed mean complete service time is **1527.83 ms/term**, corresponding to **0.655 terms/s** or **2356 terms/hour** when calculated from the sum of measured request times. This is an empirical serial service-time rate, not a measured production throughput guarantee. The service fits the allocated A6000 under the observed memory reservation.

The following are arithmetic projections using the measured **mean**, with one GPU, serial requests, no queueing, no competing work and the same input mix. They are **not measured chart or batch runtimes**.

| Illustrative workload | Projected serial compute time | Interpretation |
| --- | --- | --- |
| 100 terms | 2.55 min | One hypothetical chart containing 100 extracted mentions |
| 1,000 terms | 25.46 min | Density/number of charts unspecified |
| 10,000 terms | 4.24 h | Service-time projection only |
| 100,000 terms | 42.44 h | Exceeds a 24-hour serial one-GPU budget if this value is above 24 h |

**Verdict:** the measured configuration supports a warm interactive demo and provides evidence for modest offline processing on the reference GPU. Deployability at a specified chart-processing volume remains **unverified**: the PS gives no target chart rate or mentions per chart, and 50 sequential DEV inputs do not establish sustained concurrency, queueing behavior, reliability, or service-level compliance. Large-volume claims require a specified workload and a separate sustained load test; multiple workers or GPUs cannot be assumed to give linear speedup. Deployment acceptance of a restored/rebuilt curated package is a separate check, not established by this timing measurement.

## Relationship to the previously published latency

The earlier coarser benchmark reported **1.346 s median / 1.757 s p95**. It combined preprocessing, all encoders and channel scoring and stopped its total timer before output validation. This appendix is a separate instrumented measurement with a finer breakdown and validation included. It is not an accuracy rerun, a backend comparison or evidence of a speedup/slowdown caused by model changes. Earlier DEV/TEST reports and their saved values remain unchanged.

## Suggested paper text

> On one NVIDIA RTX A6000, the frozen F11 pipeline processed 50 seeded DEV inputs sequentially after warm-up, predicting both coding systems per input. Median and p95 complete latency, including output assembly and validation, were 1.567 s and 1.965 s. Observed peak allocated-device memory was 39.53 GiB, including the hosted reranker reservation, encoders and indexes. Component timings are given in the appendix. These warm single-request measurements do not establish sustained chart-volume throughput.

## Evidence and reproducibility

[Machine-readable appendix metrics](Latency_and_Footprint_metrics.json) retain full precision, implementation hashes, timing-table provenance and footprint scope. [Earlier judging diagnostics](Judging_metrics.json) preserve the original coarser measurement. [Complete prediction results](README.md) remain separate. Main-repository reproduction uses `configs/latency_appendix.yaml`, `scripts/continue_latency_appendix.py`, `scripts/measure_component_latency.py` and `slurm/run_component_latency.sh`, with the documented screen/Slurm smoke/full procedure. No model retraining or TEST reopening is needed.

Private evidence in the research checkout: `artifacts/runs/latency_appendix_20261018/benchmark/component_per_input_timings.parquet`, `component_metrics.json`, `footprint.json`, and the service startup/batch audit under the same run. The aggregate JSON records their hashes; private input/group identities are not copied into the public appendix.
