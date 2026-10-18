# Evaluation and validation

`metrics.py` computes exact-code accuracy and R@1/10/32/64, ICD structural diagnostics, SNOMED tag/top-level accuracy/error classes, frequency buckets, grouped bootstrap CIs and paired flow deltas. Accuracy equals R@1 at each ranking stage. The primary comparison is min(ICD, SNOMED all types); DiagnosisCode and paired both-correct diagnostics are separate.

`splits.py` preserves grouped seed-13 splits and dictionary-code probes. `export.py` emits both code columns, validates dictionary membership, strings, legal headers, input order and held-out group isolation, and writes private diagnostics. `scope.py` restricts requested flow combinations. `authorization.py` contains the research fixed-model TEST-scoring guard; portable deployment has a separate release manifest.

The internal saved split is train 6,751 / DEV 1,368 / TEST 1,527. The 6,000 clean/noisy probe inputs are a different distribution. Final comparisons must separate retrieval availability, raw reranker rankings and complete decisions. Only supplied gold labels contribute to accuracy; auxiliary predictions never become gold.

The first submission snapshot intentionally contains no partial/final-looking result table. Final DEV all-four, TEST F00/F11, paired confidence intervals, per-bucket/probe results, latency/backend evidence and validated predictions will be imported through the final handoff.
