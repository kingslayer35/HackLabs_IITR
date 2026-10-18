# Candidate retrieval

`lexical.py` provides character TF-IDF, BM25, deterministic ranking and reciprocal-rank fusion. `candidates.py` assembles exact/literal, lexical, dense and train-memory evidence. RRF uses k=60, retains 50 fused candidates, and applies bounded structural expansion to at most 64. Current optional-channel keep/drop flags are fixed in config; CDC, bridge and family expansion are disabled in the retained policy, while supported subtree/look-alike handling remains available.

`dense.py` uses exact chunked torch matrix multiplication, never faiss. `asymmetric.py` implements the selected P2-v2 scheme: original SapBERT alias vectors; per-system adapted query and memory vectors; original query vectors for type inference. `confirmation.py` implements six fixed per-system checks, with independent original-model fallback and no retraining loop. `online.py` applies the same channels to new input and isolates architecture-specific query/memory scoring.

`cache.py` owns disposable index/query files and source fingerprints; these caches are rebuilt locally and excluded from the ZIP/Git. CPU preprocessing and GPU encoding are separate. `scripts.retrieval.build_indexes` rebuilds original anchors and both retrieval arms from private assets; `scripts.retrieval.confirm` runs confirmation on shared DEV/probe inputs.

Retrieval R@64 is a candidate-availability ceiling, not final top-1 accuracy. Dense-only metrics, memory R@10, full retrieval and final decisions must be reported separately.
