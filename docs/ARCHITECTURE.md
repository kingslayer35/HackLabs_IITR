# Architecture and retained decisions

| Order | Stage | Inputs | Outputs / safeguards |
|---|---|---|---|
| 1 | Mention preprocessing | Mention, optional chunk, optional type | Conservative V0–V4, anchored context, cues, literal codes; protected clinical values |
| 2 | Terminology structure | Supplied ICD/SNOMED workbooks, CDC XML | Legal string codes, ICD hierarchy/seventh meanings, SNOMED official tags/top levels and conservative links |
| 3 | Candidate retrieval | Views, dictionary aliases, train-only mention memory | Exact/character/BM25/dense/memory channels; RRF k=60; top 50 and bounded expansion to 64 |
| 4 | Contextual reranking | Mention, chunk window and candidate cards | Raw pretrained or LoRA Qwen yes-minus-no scores, maximum 512 tokens |
| 5 | Independent decision layer | Candidate features, train priors and reranker scores | Fitted fusion; supported specificity; optional guarded coupling; calibrated confidence; missing-code probability |
| 6 | Validation/export | Both systems' final decisions | One legal code per system, original input order, confidence diagnostics |

SapBERT is the sole dense model; Qwen3-Reranker-0.6B is the sole reranker. The latest query encoders are trained against original dictionary vectors; updating dictionary vectors with adapted weights is intentionally forbidden. Type inference remains based on original SapBERT vectors. Each system's output is predicted from its own candidate set, with soft joint agreement rather than a one-way crosswalk.

The retained retrieval policy is recorded in config, including disabled CDC/bridge/family options. Hierarchy code is still included because structure supports features, validation and specificity independently of optional retrieval expansion.

The production source is reorganized by component. Import/source-provenance paths were rewritten; `SOURCE_MANIFEST.json` records the original and packaged file hashes. Portable deployment adds an explicit train-only evaluation profile and a standalone release manifest, so it does not depend on abandoned all-label retraining or absolute research-run paths. No claim is made that moving files alone re-ran the main evaluation.

Final reports must disclose model, preprocessing, inference backend and GPU architecture. A model's loss, incoming R@64 and complete-pipeline top-1 accuracy are distinct measurements.
