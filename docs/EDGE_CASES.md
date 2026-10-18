# Edge cases and practical limits

| Case | Handling / evidence |
|---|---|
| OCR joins/truncations/spelling | Guarded mention-only correction/segmentation; original view retained |
| MS / RA / PID collisions | Evidence-gated readings and contextual candidate scoring; no universal expansion |
| Inequalities, signs, measurements | Protected clinical symbols, units, stage/type/grade values and numerals |
| Missing or repeated mention in chunk | No-context/ambiguity flags; whole-mention anchoring; no chunk spell correction |
| Negation/history/family | Cue features and candidate qualifier features; no hidden rule that relabels a term |
| Unspecified versus specific ICD | Supported refinement and legal-header policy; no unsupported laterality or acuity invention |
| ICD seventh character | Official meaning plus context/prior evidence and legal-code validation |
| SNOMED lexical look-alikes | Soft structural features and bounded expansion; no claim of official IS-A relations |
| Missing/unknown type | Train-only original-SapBERT classifier; types remain soft priors |
| Missing gold from candidate pool | Dedicated missing-code probability; never fabricate a gold label |
| Rare/unseen codes | Dictionary aliases + dense/lexical retrieval, separate frequency buckets and probes |
| New dictionary release | Raw workbook hash selects a fresh terminology namespace; rebuild and re-index |
| Invalid gold L6.0 | Retained/flagged; cannot emit a non-dictionary code to improve measured accuracy |
| ICD Z99.3 marked ProcedureCode | Preserved exactly as supplied; included in all ICD denominators |
| Auxiliary opposite-system candidate | Available for joint decisions; never treated as a supplied gold label |
| GPU/backend differences | Fingerprinted caches and disclosed bf16 differences; no incompatible reuse |

Incoming R@64 does not imply the reranker or final decision chooses gold. The final minimum can be much lower than candidate recall. Final failure counts/examples belong in the later error analysis; no error count is invented in this source snapshot. Use de-identified or synthetic examples in public-facing narrative; keep chart details inside the private submission.
