# TermMap — finalized DEV and internal TEST results

All four complete flows are finished. **F11 reaches 51.30% on DEV and 50.71% on internal TEST**, using the primary score **min(ICD exact-code accuracy, SNOMED all-types exact-code accuracy)**. TEST F00 is 42.71%; F11 improves it by **7.99 percentage points**, paired grouped 95% CI **[5.61, 10.37]**. SNOMED remains the limiting system. The problem statement’s roughly 95% accuracy target for both systems is **not met**.

This README is the analytical paper/report entry point. [Full stage/category tables](Final_all_flows_DEV_TEST.md), [664-row aggregate CSV](Final_all_flows_metrics.csv), [complete saved metric JSON](Final_all_flows_full_metrics.json), and [additional judging diagnostics](Judging_metrics.json) provide the detailed saved versions. Clinical inputs and per-row diagnostics stay in local ignored artifacts.

## Models, sets and evaluation rules

| Flow | Retriever query / memory encoder | Reranker |
| --- | --- | --- |
| F00 | Original SapBERT | Original pretrained Qwen3-Reranker-0.6B; no LoRA |
| F10 | Selected asymmetric SapBERT P2-v2 | Original Qwen; no LoRA |
| F01 | Original SapBERT | Revised-input Qwen LoRA, user-selected epoch 2 |
| F11 | Selected asymmetric SapBERT P2-v2 | Revised-input Qwen LoRA, user-selected epoch 2 |

Every flow uses latest preprocessing (`reviewed_20261018_v2`) and its own fitted decision layer. F00 includes the full architecture: retrieval channels, reranking, fusion, prescribed specificity/coupling keep/drop decisions, calibrated confidence, missing-code detection and final validation. Both its neural models use original pretrained weights. Dictionary/alias vectors and type-inference vectors remain original SapBERT in all flows; only adapted query and memory encoders change in F10/F11. The revised epoch-2 adapter supersedes earlier-LoRA results for the current F01/F11 tables.

The retrieval stage combines exact, character, BM25, SapBERT dense and train-mention-memory channels with reciprocal-rank fusion and hierarchy/qualifier expansion. The SNOMED→ICD bridge remains disabled by the saved retrieval policy. Qwen receives the mention, scoped chart context and structured candidate descriptions; it does not score only the code string. Tags and top-level types are soft features, never candidate filters. Known input subconcepts are used as supplied; missing ones use the train-fitted classifier. Consequently, the main chart evaluation and the benchmark without a supplied subconcept have different input-information conditions.

| Set | Supplied gold rows | Protocol |
| --- | --- | --- |
| Training | 6,751 | Training-only priors/memory; no training accuracy presented as held-out performance |
| DEV | 1,368: ICD 327; SNOMED 1,041 (DiagnosisCode 386) | Per-flow nested grouped cross-validation; no evaluated group fits its own decisions |
| Internal TEST | 1,527: ICD 326; SNOMED 1,201 (DiagnosisCode 390) | Frozen full-DEV decisions; no TEST fitting or tuning |
| Dictionary PROBE | 6,000: 1,500 codes/system × clean/noisy | Held-out aliases excluded from training and evaluation indexes; separate from chart sets |

DEV and TEST groups do not overlap. TEST is the saved internal held-out split, **not an organizer-provided hidden evaluation set**. Both systems are predicted for every input, but only its supplied gold system is evaluated. Auxiliary opposite-system predictions never become labels. DEV retains its one ICD ProcedureCode label; all 327 ICD rows count. Empty subtypes, including MedicationProfileMedication, are NA rather than zero accuracy. The split is group-based and stratified, so realized row proportions differ from the nominal 70/15/15 group split.

Accuracy equals recall@1. R@k is exact-gold presence among the first k ranked codes. Primary minimum uses ICD and **all** SNOMED types; SNOMED DiagnosisCode and paired-both-correct are supplementary. Candidate R@64 is an availability ceiling for choosing within that pool, not expected final accuracy. The specificity layer can add an ICD descendant beyond the incoming pool; some final rankings have 65 codes. Reported R@200 is within saved rankings, not a separate 200-candidate search. All rates below are percentages; CIs use 1,000 grouped bootstrap draws, seed 13. Deltas are paired on identical groups.

### Checkpoint references and the lexical-only diagnostic

The current checkpoint mapping is **C1 = F00 stage 1; C2 = F10 stage 1; C3 = F11 stage 2; C4 = F11 stage 3**. Their complete current DEV/TEST tables follow separately. C0 below is the saved original char-TF-IDF-only DEV diagnostic on the same split. It has no reranker or decision layer and is not a recomputed current-preprocessing TEST result. C0 is supplementary; F00 remains the complete original-model baseline.

| C0 DEV system | Bucket | n | Accuracy [CI] | R@64 [CI] | Selection given lexical pool | ICD category / SNOMED tag |
| --- | --- | --- | --- | --- | --- | --- |
| ICD | all | 327 | 18.96 [14.63, 23.68] | 59.94 [54.33, 65.48] | 31.63 | 38.23 |
| ICD | common | 78 | 12.82 [5.55, 20.48] | 47.44 [35.13, 59.21] | 27.03 | 28.21 |
| ICD | rare | 168 | 19.05 [13.64, 25.00] | 62.50 [54.28, 70.41] | 30.48 | 38.69 |
| ICD | unseen | 81 | 24.69 [14.81, 34.94] | 66.67 [55.00, 76.74] | 37.04 | 46.91 |
| SNOMED | all | 1041 | 25.84 [23.11, 28.68] | 65.51 [62.67, 68.48] | 39.44 | 45.24 |
| SNOMED | common | 63 | 19.05 [10.60, 30.16] | 52.38 [40.29, 65.08] | 36.36 | 44.44 |
| SNOMED | rare | 468 | 24.36 [20.42, 28.30] | 66.45 [62.23, 71.00] | 36.66 | 42.31 |
| SNOMED | unseen | 510 | 28.04 [24.15, 32.38] | 66.27 [61.97, 70.35] | 42.31 | 48.04 |

## DEV — grouped out-of-fold results

### 1. Before reranker: complete retrieval

| Flow | Gold system | n | Accuracy = R@1 [95% CI] | R@10 | R@32 | R@64 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 327 | 39.45 [34.04, 44.92] | 67.28 | 79.20 | 85.93 [81.82, 89.81] |
| F00 | SNOMED | 1041 | 36.79 [33.96, 39.52] | 69.16 | 79.15 | 83.48 [81.01, 85.82] |
| F10 | ICD | 327 | 43.43 [37.54, 49.07] | 77.06 | 87.77 | 90.83 [87.65, 93.92] |
| F10 | SNOMED | 1041 | 37.08 [34.14, 39.91] | 72.62 | 82.80 | 85.78 [83.61, 88.07] |
| F01 | ICD | 327 | 39.45 [34.04, 44.92] | 67.28 | 79.20 | 85.93 [81.82, 89.81] |
| F01 | SNOMED | 1041 | 36.79 [33.96, 39.52] | 69.16 | 79.15 | 83.48 [81.01, 85.82] |
| F11 | ICD | 327 | 43.43 [37.54, 49.07] | 77.06 | 87.77 | 90.83 [87.65, 93.92] |
| F11 | SNOMED | 1041 | 37.08 [34.14, 39.91] | 72.62 | 82.80 | 85.78 [83.61, 88.07] |

| Flow | SNOMED DiagnosisCode accuracy | Primary minimum | DiagnosisCode minimum (secondary) |
| --- | --- | --- | --- |
| F00 | 45.34 | 36.79 [33.58, 39.39] | 39.45 |
| F10 | 44.04 | 37.08 [34.14, 40.06] | 43.43 |
| F01 | 45.34 | 36.79 [33.58, 39.39] | 39.45 |
| F11 | 44.04 | 37.08 [34.14, 40.06] | 43.43 |

### 2. After reranker, before the decision layer

| Flow | Gold system | n | Accuracy = R@1 [95% CI] | R@10 | R@32 | R@64 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 327 | 32.42 [27.27, 37.54] | 69.72 | 81.65 | 85.93 [81.82, 89.81] |
| F00 | SNOMED | 1041 | 30.07 [26.99, 32.94] | 69.16 | 81.17 | 83.48 [81.01, 85.82] |
| F10 | ICD | 327 | 31.19 [26.27, 36.44] | 71.87 | 85.93 | 90.83 [87.65, 93.92] |
| F10 | SNOMED | 1041 | 30.26 [27.16, 33.17] | 69.07 | 82.71 | 85.78 [83.61, 88.07] |
| F01 | ICD | 327 | 55.66 [50.31, 61.15] | 84.71 | 85.93 | 85.93 [81.82, 89.81] |
| F01 | SNOMED | 1041 | 45.44 [42.31, 48.50] | 76.95 | 83.00 | 83.48 [81.01, 85.82] |
| F11 | ICD | 327 | 54.43 [49.19, 60.06] | 88.38 | 90.83 | 90.83 [87.65, 93.92] |
| F11 | SNOMED | 1041 | 44.48 [41.10, 47.68] | 77.62 | 85.21 | 85.78 [83.61, 88.07] |

| Flow | SNOMED DiagnosisCode accuracy | Primary minimum | DiagnosisCode minimum (secondary) |
| --- | --- | --- | --- |
| F00 | 35.23 | 30.07 [26.79, 32.85] | 32.42 |
| F10 | 36.01 | 30.26 [26.21, 32.55] | 31.19 |
| F01 | 53.63 | 45.44 [42.39, 48.65] | 53.63 |
| F11 | 53.37 | 44.48 [41.20, 47.61] | 53.37 |

### 3. After the final decision layer

| Flow | Gold system | n | Accuracy = R@1 [95% CI] | R@10 | R@32 | R@64 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 327 | 52.91 [47.54, 58.36] | 81.35 | 85.63 | 85.93 [81.82, 89.81] |
| F00 | SNOMED | 1041 | 43.13 [40.04, 46.11] | 76.95 | 82.61 | 83.48 [81.01, 85.82] |
| F10 | ICD | 327 | 56.88 [51.46, 62.19] | 86.85 | 90.83 | 90.83 [87.65, 93.92] |
| F10 | SNOMED | 1041 | 45.24 [42.08, 48.42] | 80.02 | 84.92 | 85.78 [83.61, 88.07] |
| F01 | ICD | 327 | 60.86 [55.66, 65.85] | 84.40 | 86.24 | 86.24 [82.12, 90.03] |
| F01 | SNOMED | 1041 | 50.53 [47.57, 53.54] | 79.15 | 83.29 | 83.48 [81.01, 85.82] |
| F11 | ICD | 327 | 66.97 [61.81, 71.81] | 89.30 | 90.83 | 90.83 [87.65, 93.92] |
| F11 | SNOMED | 1041 | 51.30 [48.23, 54.50] | 81.17 | 85.49 | 85.78 [83.61, 88.07] |

| Flow | SNOMED DiagnosisCode accuracy | Primary minimum | DiagnosisCode minimum (secondary) |
| --- | --- | --- | --- |
| F00 | 51.81 | 43.13 [40.18, 46.19] | 51.81 |
| F10 | 52.59 | 45.24 [41.96, 48.26] | 52.59 |
| F01 | 57.77 | 50.53 [47.60, 53.70] | 57.77 |
| F11 | 57.25 | 51.30 [48.12, 54.47] | 57.25 |

### Final paired improvements

| Comparison | ICD delta [95% CI] | SNOMED all delta [95% CI] | Primary-minimum delta [95% CI] |
| --- | --- | --- | --- |
| F01-F00 | +7.95 [3.42, 12.61] | +7.40 [5.12, 9.71] | +7.40 [5.12, 9.71] |
| F10-F00 | +3.98 [-0.31, 7.99] | +2.11 [0.38, 3.82] | +2.11 [0.38, 3.82] |
| F11-F00 | +14.07 [8.96, 19.18] | +8.17 [5.74, 10.52] | +8.17 [5.74, 10.52] |
| F11-F01 | +6.12 [2.42, 10.29] | +0.77 [-1.02, 2.40] | +0.77 [-1.02, 2.40] |
| F11-F10 | +10.09 [5.55, 14.54] | +6.05 [4.05, 8.04] | +6.05 [4.05, 8.04] |

### Final supplied categories and paired-input diagnostic

| SNOMED supplied category | n | F00 | F10 | F01 | F11 |
| --- | --- | --- | --- | --- | --- |
| Device | 67 | 37.31 | 34.33 | 37.31 | 43.28 |
| DiagnosisCode | 386 | 51.81 | 52.59 | 57.77 | 57.25 |
| EncounterType | 56 | 41.07 | 48.21 | 51.79 | 53.57 |
| LabService | 137 | 36.50 | 41.61 | 48.91 | 49.64 |
| ProcedureCode | 395 | 38.23 | 40.76 | 46.08 | 47.09 |

| Flow | Paired inputs n | Both-correct accuracy [95% CI] |
| --- | --- | --- |
| F00 | 307 | 34.85 [29.67, 40.19] |
| F10 | 307 | 39.74 [34.51, 45.14] |
| F01 | 307 | 40.72 [35.20, 46.41] |
| F11 | 307 | 46.58 [41.17, 52.34] |

Both-correct is supplementary and does **not** replace the minimum of the two system accuracies. All ICD rows, including the supplied DEV ProcedureCode observation, remain in the ICD denominator; its separate n=1 outcome is in the full category tables.

### Long tail: final performance and candidate selection

Unseen = 0 training occurrences of the code; rare = 1–5; common = more than 5. Frequencies use the training split only. Conditional selection is final accuracy **among inputs whose gold was in the incoming retrieval pool**; specificity rescues outside that pool are counted separately. This corrects the ambiguity of historical `selection_given_pool`, whose denominator used the final ranking.

| Flow | System | Bucket | n | Final accuracy [CI] | Incoming R@64 [CI] | Selection given incoming gold [CI] | Outside-pool rescues | ICD category / SNOMED tag [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | unseen | 81 | 33.33 [21.69, 44.58] | 76.54 [66.26, 85.19] | 43.55 [30.51, 56.26] | 0 | 54.32 [43.21, 65.39] |
| F00 | ICD | rare | 168 | 62.50 [55.62, 69.50] | 91.67 [87.57, 95.71] | 68.18 [61.04, 75.16] | 0 | 73.21 [66.66, 79.88] |
| F00 | ICD | common | 78 | 52.56 [41.77, 62.96] | 83.33 [73.97, 91.03] | 63.08 [50.77, 75.00] | 0 | 66.67 [55.26, 76.47] |
| F00 | SNOMED | unseen | 510 | 28.24 [24.45, 32.39] | 77.65 [73.89, 81.31] | 36.36 [31.37, 41.22] | 0 | 56.27 [51.43, 60.71] |
| F00 | SNOMED | rare | 468 | 57.91 [53.29, 62.67] | 89.32 [86.37, 91.96] | 64.83 [60.10, 69.54] | 0 | 76.50 [72.69, 80.72] |
| F00 | SNOMED | common | 63 | 53.97 [42.86, 65.57] | 87.30 [78.46, 95.24] | 61.82 [50.00, 74.55] | 0 | 79.37 [68.85, 88.53] |
| F10 | ICD | unseen | 81 | 39.51 [27.84, 51.16] | 81.48 [71.95, 90.00] | 48.48 [36.92, 60.61] | 0 | 55.56 [45.00, 66.25] |
| F10 | ICD | rare | 168 | 65.48 [58.28, 72.73] | 93.45 [89.65, 96.91] | 70.06 [63.23, 77.36] | 0 | 78.57 [72.00, 84.57] |
| F10 | ICD | common | 78 | 56.41 [44.44, 67.95] | 94.87 [89.33, 98.75] | 59.46 [47.89, 71.43] | 0 | 73.08 [61.25, 82.67] |
| F10 | SNOMED | unseen | 510 | 29.41 [25.57, 33.53] | 78.24 [74.40, 81.80] | 37.59 [32.72, 42.23] | 0 | 58.43 [53.60, 62.48] |
| F10 | SNOMED | rare | 468 | 60.04 [55.53, 64.63] | 92.95 [90.44, 95.29] | 64.60 [60.04, 69.36] | 0 | 77.35 [73.53, 81.49] |
| F10 | SNOMED | common | 63 | 63.49 [53.03, 75.00] | 93.65 [87.10, 98.48] | 67.80 [57.62, 78.95] | 0 | 85.71 [77.05, 93.55] |
| F01 | ICD | unseen | 81 | 39.51 [27.85, 51.33] | 76.54 [66.26, 85.19] | 51.61 [38.97, 63.94] | 0 | 55.56 [43.90, 66.68] |
| F01 | ICD | rare | 168 | 71.43 [65.17, 77.84] | 91.67 [87.57, 95.71] | 77.27 [70.78, 83.54] | 1 | 82.74 [76.96, 88.13] |
| F01 | ICD | common | 78 | 60.26 [49.40, 70.24] | 83.33 [73.97, 91.03] | 72.31 [61.19, 82.54] | 0 | 74.36 [63.51, 82.72] |
| F01 | SNOMED | unseen | 510 | 36.67 [32.40, 41.12] | 77.65 [73.89, 81.31] | 47.22 [42.50, 51.88] | 0 | 63.33 [58.77, 67.84] |
| F01 | SNOMED | rare | 468 | 63.46 [59.39, 67.69] | 89.32 [86.37, 91.96] | 71.05 [66.51, 75.52] | 0 | 78.85 [75.38, 82.51] |
| F01 | SNOMED | common | 63 | 66.67 [56.25, 77.78] | 87.30 [78.46, 95.24] | 76.36 [64.91, 87.04] | 0 | 87.30 [78.69, 95.08] |
| F11 | ICD | unseen | 81 | 46.91 [35.00, 58.98] | 81.48 [71.95, 90.00] | 57.58 [45.31, 69.84] | 0 | 59.26 [48.72, 70.51] |
| F11 | ICD | rare | 168 | 73.81 [67.27, 80.36] | 93.45 [89.65, 96.91] | 78.98 [72.33, 85.53] | 0 | 83.93 [78.12, 89.41] |
| F11 | ICD | common | 78 | 73.08 [62.81, 82.50] | 94.87 [89.33, 98.75] | 77.03 [67.53, 85.53] | 0 | 84.62 [76.62, 92.41] |
| F11 | SNOMED | unseen | 510 | 36.27 [31.85, 40.64] | 78.24 [74.40, 81.80] | 46.37 [41.22, 51.20] | 0 | 63.73 [58.87, 68.08] |
| F11 | SNOMED | rare | 468 | 65.17 [60.77, 69.80] | 92.95 [90.44, 95.29] | 70.11 [65.31, 74.71] | 0 | 81.41 [77.80, 84.97] |
| F11 | SNOMED | common | 63 | 69.84 [59.02, 80.00] | 93.65 [87.10, 98.48] | 74.58 [64.91, 85.00] | 0 | 93.65 [87.30, 98.48] |

Paired changes in the primary minimum within each frequency bucket (percentage points):

| Comparison | Unseen delta [CI] | Rare delta [CI] | Common delta [CI] |
| --- | --- | --- | --- |
| F01-F00 | +8.43 [1.40, 12.37] | +5.56 [2.24, 9.48] | +7.69 [0.00, 17.48] |
| F10-F00 | +1.18 [-0.99, 6.38] | +2.14 [-0.43, 5.17] | +3.85 [-4.17, 14.52] |
| F11-F00 | +8.04 [4.75, 13.16] | +7.26 [3.95, 11.59] | +17.28 [8.93, 28.16] |
| F11-F01 | -0.39 [-2.42, 7.35] | +1.71 [-1.03, 4.60] | +9.58 [-1.61, 20.06] |
| F11-F10 | +6.86 [3.85, 9.96] | +5.13 [2.04, 8.66] | +13.43 [2.86, 24.71] |

### Error location and stage changes

These are reproducible pipeline-location diagnostics. Among wrong predictions, absence from the incoming pool takes precedence, followed by detected over/under-specificity, then other wrong selection. The counts do not establish that the gold annotation is wrong or that a convention caused an error. Clinical adjudication of those causes remains separate. Rescues/demotions compare top-1 correctness across stages on the same supplied labels.

| Flow | System | Errors: gold absent | Errors: specificity | Errors: other selection | Reranker rescues / demotions | Decision rescues / demotions |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 46 | 11 | 97 | 34 / 57 | 83 / 16 |
| F00 | SNOMED | 172 | 5 | 415 | 124 / 194 | 214 / 78 |
| F10 | ICD | 30 | 14 | 97 | 30 / 70 | 98 / 14 |
| F10 | SNOMED | 148 | 7 | 415 | 121 / 192 | 223 / 67 |
| F01 | ICD | 45 | 8 | 75 | 77 / 24 | 31 / 14 |
| F01 | SNOMED | 172 | 4 | 339 | 200 / 110 | 110 / 57 |
| F11 | ICD | 30 | 8 | 70 | 71 / 35 | 50 / 9 |
| F11 | SNOMED | 148 | 3 | 356 | 191 / 114 | 133 / 62 |

### Coverage, confidence and calibration

All inputs receive both legal codes: output coverage is **100%**, actual abstention **0%**. `low_conf` is a review flag, not omission of a prediction. The following evaluates the existing DEV-fitted flag without choosing any TEST threshold. Brier and ten-equal-width-bin ECE are diagnostic, lower is better; ECE is not an accuracy percentage.

| Flow | System | Unflagged coverage % | Unflagged n | Unflagged accuracy [CI] | Flagged % | Brier | ECE | Missing-code AUROC % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 91.74 | 300 | 56.33 [50.68, 61.92] | 8.26 | 0.1771 | 0.0885 | 79.14 |
| F00 | SNOMED | 95.77 | 997 | 44.43 [41.38, 47.48] | 4.23 | 0.1852 | 0.0265 | 73.05 |
| F10 | ICD | 93.27 | 305 | 59.67 [54.61, 64.92] | 6.73 | 0.1758 | 0.0922 | 79.67 |
| F10 | SNOMED | 95.68 | 996 | 46.89 [43.87, 50.05] | 4.32 | 0.1754 | 0.0384 | 75.05 |
| F01 | ICD | 94.50 | 309 | 62.46 [57.14, 67.77] | 5.50 | 0.1813 | 0.0959 | 82.72 |
| F01 | SNOMED | 95.00 | 989 | 52.78 [49.74, 55.99] | 5.00 | 0.1859 | 0.0382 | 77.03 |
| F11 | ICD | 94.50 | 309 | 70.23 [64.96, 74.84] | 5.50 | 0.1796 | 0.1163 | 88.00 |
| F11 | SNOMED | 94.91 | 988 | 53.85 [50.75, 57.16] | 5.09 | 0.1837 | 0.0351 | 77.52 |

### Hierarchy, specificity and SNOMED error classes

| Flow | ICD category % | SN tag % | SN top-level % | SN same-core % | ICD over / under % | SN over / under % | Illegal ICD headers | SN near-miss errors | Same-tag errors | Same-top-level errors | Wrong-top-level errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 66.97 | 66.76 | 71.57 | 44.38 | 1.22 / 2.14 | 0.10 / 0.38 | 0 | 10 | 240 | 50 | 292 |
| F10 | 71.56 | 68.59 | 72.91 | 46.88 | 2.45 / 1.83 | 0.19 / 0.58 | 0 | 13 | 235 | 45 | 277 |
| F01 | 74.01 | 71.76 | 75.60 | 51.78 | 1.22 / 1.22 | 0.10 / 0.29 | 0 | 8 | 216 | 40 | 251 |
| F11 | 77.98 | 73.49 | 77.14 | 52.74 | 1.22 / 1.53 | 0.10 / 0.29 | 0 | 10 | 226 | 38 | 233 |

### Confidence-ranked risk–coverage

Error percentage among retained highest-confidence supplied labels, pooled across systems. This is a diagnostic curve, not a change to the always-predict output policy.

| Flow | 10% coverage | 25% | 50% | 75% | 90% | 95% | 100% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 9.49 | 19.59 | 31.73 | 44.15 | 50.81 | 52.85 | 54.53 |
| F10 | 8.03 | 14.91 | 27.92 | 41.33 | 47.73 | 50.08 | 51.97 |
| F01 | 7.30 | 13.74 | 24.27 | 35.58 | 42.69 | 45.00 | 47.00 |
| F11 | 7.30 | 11.70 | 22.08 | 34.11 | 40.58 | 42.46 | 44.96 |

**DEV verdict.** Both model adaptations improve the full pipeline over F00; the final decision layer repairs many raw-reranker mistakes. The raw original Qwen rankings can be worse than retrieval top-1, so high incoming R@64 does not ensure a correct first choice. DEV values are out-of-fold decision estimates; full-DEV fitted decisions are used only for TEST/deployment-style diagnostics.

## Internal TEST — frozen decision results

### 1. Before reranker: complete retrieval

| Flow | Gold system | n | Accuracy = R@1 [95% CI] | R@10 | R@32 | R@64 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 326 | 40.80 [35.58, 45.54] | 69.94 | 81.29 | 86.20 [82.27, 90.52] |
| F00 | SNOMED | 1201 | 36.72 [33.72, 39.46] | 67.11 | 75.35 | 80.02 [77.75, 82.19] |
| F10 | ICD | 326 | 44.48 [39.31, 50.00] | 76.07 | 87.73 | 91.41 [88.14, 94.45] |
| F10 | SNOMED | 1201 | 36.39 [33.39, 39.07] | 70.27 | 78.52 | 82.10 [79.79, 84.17] |
| F01 | ICD | 326 | 40.80 [35.58, 45.54] | 69.94 | 81.29 | 86.20 [82.27, 90.52] |
| F01 | SNOMED | 1201 | 36.72 [33.72, 39.46] | 67.11 | 75.35 | 80.02 [77.75, 82.19] |
| F11 | ICD | 326 | 44.48 [39.31, 50.00] | 76.07 | 87.73 | 91.41 [88.14, 94.45] |
| F11 | SNOMED | 1201 | 36.39 [33.39, 39.07] | 70.27 | 78.52 | 82.10 [79.79, 84.17] |

| Flow | SNOMED DiagnosisCode accuracy | Primary minimum | DiagnosisCode minimum (secondary) |
| --- | --- | --- | --- |
| F00 | 44.62 | 36.72 [33.57, 39.50] | 40.80 |
| F10 | 44.10 | 36.39 [33.42, 39.17] | 44.10 |
| F01 | 44.62 | 36.72 [33.57, 39.50] | 40.80 |
| F11 | 44.10 | 36.39 [33.42, 39.17] | 44.10 |

### 2. After reranker, before the decision layer

| Flow | Gold system | n | Accuracy = R@1 [95% CI] | R@10 | R@32 | R@64 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 326 | 32.21 [26.86, 37.17] | 71.17 | 85.28 | 86.20 [82.27, 90.52] |
| F00 | SNOMED | 1201 | 29.14 [26.55, 31.61] | 65.36 | 77.52 | 80.02 [77.75, 82.19] |
| F10 | ICD | 326 | 31.29 [25.96, 36.36] | 74.85 | 88.34 | 91.41 [88.14, 94.45] |
| F10 | SNOMED | 1201 | 29.23 [26.77, 31.57] | 66.61 | 78.93 | 82.10 [79.79, 84.17] |
| F01 | ICD | 326 | 58.28 [53.07, 63.06] | 83.13 | 86.20 | 86.20 [82.27, 90.52] |
| F01 | SNOMED | 1201 | 44.05 [41.22, 46.83] | 73.86 | 79.35 | 80.02 [77.75, 82.19] |
| F11 | ICD | 326 | 53.37 [47.80, 58.41] | 87.42 | 90.80 | 91.41 [88.14, 94.45] |
| F11 | SNOMED | 1201 | 43.13 [40.37, 45.72] | 74.27 | 81.35 | 82.10 [79.79, 84.17] |

| Flow | SNOMED DiagnosisCode accuracy | Primary minimum | DiagnosisCode minimum (secondary) |
| --- | --- | --- | --- |
| F00 | 41.03 | 29.14 [26.54, 31.64] | 32.21 |
| F10 | 40.51 | 29.23 [25.96, 31.57] | 31.29 |
| F01 | 57.95 | 44.05 [41.47, 46.93] | 57.95 |
| F11 | 56.92 | 43.13 [40.35, 45.88] | 53.37 |

### 3. After the final decision layer

| Flow | Gold system | n | Accuracy = R@1 [95% CI] | R@10 | R@32 | R@64 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 326 | 53.07 [47.38, 58.03] | 84.05 | 86.50 | 86.50 [82.81, 90.52] |
| F00 | SNOMED | 1201 | 42.71 [39.83, 45.39] | 74.52 | 79.35 | 80.02 [77.75, 82.19] |
| F10 | ICD | 326 | 62.27 [56.88, 67.06] | 86.50 | 91.41 | 91.41 [88.14, 94.45] |
| F10 | SNOMED | 1201 | 44.80 [41.69, 47.49] | 77.35 | 81.43 | 82.10 [79.79, 84.17] |
| F01 | ICD | 326 | 65.03 [60.00, 69.85] | 85.58 | 86.50 | 86.50 [82.81, 90.52] |
| F01 | SNOMED | 1201 | 50.46 [47.39, 53.15] | 76.85 | 79.77 | 80.02 [77.75, 82.19] |
| F11 | ICD | 326 | 68.40 [63.38, 73.45] | 89.88 | 91.41 | 91.41 [88.14, 94.45] |
| F11 | SNOMED | 1201 | 50.71 [47.87, 53.36] | 78.27 | 81.85 | 82.10 [79.79, 84.17] |

| Flow | SNOMED DiagnosisCode accuracy | Primary minimum | DiagnosisCode minimum (secondary) |
| --- | --- | --- | --- |
| F00 | 51.54 | 42.71 [39.91, 45.73] | 51.54 |
| F10 | 55.13 | 44.80 [42.03, 47.91] | 55.13 |
| F01 | 61.54 | 50.46 [47.58, 53.32] | 61.54 |
| F11 | 62.05 | 50.71 [48.12, 53.57] | 62.05 |

### Final paired improvements

| Comparison | ICD delta [95% CI] | SNOMED all delta [95% CI] | Primary-minimum delta [95% CI] |
| --- | --- | --- | --- |
| F01-F00 | +11.96 [7.37, 16.72] | +7.74 [5.36, 9.97] | +7.74 [5.36, 9.97] |
| F10-F00 | +9.20 [5.76, 12.93] | +2.08 [0.58, 3.63] | +2.08 [0.58, 3.63] |
| F11-F00 | +15.34 [11.04, 20.69] | +7.99 [5.61, 10.37] | +7.99 [5.61, 10.37] |
| F11-F01 | +3.37 [0.00, 6.96] | +0.25 [-1.13, 1.92] | +0.25 [-1.13, 1.92] |
| F11-F10 | +6.13 [2.50, 10.29] | +5.91 [3.74, 7.92] | +5.91 [3.74, 7.92] |

### Final supplied categories and paired-input diagnostic

| SNOMED supplied category | n | F00 | F10 | F01 | F11 |
| --- | --- | --- | --- | --- | --- |
| Device | 78 | 53.85 | 52.56 | 60.26 | 57.69 |
| DiagnosisCode | 390 | 51.54 | 55.13 | 61.54 | 62.05 |
| EncounterType | 97 | 45.36 | 44.33 | 48.45 | 48.45 |
| LabService | 144 | 37.50 | 43.75 | 44.44 | 44.44 |
| ProcedureCode | 492 | 34.96 | 35.77 | 42.28 | 42.89 |

| Flow | Paired inputs n | Both-correct accuracy [95% CI] |
| --- | --- | --- |
| F00 | 300 | 37.67 [32.11, 42.60] |
| F10 | 300 | 43.67 [37.84, 49.64] |
| F01 | 300 | 50.33 [43.93, 56.21] |
| F11 | 300 | 51.67 [45.67, 57.34] |

Both-correct is supplementary and does **not** replace the minimum of the two system accuracies. All ICD rows, including the supplied DEV ProcedureCode observation, remain in the ICD denominator; its separate n=1 outcome is in the full category tables.

### Long tail: final performance and candidate selection

Unseen = 0 training occurrences of the code; rare = 1–5; common = more than 5. Frequencies use the training split only. Conditional selection is final accuracy **among inputs whose gold was in the incoming retrieval pool**; specificity rescues outside that pool are counted separately. This corrects the ambiguity of historical `selection_given_pool`, whose denominator used the final ranking.

| Flow | System | Bucket | n | Final accuracy [CI] | Incoming R@64 [CI] | Selection given incoming gold [CI] | Outside-pool rescues | ICD category / SNOMED tag [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | unseen | 77 | 42.86 [32.43, 54.29] | 76.62 [66.26, 86.11] | 55.93 [42.86, 69.23] | 0 | 68.83 [58.44, 78.38] |
| F00 | ICD | rare | 156 | 57.05 [48.38, 65.00] | 88.46 [82.91, 93.79] | 63.77 [55.48, 71.74] | 1 | 67.31 [59.03, 74.50] |
| F00 | ICD | common | 93 | 54.84 [44.44, 65.27] | 90.32 [84.21, 95.70] | 60.71 [50.62, 70.93] | 0 | 60.22 [50.00, 70.12] |
| F00 | SNOMED | unseen | 621 | 29.31 [25.32, 32.83] | 72.79 [68.90, 75.97] | 40.27 [35.34, 44.70] | 0 | 57.00 [53.22, 60.96] |
| F00 | SNOMED | rare | 494 | 55.67 [51.38, 60.32] | 87.65 [84.75, 90.47] | 63.51 [59.23, 67.77] | 0 | 73.08 [68.71, 77.68] |
| F00 | SNOMED | common | 86 | 65.12 [55.43, 76.47] | 88.37 [81.61, 95.06] | 73.68 [63.75, 83.33] | 0 | 76.74 [67.67, 86.25] |
| F10 | ICD | unseen | 77 | 51.95 [40.84, 63.64] | 81.82 [72.22, 90.54] | 63.49 [50.85, 75.44] | 0 | 68.83 [58.10, 79.22] |
| F10 | ICD | rare | 156 | 66.67 [58.66, 74.03] | 95.51 [91.67, 98.73] | 69.80 [62.41, 77.08] | 0 | 74.36 [66.66, 81.64] |
| F10 | ICD | common | 93 | 63.44 [53.84, 72.92] | 92.47 [86.81, 96.94] | 68.60 [58.24, 79.07] | 0 | 69.89 [60.20, 79.07] |
| F10 | SNOMED | unseen | 621 | 31.56 [27.71, 34.99] | 73.11 [69.23, 76.40] | 43.17 [38.08, 47.30] | 0 | 59.42 [55.72, 63.56] |
| F10 | SNOMED | rare | 494 | 58.10 [53.86, 62.65] | 91.50 [88.76, 93.65] | 63.50 [59.37, 67.87] | 0 | 76.11 [72.12, 80.34] |
| F10 | SNOMED | common | 86 | 63.95 [53.84, 74.70] | 93.02 [87.88, 97.59] | 68.75 [58.62, 79.52] | 0 | 81.40 [72.94, 88.75] |
| F01 | ICD | unseen | 77 | 57.14 [45.78, 68.12] | 76.62 [66.26, 86.11] | 74.58 [60.71, 86.67] | 0 | 71.43 [60.81, 81.69] |
| F01 | ICD | rare | 156 | 67.31 [58.82, 74.50] | 88.46 [82.91, 93.79] | 75.36 [68.15, 82.01] | 1 | 75.64 [67.84, 82.29] |
| F01 | ICD | common | 93 | 67.74 [58.16, 77.18] | 90.32 [84.21, 95.70] | 75.00 [66.26, 83.75] | 0 | 70.97 [61.54, 80.86] |
| F01 | SNOMED | unseen | 621 | 39.45 [34.51, 43.46] | 72.79 [68.90, 75.97] | 54.20 [48.89, 59.02] | 0 | 65.06 [61.26, 68.45] |
| F01 | SNOMED | rare | 494 | 60.93 [56.49, 65.20] | 87.65 [84.75, 90.47] | 69.52 [65.53, 73.57] | 0 | 76.72 [72.97, 80.61] |
| F01 | SNOMED | common | 86 | 69.77 [58.69, 82.28] | 88.37 [81.61, 95.06] | 78.95 [67.50, 88.89] | 0 | 83.72 [73.91, 93.59] |
| F11 | ICD | unseen | 77 | 57.14 [45.83, 67.57] | 81.82 [72.22, 90.54] | 69.84 [56.92, 81.16] | 0 | 71.43 [60.87, 81.93] |
| F11 | ICD | rare | 156 | 73.72 [66.19, 80.63] | 95.51 [91.67, 98.73] | 77.18 [70.55, 83.34] | 0 | 83.33 [76.25, 89.61] |
| F11 | ICD | common | 93 | 68.82 [58.69, 77.89] | 92.47 [86.81, 96.94] | 74.42 [64.70, 84.53] | 0 | 72.04 [62.62, 81.11] |
| F11 | SNOMED | unseen | 621 | 38.49 [33.76, 42.21] | 73.11 [69.23, 76.40] | 52.64 [47.34, 56.77] | 0 | 63.93 [60.04, 67.52] |
| F11 | SNOMED | rare | 494 | 63.16 [58.79, 67.51] | 91.50 [88.76, 93.65] | 69.03 [64.71, 73.08] | 0 | 81.17 [77.59, 84.42] |
| F11 | SNOMED | common | 86 | 67.44 [52.88, 81.58] | 93.02 [87.88, 97.59] | 72.50 [59.34, 86.67] | 0 | 84.88 [77.19, 92.41] |

Paired changes in the primary minimum within each frequency bucket (percentage points):

| Comparison | Unseen delta [CI] | Rare delta [CI] | Common delta [CI] |
| --- | --- | --- | --- |
| F01-F00 | +10.14 [6.97, 13.30] | +5.26 [2.48, 11.70] | +12.90 [1.33, 21.58] |
| F10-F00 | +2.25 [0.30, 4.45] | +2.43 [0.24, 9.82] | +8.60 [-1.98, 14.47] |
| F11-F00 | +9.18 [6.10, 12.06] | +7.49 [4.36, 14.86] | +12.60 [-1.22, 22.00] |
| F11-F01 | -0.97 [-2.86, 1.08] | +2.23 [-0.18, 5.20] | -0.30 [-7.60, 6.10] |
| F11-F10 | +6.92 [3.86, 9.55] | +5.06 [1.73, 8.16] | +4.00 [-4.95, 13.16] |

### Error location and stage changes

These are reproducible pipeline-location diagnostics. Among wrong predictions, absence from the incoming pool takes precedence, followed by detected over/under-specificity, then other wrong selection. The counts do not establish that the gold annotation is wrong or that a convention caused an error. Clinical adjudication of those causes remains separate. Rescues/demotions compare top-1 correctness across stages on the same supplied labels.

| Flow | System | Errors: gold absent | Errors: specificity | Errors: other selection | Reranker rescues / demotions | Decision rescues / demotions |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 44 | 19 | 90 | 36 / 64 | 89 / 21 |
| F00 | SNOMED | 240 | 9 | 439 | 136 / 227 | 249 / 86 |
| F10 | ICD | 28 | 14 | 81 | 33 / 76 | 115 / 14 |
| F10 | SNOMED | 215 | 8 | 440 | 136 / 222 | 266 / 79 |
| F01 | ICD | 44 | 8 | 62 | 80 / 23 | 29 / 7 |
| F01 | SNOMED | 240 | 5 | 350 | 215 / 127 | 124 / 47 |
| F11 | ICD | 28 | 11 | 64 | 69 / 40 | 62 / 13 |
| F11 | SNOMED | 215 | 6 | 371 | 212 / 131 | 143 / 52 |

### Coverage, confidence and calibration

All inputs receive both legal codes: output coverage is **100%**, actual abstention **0%**. `low_conf` is a review flag, not omission of a prediction. The following evaluates the existing DEV-fitted flag without choosing any TEST threshold. Brier and ten-equal-width-bin ECE are diagnostic, lower is better; ECE is not an accuracy percentage.

| Flow | System | Unflagged coverage % | Unflagged n | Unflagged accuracy [CI] | Flagged % | Brier | ECE | Missing-code AUROC % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 91.41 | 298 | 57.05 [51.81, 62.58] | 8.59 | 0.1745 | 0.1104 | 80.89 |
| F00 | SNOMED | 95.17 | 1143 | 44.27 [41.35, 47.22] | 4.83 | 0.1812 | 0.0373 | 76.40 |
| F10 | ICD | 93.87 | 306 | 66.34 [61.22, 71.48] | 6.13 | 0.1688 | 0.1288 | 84.16 |
| F10 | SNOMED | 95.34 | 1145 | 46.64 [43.53, 49.28] | 4.66 | 0.1750 | 0.0540 | 78.63 |
| F01 | ICD | 92.94 | 303 | 68.98 [63.45, 73.70] | 7.06 | 0.1679 | 0.1310 | 83.91 |
| F01 | SNOMED | 95.00 | 1141 | 52.94 [50.00, 55.81] | 5.00 | 0.1831 | 0.0549 | 79.00 |
| F11 | ICD | 93.87 | 306 | 71.90 [66.90, 76.82] | 6.13 | 0.1645 | 0.1346 | 87.57 |
| F11 | SNOMED | 94.09 | 1130 | 53.54 [50.57, 56.10] | 5.91 | 0.1819 | 0.0496 | 80.04 |

### Hierarchy, specificity and SNOMED error classes

| Flow | ICD category % | SN tag % | SN top-level % | SN same-core % | ICD over / under % | SN over / under % | Illegal ICD headers | SN near-miss errors | Same-tag errors | Same-top-level errors | Wrong-top-level errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 65.64 | 65.03 | 69.94 | 43.88 | 1.84 / 3.99 | 0.42 / 0.33 | 0 | 11 | 258 | 59 | 360 |
| F10 | 71.78 | 67.86 | 72.44 | 46.13 | 1.84 / 2.45 | 0.33 / 0.33 | 0 | 13 | 267 | 55 | 328 |
| F01 | 73.31 | 71.19 | 75.35 | 51.46 | 1.53 / 0.92 | 0.25 / 0.17 | 0 | 10 | 243 | 50 | 292 |
| F11 | 77.30 | 72.52 | 76.10 | 51.62 | 1.84 / 1.53 | 0.25 / 0.25 | 0 | 10 | 254 | 43 | 285 |

### Confidence-ranked risk–coverage

Error percentage among retained highest-confidence supplied labels, pooled across systems. This is a diagnostic curve, not a change to the always-predict output policy.

| Flow | 10% coverage | 25% | 50% | 75% | 90% | 95% | 100% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 5.88 | 17.80 | 32.46 | 44.59 | 51.35 | 53.27 | 55.08 |
| F10 | 2.61 | 13.35 | 27.49 | 39.88 | 47.27 | 49.21 | 51.47 |
| F01 | 4.58 | 11.26 | 23.43 | 34.64 | 41.60 | 43.97 | 46.43 |
| F11 | 3.27 | 8.64 | 22.64 | 34.55 | 40.51 | 43.07 | 45.51 |

**TEST verdict.** F11 has the highest primary score (50.71%), versus F01 50.46%, F10 44.80% and F00 42.71%. F11−F01 is only **+0.25 points [−1.13, 1.92]**: these data do not establish an extra overall gain from the encoder once epoch-2 LoRA is used. The ICD gain is larger, while SNOMED all-types remains the bottleneck. The TEST results do not select or refit a model.

**Where the remaining errors occur.** F11 incoming R@64 is 91.41% for ICD and 82.10% for SNOMED; even perfect selection within those fixed pools would miss the 95% target. Among 592 final SNOMED errors, 215 have gold absent from retrieval, while 377 retain gold in the incoming pool (6 specificity errors and 371 other wrong selections). Conditional final selection accuracy is 61.76% when SNOMED gold is available. The decision layer rescues 143 SNOMED inputs and demotes 52 relative to raw reranking, so it helps overall but does not resolve all ranking errors. These are observations from the fixed evaluation, not TEST-driven tuning.

## Separate dictionary probes

These measure withheld alias/noise generalization, not chart TEST accuracy. Each system has 1,500 clean plus 1,500 noisy examples, grouped by code. Probe safeguards were used in prescribed component keep/drop rules, so this is not an independent untouched clinical holdout. No DiagnosisCode labels are invented for probes.

| Flow | System | n | Final accuracy [CI] | Clean accuracy | Noisy accuracy | Final R@10 | Final R@32 | Final R@64 | Incoming R@64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | ICD | 3000 | 15.07 [13.43, 16.87] | 15.93 | 14.20 | 52.03 | 74.30 | 84.53 | 84.23 |
| F00 | SNOMED | 3000 | 51.60 [49.00, 53.90] | 53.87 | 49.33 | 88.03 | 91.07 | 91.87 | 91.87 |
| F10 | ICD | 3000 | 22.13 [20.20, 24.23] | 22.93 | 21.33 | 71.67 | 83.97 | 85.80 | 85.70 |
| F10 | SNOMED | 3000 | 56.70 [54.30, 58.90] | 58.60 | 54.80 | 88.63 | 91.40 | 91.90 | 91.90 |
| F01 | ICD | 3000 | 60.97 [58.90, 63.24] | 63.13 | 58.80 | 82.87 | 84.87 | 85.20 | 84.23 |
| F01 | SNOMED | 3000 | 77.00 [75.07, 78.97] | 77.73 | 76.27 | 90.37 | 91.60 | 91.87 | 91.87 |
| F11 | ICD | 3000 | 67.03 [65.03, 69.27] | 68.53 | 65.53 | 84.30 | 85.73 | 86.03 | 85.70 |
| F11 | SNOMED | 3000 | 77.20 [75.23, 79.07] | 78.07 | 76.33 | 90.63 | 91.70 | 91.90 | 91.90 |

| Comparison | Probe ICD delta [CI] | Probe SNOMED delta [CI] | Probe primary delta [CI] |
| --- | --- | --- | --- |
| F01-F00 | +45.90 [43.57, 48.36] | +25.40 [23.41, 27.48] | +45.90 [43.57, 48.36] |
| F10-F00 | +7.07 [5.65, 8.53] | +5.10 [3.76, 6.58] | +7.07 [5.65, 8.53] |
| F11-F00 | +51.97 [49.76, 54.27] | +25.60 [23.59, 27.69] | +51.97 [49.76, 54.27] |
| F11-F01 | +6.07 [4.80, 7.43] | +0.20 [-0.62, 0.95] | +6.07 [4.80, 7.43] |
| F11-F10 | +44.90 [42.55, 47.24] | +20.50 [18.51, 22.45] | +44.90 [42.55, 47.24] |

The probe contains 136/1,500 ICD target headers outside train-derived permitted exceptions. They remain in the denominator; the corresponding maximum under this output policy is 90.93%. No target is removed to inflate accuracy. Full three-stage probe rankings and variant CIs are in the aggregate exports.

## Missing-code drill — latest epoch-2 F11

On all 1,368 DEV inputs, remove each supplied gold from retrieval, expansion and decision/coupling output, then apply the unchanged frozen full-DEV F11 decision model. This is a deliberate stress test, **not ordinary held-out accuracy**. A nearby ICD prediction means matching three-character category; a nearby SNOMED prediction means a trusted qualifier-sibling/lexical link. No threshold is fitted on this drill. Normal-reference flag rates use the same full-DEV model in-sample and must not be presented as out-of-fold estimates.

| System | n | Nearby prediction % [CI] | Low-confidence detection recall % [CI] | Nearby AND flagged % [CI] | Normal-reference flag % [CI] | Gold-present normal false-alarm % [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| ICD | 327 | 46.48 [40.66, 51.89] | 12.84 [9.48, 16.41] | 2.75 [1.20, 4.59] | 5.81 [3.35, 8.33] | 3.37 [1.37, 5.75] |
| SNOMED | 1041 | 2.88 [1.83, 4.04] | 7.78 [6.02, 9.51] | 0.19 [0.00, 0.49] | 4.80 [3.48, 6.21] | 2.58 [1.53, 3.85] |

Removed-gold emissions: **0**. The low-confidence flag catches only 12.84% of removed-gold ICD cases and 7.78% of SNOMED cases: **missing-code detection is weak under this stress test**. This separates detecting uncertainty from choosing a related replacement; neither is equivalent to exact-code correctness.

## Measured latency and footprint

Latest F11, RTX A6000/sm86, local vLLM epoch-2 LoRA. Fifty seeded unique DEV groups, after three public warm-ups; each request predicts **both systems**. Query embeddings, lexical/dense/memory score rows and reranker scores are recomputed. No query, score or token-input cache is used. Indexes and model weights are loaded; requests run sequentially, so this is warm per-input latency, not concurrent throughput or a cold-start benchmark. The trained type classifier is used without a gold subconcept.

| Component | Median ms | p95 ms |
| --- | --- | --- |
| Preprocessing, encoders and channel scoring | 847.58 | 893.02 |
| Retrieval fusion and expansion | 23.21 | 449.70 |
| Qwen reranker | 452.58 | 554.93 |
| Decision layer | 16.66 | 27.78 |
| End-to-end total | 1345.68 | 1757.22 |

Retriever/model/index initialization: 49.35 s; separate hosted reranker launch-to-healthy: 26.02 s (both excluded from warm latency). Observed allocated-device peak: **39.52 GiB**; sampled owned process-tree RSS peak: **14.47 GiB** across the diagnostic session, including drill and benchmark. Device memory includes the vLLM reservation, three SapBERT encoders and alias indexes; RSS may count shared pages more than once. Sampled peaks are observations, not minimum memory requirements. vLLM's normal prefix/KV reuse remains enabled; no client score, query-vector or token-input cache is used.

| Selected asset group | Disk GiB | Files |
| --- | --- | --- |
| Original SapBERT | 0.408 | 6 |
| Adapted ICD | 0.409 | 6 |
| Adapted SNOMED | 0.409 | 6 |
| Qwen base | 1.125 | 7 |
| Epoch-2 adapter | 0.091 | 12 |
| F11 decisions | 0.000 | 1 |
| Alias and mention-memory arrays | 1.825 | 8 |

Disk sizes count resolved files once per asset group. Alias/memory arrays exclude lexical indexes, terminology tables and temporary offline score/query caches; the table is an asset breakdown, not a complete packaged deployment size. Historical full-batch TEST scoring times are in the full stage report; they are not per-term latency. This benchmark measures F11 only and cannot establish speedups over other flows.

## Worked long-tail examples

Three successes and two failures sampled with seed 13 from rare/unseen **F11 out-of-fold DEV** cases, for explanation only. Codes and descriptions below come from the public dictionaries. Original mention/context and top candidates are preserved locally in `artifacts/runs/judging_metrics_20261018/worked_cases_private.json`; they are deliberately excluded from committed reports.

| Case | System / bucket | Gold code and public description | Final code and public description | Outcome | Gold in incoming pool? | Raw reranker correct? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ICD / rare | M19.011 — Primary osteoarthritis, right shoulder | M19.011 — Primary osteoarthritis, right shoulder | success | yes | yes |
| 2 | SNOMED / rare | 309538000 — Numbness of foot | 309538000 — Numbness of foot | success | yes | yes |
| 3 | ICD / rare | R53.81 — Nervous debility | R53.81 — Nervous debility | success | yes | no |
| 4 | SNOMED / unseen | 784580008 — Estimated creatinine clearance | 167181009 — Measurement of renal clearance of creatinine | failure (wrong selection) | yes | yes |
| 5 | SNOMED / rare | 443694000 — Uncontrolled type 2 diabetes mellitus | 268519009 — Diabetic - poor control | failure (wrong selection) | yes | no |

Case 3 illustrates a decision-layer rescue after a wrong raw ranking; case 4 illustrates a decision-layer demotion after a correct raw ranking. Both failures retained the exact gold in the incoming pool, so retrieval availability alone does not solve these examples. Related descriptions still count as incorrect when the supplied exact concept ID differs; these samples do not adjudicate the gold annotation.

Public synthetic abbreviation, typo and specificity demo inputs were also executed through all stages and validated; their actual outputs are saved in the benchmark’s `public_demo_cases.json`. They are illustrations, not a labelled accuracy set. No sample or TEST result was used to tune a prompt or model.

## Judging criteria and remaining deliverable limits

The [submission requirements](../docs/DELIVERABLES.md) assigns mapping accuracy 50%, long-tail performance 20%, engineering/reproducibility 15% and error analysis 15%. It does not provide a formula converting these measurements into a total judging grade, so no such grade is invented.

| Criterion / requested metric | Evidence / status |
| --- | --- |
| Exact-code mapping and primary minimum | All four DEV/TEST flows, per-system/subtype counts, grouped CIs and paired deltas; above and full exports |
| Long tail | Train-defined unseen/rare/common accuracy, R@64, conditional selection and hierarchy CIs; error location and five cases |
| ICD hierarchical / partial credit | Three-character category accuracy; over/under-specificity and illegal header counts |
| SNOMED hierarchy | Semantic tag/top-level and trusted lexical/qualifier links are proxies; **true SNOMED IS-A path distance unavailable** without RF2, which is outside the allowed resources |
| Coverage / abstention | 100% code output; review flags, accuracy at unflagged coverage, risk–coverage and calibration |
| Missing codes / unseen labels | Held-out alias probes, paired probe deltas and current-model gold-removal drill |
| Latency / footprint | Measured F11 warm median/p95 by stage and observed memory; selected disk assets, scope stated |
| Error analysis | Stage rescues/demotions, exclusive error-location counts, SNOMED classes, specificity; annotation/convention causes remain unadjudicated |
| Reproducibility / predictions | Saved configs, model/decision hashes, original split, deterministic seed, 12 passing validators; local prediction/diagnostic links in full report |
| Documentation PDF, 2–3-page write-up, demo/video, organizer hidden evaluation | This README is evidence for those deliverables, **not** their completed submission. Hidden-set results, final submission packaging/access and video remain separate |
| All-label final retrain / deployment artifacts | Not performed for this requested held-out comparison; evaluation checkpoints and train-only indexes were used |

The principal performance limitation is SNOMED all-types exact match, especially procedure/service distinctions and long-tail selection. The available candidate recall is much higher than top-1 accuracy: model ranking and final selection remain material error sources. The current results demonstrate improvement over the original-model pipeline; they do not meet the 95% target, establish true ontology distance, or certify production throughput.

## Provenance, validation and historical versions

Original-model DEV F00/F10 uses vLLM on Blackwell; revised-LoRA DEV F01/F11 uses vLLM on A6000. TEST F00/F10 uses Transformers/A6000 with matched scorer identities; TEST F01/F11 uses vLLM/Blackwell with matched epoch-2 identities. Cross-reranker deltas include possible bf16 backend/hardware effects; they are not a pure training-effect estimate. Architecture/model/template-specific caches are never interchanged. Both missing TEST arms were added after the initial F00/F11 report, using already frozen decisions without TEST-driven tuning.

All twelve DEV/PROBE/TEST validators pass. The final earlier publication audit checks all 664 aggregate records, 2,816 displayed values, all ten paired DEV/TEST comparisons, 40 preserved result hashes, group separation and frozen decisions. Additional metrics here reuse the saved supplied-label details; no raw sealed partition was reopened, no models were trained and no decision parameters were refitted.

- [Complete latest stage/category tables and local prediction links](Final_all_flows_DEV_TEST.md)
- [Aggregate CSV](Final_all_flows_metrics.csv) and [complete original metrics JSON](Final_all_flows_full_metrics.json)
- [Additional judging metrics JSON](Judging_metrics.json)

Historical snapshots remain in the research repository. This submission contains the authoritative final epoch-2 four-flow tables and aggregate exports.

Per-input predictions, diagnostics and validation files named above are distributed only in the restricted private bundle. See [results provenance and private-file access](../docs/RESULTS_PROVENANCE.md).
