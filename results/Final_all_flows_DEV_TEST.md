# Final TermMap results: all four flows on DEV and TEST

Latest preprocessing (`reviewed_20261018_v2`), fixed SapBERT P2-v2 selection and user-selected revised Qwen LoRA **epoch 2**. All four complete flows are evaluated on DEV and internal TEST. No model retraining or TEST fitting was performed. The 6,000 dictionary probes are reported separately. Historical model versions remain separate below.

| Flow | Query / memory encoder | Reranker |
|---|---|---|
| F00 | Original SapBERT | Original Qwen, no LoRA |
| F10 | Asymmetric P2-v2 SapBERT | Original Qwen, no LoRA |
| F01 | Original SapBERT | Revised Qwen LoRA epoch 2 |
| F11 | Asymmetric P2-v2 SapBERT | Revised Qwen LoRA epoch 2 |

Dictionary/alias vectors and type-inference vectors remain original SapBERT in every flow. Every flow has its own DEV-fitted fusion, specificity, coupling, confidence and missing-code layer, with prescribed component keep/drop rules.

## Sets, categories and metric definitions

| Set | n | Protocol |
|---|---:|---|
| Training | 6,751 | Training split; no training accuracy presented as held-out performance |
| DEV | 1,368 | Nested grouped cross-validation, separately fitted per flow |
| Internal TEST | 1,527 | Frozen full-DEV decisions; prepared once, cached pools reused |
| Dictionary PROBE | 6,000 | 1,500 codes/system with clean + noisy aliases; not chart DEV/TEST |

Accuracy = recall@1 = the percentage whose first predicted code exactly equals the supplied gold code. Recall@k is the percentage whose gold code appears among the first k ranked codes. **Primary score = min(ICD accuracy, SNOMED all-types accuracy)**. The DiagnosisCode minimum and paired-both-correct are supplementary. A minimum combines two system rates and has no single row denominator. Incoming R@64 is candidate availability, not expected final accuracy. Specificity may introduce an ICD descendant beyond the original pool.

Categories preserve supplied subconcepts. SNOMED all-types includes all supplied SNOMED labels, rather than only DiagnosisCode. NA with n=0 means there are no labelled examples to evaluate; no rows were relabelled or dropped. DEV contains 326 ICD DiagnosisCode rows plus the one supplied ICD ProcedureCode row; all 327 contribute to ICD accuracy. TEST has 326 ICD DiagnosisCode rows and no ICD ProcedureCode row. There are no MedicationProfileMedication labels in DEV or TEST. An n=1 category is a single observation, not a reliable estimate of generalization. Probe subtype labels are unknown; they are not treated as clinical gold categories.

Stage 1 = full retrieval; stage 2 = reranker before decisions; stage 3 = complete validated predictions. All displayed rates are percentages. Accuracy and R@64 intervals use saved grouped 95% bootstrap CIs where calculated; primary minimum and paired delta intervals use joint group resampling, 1,000 draws, seed 13. The CSV also preserves R@5/20/50/200 within the saved rankings. Incoming retrieval is capped at 64 codes; final specificity can add an ICD code beyond that pool (some final lists have 65). R@200 does not measure an independently retrieved 200-code pool. Probe labels support the prescribed component safeguards and do not constitute an independent chart test.

## Final complete-pipeline headlines

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

## Full chart tables by stage, system and category

### Stage 1

| Set | Flow | Stage | System | Category | n | Accuracy = R@1 (95% CI) | R@10 | R@32 | R@64 (95% CI) |
|---|---|---|---|---|---:|---|---:|---:|---|
| DEV | F00 | 1 | ICD | All | 327 | 39.45 [34.04, 44.92] | 67.28 | 79.20 | 85.93 [81.82, 89.81] |
| DEV | F00 | 1 | ICD | DiagnosisCode | 326 | 39.26 [33.63, 44.66] | 67.18 | 79.14 | 85.89 [82.02, 89.86] |
| DEV | F00 | 1 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F00 | 1 | SNOMED | All | 1041 | 36.79 [33.96, 39.52] | 69.16 | 79.15 | 83.48 [81.01, 85.82] |
| DEV | F00 | 1 | SNOMED | DiagnosisCode | 386 | 45.34 [40.54, 50.63] | 75.39 | 83.16 | 86.79 [83.55, 90.25] |
| DEV | F00 | 1 | SNOMED | ProcedureCode | 395 | 32.15 [27.66, 36.77] | 64.30 | 75.19 | 78.99 [74.63, 82.90] |
| DEV | F00 | 1 | SNOMED | LabService | 137 | 29.20 [21.51, 36.62] | 66.42 | 79.56 | 85.40 [78.62, 90.85] |
| DEV | F00 | 1 | SNOMED | EncounterType | 56 | 37.50 [24.00, 51.67] | 73.21 | 83.93 | 91.07 [83.33, 98.08] |
| DEV | F00 | 1 | SNOMED | Device | 67 | 29.85 [19.35, 40.91] | 64.18 | 74.63 | 80.60 [70.59, 89.07] |
| DEV | F00 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F00 | 1 | Combined | Primary minimum | — | 36.79 [33.58, 39.39] | 67.28 | 79.15 | 83.48 |
| DEV | F10 | 1 | ICD | All | 327 | 43.43 [37.54, 49.07] | 77.06 | 87.77 | 90.83 [87.65, 93.92] |
| DEV | F10 | 1 | ICD | DiagnosisCode | 326 | 43.25 [37.95, 48.91] | 76.99 | 87.73 | 90.80 [87.50, 94.03] |
| DEV | F10 | 1 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F10 | 1 | SNOMED | All | 1041 | 37.08 [34.14, 39.91] | 72.62 | 82.80 | 85.78 [83.61, 88.07] |
| DEV | F10 | 1 | SNOMED | DiagnosisCode | 386 | 44.04 [39.28, 49.23] | 79.02 | 85.75 | 89.12 [86.05, 92.24] |
| DEV | F10 | 1 | SNOMED | ProcedureCode | 395 | 32.66 [28.06, 36.96] | 67.59 | 77.72 | 80.76 [76.74, 84.51] |
| DEV | F10 | 1 | SNOMED | LabService | 137 | 32.85 [25.31, 40.74] | 72.26 | 84.67 | 86.86 [80.71, 92.25] |
| DEV | F10 | 1 | SNOMED | EncounterType | 56 | 39.29 [25.93, 51.79] | 75.00 | 91.07 | 92.86 [85.96, 98.18] |
| DEV | F10 | 1 | SNOMED | Device | 67 | 29.85 [19.44, 40.33] | 64.18 | 85.07 | 88.06 [79.69, 94.29] |
| DEV | F10 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F10 | 1 | Combined | Primary minimum | — | 37.08 [34.14, 40.06] | 72.62 | 82.80 | 85.78 |
| DEV | F01 | 1 | ICD | All | 327 | 39.45 [34.04, 44.92] | 67.28 | 79.20 | 85.93 [81.82, 89.81] |
| DEV | F01 | 1 | ICD | DiagnosisCode | 326 | 39.26 [33.63, 44.66] | 67.18 | 79.14 | 85.89 [82.02, 89.86] |
| DEV | F01 | 1 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F01 | 1 | SNOMED | All | 1041 | 36.79 [33.96, 39.52] | 69.16 | 79.15 | 83.48 [81.01, 85.82] |
| DEV | F01 | 1 | SNOMED | DiagnosisCode | 386 | 45.34 [40.54, 50.63] | 75.39 | 83.16 | 86.79 [83.55, 90.25] |
| DEV | F01 | 1 | SNOMED | ProcedureCode | 395 | 32.15 [27.66, 36.77] | 64.30 | 75.19 | 78.99 [74.63, 82.90] |
| DEV | F01 | 1 | SNOMED | LabService | 137 | 29.20 [21.51, 36.62] | 66.42 | 79.56 | 85.40 [78.62, 90.85] |
| DEV | F01 | 1 | SNOMED | EncounterType | 56 | 37.50 [24.00, 51.67] | 73.21 | 83.93 | 91.07 [83.33, 98.08] |
| DEV | F01 | 1 | SNOMED | Device | 67 | 29.85 [19.35, 40.91] | 64.18 | 74.63 | 80.60 [70.59, 89.07] |
| DEV | F01 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F01 | 1 | Combined | Primary minimum | — | 36.79 [33.58, 39.39] | 67.28 | 79.15 | 83.48 |
| DEV | F11 | 1 | ICD | All | 327 | 43.43 [37.54, 49.07] | 77.06 | 87.77 | 90.83 [87.65, 93.92] |
| DEV | F11 | 1 | ICD | DiagnosisCode | 326 | 43.25 [37.95, 48.91] | 76.99 | 87.73 | 90.80 [87.50, 94.03] |
| DEV | F11 | 1 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F11 | 1 | SNOMED | All | 1041 | 37.08 [34.14, 39.91] | 72.62 | 82.80 | 85.78 [83.61, 88.07] |
| DEV | F11 | 1 | SNOMED | DiagnosisCode | 386 | 44.04 [39.28, 49.23] | 79.02 | 85.75 | 89.12 [86.05, 92.24] |
| DEV | F11 | 1 | SNOMED | ProcedureCode | 395 | 32.66 [28.06, 36.96] | 67.59 | 77.72 | 80.76 [76.74, 84.51] |
| DEV | F11 | 1 | SNOMED | LabService | 137 | 32.85 [25.31, 40.74] | 72.26 | 84.67 | 86.86 [80.71, 92.25] |
| DEV | F11 | 1 | SNOMED | EncounterType | 56 | 39.29 [25.93, 51.79] | 75.00 | 91.07 | 92.86 [85.96, 98.18] |
| DEV | F11 | 1 | SNOMED | Device | 67 | 29.85 [19.44, 40.33] | 64.18 | 85.07 | 88.06 [79.69, 94.29] |
| DEV | F11 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F11 | 1 | Combined | Primary minimum | — | 37.08 [34.14, 40.06] | 72.62 | 82.80 | 85.78 |
| TEST | F00 | 1 | ICD | All | 326 | 40.80 [35.58, 45.54] | 69.94 | 81.29 | 86.20 [82.27, 90.52] |
| TEST | F00 | 1 | ICD | DiagnosisCode | 326 | 40.80 [35.58, 45.54] | 69.94 | 81.29 | 86.20 [82.27, 90.52] |
| TEST | F00 | 1 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F00 | 1 | SNOMED | All | 1201 | 36.72 [33.72, 39.46] | 67.11 | 75.35 | 80.02 [77.75, 82.19] |
| TEST | F00 | 1 | SNOMED | DiagnosisCode | 390 | 44.62 [39.39, 49.58] | 76.41 | 83.85 | 86.67 [83.12, 89.93] |
| TEST | F00 | 1 | SNOMED | ProcedureCode | 492 | 28.86 [24.39, 32.85] | 60.16 | 69.11 | 74.19 [70.27, 77.62] |
| TEST | F00 | 1 | SNOMED | LabService | 144 | 34.03 [26.62, 41.18] | 61.11 | 70.14 | 76.39 [70.00, 83.02] |
| TEST | F00 | 1 | SNOMED | EncounterType | 97 | 41.24 [31.93, 51.49] | 70.10 | 77.32 | 82.47 [76.09, 89.11] |
| TEST | F00 | 1 | SNOMED | Device | 78 | 46.15 [36.17, 56.52] | 71.79 | 79.49 | 87.18 [78.57, 94.29] |
| TEST | F00 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F00 | 1 | Combined | Primary minimum | — | 36.72 [33.57, 39.50] | 67.11 | 75.35 | 80.02 |
| TEST | F11 | 1 | ICD | All | 326 | 44.48 [39.31, 50.00] | 76.07 | 87.73 | 91.41 [88.14, 94.45] |
| TEST | F11 | 1 | ICD | DiagnosisCode | 326 | 44.48 [39.31, 50.00] | 76.07 | 87.73 | 91.41 [88.14, 94.45] |
| TEST | F11 | 1 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F11 | 1 | SNOMED | All | 1201 | 36.39 [33.39, 39.07] | 70.27 | 78.52 | 82.10 [79.79, 84.17] |
| TEST | F11 | 1 | SNOMED | DiagnosisCode | 390 | 44.10 [38.67, 49.44] | 80.00 | 85.13 | 87.95 [84.30, 91.08] |
| TEST | F11 | 1 | SNOMED | ProcedureCode | 492 | 29.47 [24.77, 33.44] | 62.80 | 73.98 | 77.03 [73.07, 80.35] |
| TEST | F11 | 1 | SNOMED | LabService | 144 | 33.33 [25.69, 40.32] | 65.28 | 73.61 | 79.86 [73.55, 85.72] |
| TEST | F11 | 1 | SNOMED | EncounterType | 97 | 39.18 [30.93, 48.89] | 71.13 | 79.38 | 82.47 [76.29, 89.33] |
| TEST | F11 | 1 | SNOMED | Device | 78 | 43.59 [33.32, 55.88] | 76.92 | 82.05 | 88.46 [80.60, 95.00] |
| TEST | F11 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F11 | 1 | Combined | Primary minimum | — | 36.39 [33.42, 39.17] | 70.27 | 78.52 | 82.10 |
| TEST | F10 | 1 | ICD | All | 326 | 44.48 [39.31, 50.00] | 76.07 | 87.73 | 91.41 [88.14, 94.45] |
| TEST | F10 | 1 | ICD | DiagnosisCode | 326 | 44.48 [39.31, 50.00] | 76.07 | 87.73 | 91.41 [88.14, 94.45] |
| TEST | F10 | 1 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F10 | 1 | SNOMED | All | 1201 | 36.39 [33.39, 39.07] | 70.27 | 78.52 | 82.10 [79.79, 84.17] |
| TEST | F10 | 1 | SNOMED | DiagnosisCode | 390 | 44.10 [38.67, 49.44] | 80.00 | 85.13 | 87.95 [84.30, 91.08] |
| TEST | F10 | 1 | SNOMED | ProcedureCode | 492 | 29.47 [24.77, 33.44] | 62.80 | 73.98 | 77.03 [73.07, 80.35] |
| TEST | F10 | 1 | SNOMED | LabService | 144 | 33.33 [25.69, 40.32] | 65.28 | 73.61 | 79.86 [73.55, 85.72] |
| TEST | F10 | 1 | SNOMED | EncounterType | 97 | 39.18 [30.93, 48.89] | 71.13 | 79.38 | 82.47 [76.29, 89.33] |
| TEST | F10 | 1 | SNOMED | Device | 78 | 43.59 [33.32, 55.88] | 76.92 | 82.05 | 88.46 [80.60, 95.00] |
| TEST | F10 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F10 | 1 | Combined | Primary minimum | — | 36.39 [33.42, 39.17] | 70.27 | 78.52 | 82.10 |
| TEST | F01 | 1 | ICD | All | 326 | 40.80 [35.58, 45.54] | 69.94 | 81.29 | 86.20 [82.27, 90.52] |
| TEST | F01 | 1 | ICD | DiagnosisCode | 326 | 40.80 [35.58, 45.54] | 69.94 | 81.29 | 86.20 [82.27, 90.52] |
| TEST | F01 | 1 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F01 | 1 | SNOMED | All | 1201 | 36.72 [33.72, 39.46] | 67.11 | 75.35 | 80.02 [77.75, 82.19] |
| TEST | F01 | 1 | SNOMED | DiagnosisCode | 390 | 44.62 [39.39, 49.58] | 76.41 | 83.85 | 86.67 [83.12, 89.93] |
| TEST | F01 | 1 | SNOMED | ProcedureCode | 492 | 28.86 [24.39, 32.85] | 60.16 | 69.11 | 74.19 [70.27, 77.62] |
| TEST | F01 | 1 | SNOMED | LabService | 144 | 34.03 [26.62, 41.18] | 61.11 | 70.14 | 76.39 [70.00, 83.02] |
| TEST | F01 | 1 | SNOMED | EncounterType | 97 | 41.24 [31.93, 51.49] | 70.10 | 77.32 | 82.47 [76.09, 89.11] |
| TEST | F01 | 1 | SNOMED | Device | 78 | 46.15 [36.17, 56.52] | 71.79 | 79.49 | 87.18 [78.57, 94.29] |
| TEST | F01 | 1 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F01 | 1 | Combined | Primary minimum | — | 36.72 [33.57, 39.50] | 67.11 | 75.35 | 80.02 |

### Stage 2

| Set | Flow | Stage | System | Category | n | Accuracy = R@1 (95% CI) | R@10 | R@32 | R@64 (95% CI) |
|---|---|---|---|---|---:|---|---:|---:|---|
| DEV | F00 | 2 | ICD | All | 327 | 32.42 [27.27, 37.54] | 69.72 | 81.65 | 85.93 [81.82, 89.81] |
| DEV | F00 | 2 | ICD | DiagnosisCode | 326 | 32.21 [26.71, 37.05] | 69.63 | 81.60 | 85.89 [82.02, 89.86] |
| DEV | F00 | 2 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F00 | 2 | SNOMED | All | 1041 | 30.07 [26.99, 32.94] | 69.16 | 81.17 | 83.48 [81.01, 85.82] |
| DEV | F00 | 2 | SNOMED | DiagnosisCode | 386 | 35.23 [30.39, 40.05] | 76.68 | 86.01 | 86.79 [83.55, 90.25] |
| DEV | F00 | 2 | SNOMED | ProcedureCode | 395 | 23.54 [18.68, 28.02] | 60.51 | 74.94 | 78.99 [74.63, 82.90] |
| DEV | F00 | 2 | SNOMED | LabService | 137 | 31.39 [23.45, 39.86] | 67.88 | 82.48 | 85.40 [78.62, 90.85] |
| DEV | F00 | 2 | SNOMED | EncounterType | 56 | 25.00 [11.10, 38.71] | 78.57 | 89.29 | 91.07 [83.33, 98.08] |
| DEV | F00 | 2 | SNOMED | Device | 67 | 40.30 [28.57, 51.63] | 71.64 | 80.60 | 80.60 [70.59, 89.07] |
| DEV | F00 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F00 | 2 | Combined | Primary minimum | — | 30.07 [26.79, 32.85] | 69.16 | 81.17 | 83.48 |
| DEV | F10 | 2 | ICD | All | 327 | 31.19 [26.27, 36.44] | 71.87 | 85.93 | 90.83 [87.65, 93.92] |
| DEV | F10 | 2 | ICD | DiagnosisCode | 326 | 30.98 [25.23, 35.94] | 71.78 | 85.89 | 90.80 [87.50, 94.03] |
| DEV | F10 | 2 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F10 | 2 | SNOMED | All | 1041 | 30.26 [27.16, 33.17] | 69.07 | 82.71 | 85.78 [83.61, 88.07] |
| DEV | F10 | 2 | SNOMED | DiagnosisCode | 386 | 36.01 [31.05, 40.92] | 77.72 | 88.34 | 89.12 [86.05, 92.24] |
| DEV | F10 | 2 | SNOMED | ProcedureCode | 395 | 23.54 [18.67, 28.01] | 60.25 | 75.70 | 80.76 [76.74, 84.51] |
| DEV | F10 | 2 | SNOMED | LabService | 137 | 30.66 [22.75, 39.13] | 64.96 | 82.48 | 86.86 [80.71, 92.25] |
| DEV | F10 | 2 | SNOMED | EncounterType | 56 | 25.00 [11.10, 38.71] | 76.79 | 87.50 | 92.86 [85.96, 98.18] |
| DEV | F10 | 2 | SNOMED | Device | 67 | 40.30 [28.57, 51.63] | 73.13 | 88.06 | 88.06 [79.69, 94.29] |
| DEV | F10 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F10 | 2 | Combined | Primary minimum | — | 30.26 [26.21, 32.55] | 69.07 | 82.71 | 85.78 |
| DEV | F01 | 2 | ICD | All | 327 | 55.66 [50.31, 61.15] | 84.71 | 85.93 | 85.93 [81.82, 89.81] |
| DEV | F01 | 2 | ICD | DiagnosisCode | 326 | 55.83 [50.60, 61.40] | 84.66 | 85.89 | 85.89 [82.02, 89.86] |
| DEV | F01 | 2 | ICD | ProcedureCode | 1 | 0.00 [0.00, 0.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F01 | 2 | SNOMED | All | 1041 | 45.44 [42.31, 48.50] | 76.95 | 83.00 | 83.48 [81.01, 85.82] |
| DEV | F01 | 2 | SNOMED | DiagnosisCode | 386 | 53.63 [48.24, 59.01] | 82.64 | 86.79 | 86.79 [83.55, 90.25] |
| DEV | F01 | 2 | SNOMED | ProcedureCode | 395 | 37.47 [32.45, 42.33] | 71.39 | 78.48 | 78.99 [74.63, 82.90] |
| DEV | F01 | 2 | SNOMED | LabService | 137 | 48.91 [39.41, 58.46] | 79.56 | 84.67 | 85.40 [78.62, 90.85] |
| DEV | F01 | 2 | SNOMED | EncounterType | 56 | 41.07 [26.78, 55.56] | 80.36 | 91.07 | 91.07 [83.33, 98.08] |
| DEV | F01 | 2 | SNOMED | Device | 67 | 41.79 [29.41, 54.79] | 68.66 | 77.61 | 80.60 [70.59, 89.07] |
| DEV | F01 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F01 | 2 | Combined | Primary minimum | — | 45.44 [42.39, 48.65] | 76.95 | 83.00 | 83.48 |
| DEV | F11 | 2 | ICD | All | 327 | 54.43 [49.19, 60.06] | 88.38 | 90.83 | 90.83 [87.65, 93.92] |
| DEV | F11 | 2 | ICD | DiagnosisCode | 326 | 54.60 [49.25, 60.07] | 88.34 | 90.80 | 90.80 [87.50, 94.03] |
| DEV | F11 | 2 | ICD | ProcedureCode | 1 | 0.00 [0.00, 0.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F11 | 2 | SNOMED | All | 1041 | 44.48 [41.10, 47.68] | 77.62 | 85.21 | 85.78 [83.61, 88.07] |
| DEV | F11 | 2 | SNOMED | DiagnosisCode | 386 | 53.37 [48.15, 58.69] | 84.20 | 89.12 | 89.12 [86.05, 92.24] |
| DEV | F11 | 2 | SNOMED | ProcedureCode | 395 | 36.20 [30.79, 41.48] | 71.39 | 80.00 | 80.76 [76.74, 84.51] |
| DEV | F11 | 2 | SNOMED | LabService | 137 | 45.99 [36.50, 55.79] | 78.10 | 85.40 | 86.86 [80.71, 92.25] |
| DEV | F11 | 2 | SNOMED | EncounterType | 56 | 39.29 [24.99, 53.85] | 80.36 | 92.86 | 92.86 [85.96, 98.18] |
| DEV | F11 | 2 | SNOMED | Device | 67 | 43.28 [29.69, 56.95] | 73.13 | 86.57 | 88.06 [79.69, 94.29] |
| DEV | F11 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F11 | 2 | Combined | Primary minimum | — | 44.48 [41.20, 47.61] | 77.62 | 85.21 | 85.78 |
| TEST | F00 | 2 | ICD | All | 326 | 32.21 [26.86, 37.17] | 71.17 | 85.28 | 86.20 [82.27, 90.52] |
| TEST | F00 | 2 | ICD | DiagnosisCode | 326 | 32.21 [26.86, 37.17] | 71.17 | 85.28 | 86.20 [82.27, 90.52] |
| TEST | F00 | 2 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F00 | 2 | SNOMED | All | 1201 | 29.14 [26.55, 31.61] | 65.36 | 77.52 | 80.02 [77.75, 82.19] |
| TEST | F00 | 2 | SNOMED | DiagnosisCode | 390 | 41.03 [36.41, 45.90] | 76.15 | 84.62 | 86.67 [83.12, 89.93] |
| TEST | F00 | 2 | SNOMED | ProcedureCode | 492 | 20.12 [16.54, 24.49] | 54.07 | 70.93 | 74.19 [70.27, 77.62] |
| TEST | F00 | 2 | SNOMED | LabService | 144 | 30.56 [22.40, 37.59] | 63.89 | 75.00 | 76.39 [70.00, 83.02] |
| TEST | F00 | 2 | SNOMED | EncounterType | 97 | 27.84 [18.98, 37.32] | 71.13 | 79.38 | 82.47 [76.09, 89.11] |
| TEST | F00 | 2 | SNOMED | Device | 78 | 25.64 [17.10, 34.49] | 78.21 | 85.90 | 87.18 [78.57, 94.29] |
| TEST | F00 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F00 | 2 | Combined | Primary minimum | — | 29.14 [26.54, 31.64] | 65.36 | 77.52 | 80.02 |
| TEST | F11 | 2 | ICD | All | 326 | 53.37 [47.80, 58.41] | 87.42 | 90.80 | 91.41 [88.14, 94.45] |
| TEST | F11 | 2 | ICD | DiagnosisCode | 326 | 53.37 [47.80, 58.41] | 87.42 | 90.80 | 91.41 [88.14, 94.45] |
| TEST | F11 | 2 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F11 | 2 | SNOMED | All | 1201 | 43.13 [40.37, 45.72] | 74.27 | 81.35 | 82.10 [79.79, 84.17] |
| TEST | F11 | 2 | SNOMED | DiagnosisCode | 390 | 56.92 [51.38, 62.15] | 81.79 | 87.44 | 87.95 [84.30, 91.08] |
| TEST | F11 | 2 | SNOMED | ProcedureCode | 492 | 34.55 [31.12, 38.61] | 67.89 | 76.42 | 77.03 [73.07, 80.35] |
| TEST | F11 | 2 | SNOMED | LabService | 144 | 36.11 [27.69, 43.75] | 70.83 | 78.47 | 79.86 [73.55, 85.72] |
| TEST | F11 | 2 | SNOMED | EncounterType | 97 | 38.14 [28.72, 47.83] | 77.32 | 82.47 | 82.47 [76.29, 89.33] |
| TEST | F11 | 2 | SNOMED | Device | 78 | 47.44 [36.36, 58.21] | 79.49 | 85.90 | 88.46 [80.60, 95.00] |
| TEST | F11 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F11 | 2 | Combined | Primary minimum | — | 43.13 [40.35, 45.88] | 74.27 | 81.35 | 82.10 |
| TEST | F10 | 2 | ICD | All | 326 | 31.29 [25.96, 36.36] | 74.85 | 88.34 | 91.41 [88.14, 94.45] |
| TEST | F10 | 2 | ICD | DiagnosisCode | 326 | 31.29 [25.96, 36.36] | 74.85 | 88.34 | 91.41 [88.14, 94.45] |
| TEST | F10 | 2 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F10 | 2 | SNOMED | All | 1201 | 29.23 [26.77, 31.57] | 66.61 | 78.93 | 82.10 [79.79, 84.17] |
| TEST | F10 | 2 | SNOMED | DiagnosisCode | 390 | 40.51 [35.66, 45.38] | 77.44 | 85.38 | 87.95 [84.30, 91.08] |
| TEST | F10 | 2 | SNOMED | ProcedureCode | 492 | 20.53 [16.88, 25.06] | 55.28 | 72.76 | 77.03 [73.07, 80.35] |
| TEST | F10 | 2 | SNOMED | LabService | 144 | 29.17 [21.62, 35.64] | 67.36 | 77.78 | 79.86 [73.55, 85.72] |
| TEST | F10 | 2 | SNOMED | EncounterType | 97 | 29.90 [21.05, 39.03] | 70.10 | 79.38 | 82.47 [76.29, 89.33] |
| TEST | F10 | 2 | SNOMED | Device | 78 | 26.92 [17.65, 35.72] | 78.21 | 87.18 | 88.46 [80.60, 95.00] |
| TEST | F10 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F10 | 2 | Combined | Primary minimum | — | 29.23 [25.96, 31.57] | 66.61 | 78.93 | 82.10 |
| TEST | F01 | 2 | ICD | All | 326 | 58.28 [53.07, 63.06] | 83.13 | 86.20 | 86.20 [82.27, 90.52] |
| TEST | F01 | 2 | ICD | DiagnosisCode | 326 | 58.28 [53.07, 63.06] | 83.13 | 86.20 | 86.20 [82.27, 90.52] |
| TEST | F01 | 2 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F01 | 2 | SNOMED | All | 1201 | 44.05 [41.22, 46.83] | 73.86 | 79.35 | 80.02 [77.75, 82.19] |
| TEST | F01 | 2 | SNOMED | DiagnosisCode | 390 | 57.95 [52.42, 63.19] | 81.54 | 86.67 | 86.67 [83.12, 89.93] |
| TEST | F01 | 2 | SNOMED | ProcedureCode | 492 | 34.76 [30.98, 39.31] | 67.89 | 73.37 | 74.19 [70.27, 77.62] |
| TEST | F01 | 2 | SNOMED | LabService | 144 | 38.19 [29.91, 46.43] | 69.44 | 75.00 | 76.39 [70.00, 83.02] |
| TEST | F01 | 2 | SNOMED | EncounterType | 97 | 40.21 [30.53, 50.00] | 77.32 | 82.47 | 82.47 [76.09, 89.11] |
| TEST | F01 | 2 | SNOMED | Device | 78 | 48.72 [37.33, 59.09] | 76.92 | 84.62 | 87.18 [78.57, 94.29] |
| TEST | F01 | 2 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F01 | 2 | Combined | Primary minimum | — | 44.05 [41.47, 46.93] | 73.86 | 79.35 | 80.02 |

### Stage 3

| Set | Flow | Stage | System | Category | n | Accuracy = R@1 (95% CI) | R@10 | R@32 | R@64 (95% CI) |
|---|---|---|---|---|---:|---|---:|---:|---|
| DEV | F00 | 3 | ICD | All | 327 | 52.91 [47.54, 58.36] | 81.35 | 85.63 | 85.93 [81.82, 89.81] |
| DEV | F00 | 3 | ICD | DiagnosisCode | 326 | 52.76 [47.56, 58.21] | 81.29 | 85.58 | 85.89 [82.02, 89.86] |
| DEV | F00 | 3 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F00 | 3 | SNOMED | All | 1041 | 43.13 [40.04, 46.11] | 76.95 | 82.61 | 83.48 [81.01, 85.82] |
| DEV | F00 | 3 | SNOMED | DiagnosisCode | 386 | 51.81 [46.84, 56.78] | 83.16 | 86.79 | 86.79 [83.55, 90.25] |
| DEV | F00 | 3 | SNOMED | ProcedureCode | 395 | 38.23 [32.99, 43.27] | 69.87 | 77.47 | 78.99 [74.63, 82.90] |
| DEV | F00 | 3 | SNOMED | LabService | 137 | 36.50 [28.46, 44.88] | 78.83 | 83.94 | 85.40 [78.62, 90.85] |
| DEV | F00 | 3 | SNOMED | EncounterType | 56 | 41.07 [27.45, 54.24] | 82.14 | 89.29 | 91.07 [83.33, 98.08] |
| DEV | F00 | 3 | SNOMED | Device | 67 | 37.31 [25.00, 47.89] | 74.63 | 80.60 | 80.60 [70.59, 89.07] |
| DEV | F00 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F00 | 3 | Combined | Primary minimum | — | 43.13 [40.18, 46.19] | 76.95 | 82.61 | 83.48 |
| DEV | F10 | 3 | ICD | All | 327 | 56.88 [51.46, 62.19] | 86.85 | 90.83 | 90.83 [87.65, 93.92] |
| DEV | F10 | 3 | ICD | DiagnosisCode | 326 | 56.75 [51.58, 62.22] | 86.81 | 90.80 | 90.80 [87.50, 94.03] |
| DEV | F10 | 3 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F10 | 3 | SNOMED | All | 1041 | 45.24 [42.08, 48.42] | 80.02 | 84.92 | 85.78 [83.61, 88.07] |
| DEV | F10 | 3 | SNOMED | DiagnosisCode | 386 | 52.59 [47.53, 57.84] | 86.79 | 88.60 | 89.12 [86.05, 92.24] |
| DEV | F10 | 3 | SNOMED | ProcedureCode | 395 | 40.76 [35.90, 45.35] | 71.90 | 79.75 | 80.76 [76.74, 84.51] |
| DEV | F10 | 3 | SNOMED | LabService | 137 | 41.61 [33.33, 51.08] | 82.48 | 86.13 | 86.86 [80.71, 92.25] |
| DEV | F10 | 3 | SNOMED | EncounterType | 56 | 48.21 [35.85, 61.82] | 85.71 | 89.29 | 92.86 [85.96, 98.18] |
| DEV | F10 | 3 | SNOMED | Device | 67 | 34.33 [22.73, 45.71] | 79.10 | 88.06 | 88.06 [79.69, 94.29] |
| DEV | F10 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F10 | 3 | Combined | Primary minimum | — | 45.24 [41.96, 48.26] | 80.02 | 84.92 | 85.78 |
| DEV | F01 | 3 | ICD | All | 327 | 60.86 [55.66, 65.85] | 84.40 | 86.24 | 86.24 [82.12, 90.03] |
| DEV | F01 | 3 | ICD | DiagnosisCode | 326 | 60.74 [55.59, 66.35] | 84.36 | 86.20 | 86.20 [82.16, 90.12] |
| DEV | F01 | 3 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F01 | 3 | SNOMED | All | 1041 | 50.53 [47.57, 53.54] | 79.15 | 83.29 | 83.48 [81.01, 85.82] |
| DEV | F01 | 3 | SNOMED | DiagnosisCode | 386 | 57.77 [52.44, 63.22] | 84.72 | 86.79 | 86.79 [83.55, 90.25] |
| DEV | F01 | 3 | SNOMED | ProcedureCode | 395 | 46.08 [41.06, 51.02] | 73.92 | 78.73 | 78.99 [74.63, 82.90] |
| DEV | F01 | 3 | SNOMED | LabService | 137 | 48.91 [40.00, 56.85] | 81.02 | 85.40 | 85.40 [78.62, 90.85] |
| DEV | F01 | 3 | SNOMED | EncounterType | 56 | 51.79 [38.46, 66.13] | 80.36 | 91.07 | 91.07 [83.33, 98.08] |
| DEV | F01 | 3 | SNOMED | Device | 67 | 37.31 [26.15, 47.76] | 73.13 | 79.10 | 80.60 [70.59, 89.07] |
| DEV | F01 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F01 | 3 | Combined | Primary minimum | — | 50.53 [47.60, 53.70] | 79.15 | 83.29 | 83.48 |
| DEV | F11 | 3 | ICD | All | 327 | 66.97 [61.81, 71.81] | 89.30 | 90.83 | 90.83 [87.65, 93.92] |
| DEV | F11 | 3 | ICD | DiagnosisCode | 326 | 66.87 [61.79, 71.85] | 89.26 | 90.80 | 90.80 [87.50, 94.03] |
| DEV | F11 | 3 | ICD | ProcedureCode | 1 | 100.00 [100.00, 100.00] | 100.00 | 100.00 | 100.00 [100.00, 100.00] |
| DEV | F11 | 3 | SNOMED | All | 1041 | 51.30 [48.23, 54.50] | 81.17 | 85.49 | 85.78 [83.61, 88.07] |
| DEV | F11 | 3 | SNOMED | DiagnosisCode | 386 | 57.25 [52.03, 62.30] | 86.79 | 88.86 | 89.12 [86.05, 92.24] |
| DEV | F11 | 3 | SNOMED | ProcedureCode | 395 | 47.09 [42.06, 51.98] | 74.94 | 80.51 | 80.76 [76.74, 84.51] |
| DEV | F11 | 3 | SNOMED | LabService | 137 | 49.64 [39.53, 59.40] | 83.21 | 86.13 | 86.86 [80.71, 92.25] |
| DEV | F11 | 3 | SNOMED | EncounterType | 56 | 53.57 [39.62, 67.24] | 85.71 | 92.86 | 92.86 [85.96, 98.18] |
| DEV | F11 | 3 | SNOMED | Device | 67 | 43.28 [31.25, 53.53] | 77.61 | 88.06 | 88.06 [79.69, 94.29] |
| DEV | F11 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| DEV | F11 | 3 | Combined | Primary minimum | — | 51.30 [48.12, 54.47] | 81.17 | 85.49 | 85.78 |
| TEST | F00 | 3 | ICD | All | 326 | 53.07 [47.38, 58.03] | 84.05 | 86.50 | 86.50 [82.81, 90.52] |
| TEST | F00 | 3 | ICD | DiagnosisCode | 326 | 53.07 [47.38, 58.03] | 84.05 | 86.50 | 86.50 [82.81, 90.52] |
| TEST | F00 | 3 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F00 | 3 | SNOMED | All | 1201 | 42.71 [39.83, 45.39] | 74.52 | 79.35 | 80.02 [77.75, 82.19] |
| TEST | F00 | 3 | SNOMED | DiagnosisCode | 390 | 51.54 [46.81, 56.18] | 83.33 | 86.41 | 86.67 [83.12, 89.93] |
| TEST | F00 | 3 | SNOMED | ProcedureCode | 492 | 34.96 [30.35, 39.12] | 67.48 | 72.97 | 74.19 [70.27, 77.62] |
| TEST | F00 | 3 | SNOMED | LabService | 144 | 37.50 [29.41, 44.80] | 71.53 | 76.39 | 76.39 [70.00, 83.02] |
| TEST | F00 | 3 | SNOMED | EncounterType | 97 | 45.36 [36.78, 54.76] | 76.29 | 82.47 | 82.47 [76.09, 89.11] |
| TEST | F00 | 3 | SNOMED | Device | 78 | 53.85 [42.86, 65.76] | 78.21 | 85.90 | 87.18 [78.57, 94.29] |
| TEST | F00 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F00 | 3 | Combined | Primary minimum | — | 42.71 [39.91, 45.73] | 74.52 | 79.35 | 80.02 |
| TEST | F11 | 3 | ICD | All | 326 | 68.40 [63.38, 73.45] | 89.88 | 91.41 | 91.41 [88.14, 94.45] |
| TEST | F11 | 3 | ICD | DiagnosisCode | 326 | 68.40 [63.38, 73.45] | 89.88 | 91.41 | 91.41 [88.14, 94.45] |
| TEST | F11 | 3 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F11 | 3 | SNOMED | All | 1201 | 50.71 [47.87, 53.36] | 78.27 | 81.85 | 82.10 [79.79, 84.17] |
| TEST | F11 | 3 | SNOMED | DiagnosisCode | 390 | 62.05 [56.92, 67.19] | 85.38 | 87.95 | 87.95 [84.30, 91.08] |
| TEST | F11 | 3 | SNOMED | ProcedureCode | 492 | 42.89 [38.26, 47.03] | 72.36 | 76.83 | 77.03 [73.07, 80.35] |
| TEST | F11 | 3 | SNOMED | LabService | 144 | 44.44 [37.59, 54.46] | 77.08 | 79.86 | 79.86 [73.55, 85.72] |
| TEST | F11 | 3 | SNOMED | EncounterType | 97 | 48.45 [39.18, 58.62] | 79.38 | 82.47 | 82.47 [76.29, 89.33] |
| TEST | F11 | 3 | SNOMED | Device | 78 | 57.69 [47.22, 70.00] | 80.77 | 85.90 | 88.46 [80.60, 95.00] |
| TEST | F11 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F11 | 3 | Combined | Primary minimum | — | 50.71 [48.12, 53.57] | 78.27 | 81.85 | 82.10 |
| TEST | F10 | 3 | ICD | All | 326 | 62.27 [56.88, 67.06] | 86.50 | 91.41 | 91.41 [88.14, 94.45] |
| TEST | F10 | 3 | ICD | DiagnosisCode | 326 | 62.27 [56.88, 67.06] | 86.50 | 91.41 | 91.41 [88.14, 94.45] |
| TEST | F10 | 3 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F10 | 3 | SNOMED | All | 1201 | 44.80 [41.69, 47.49] | 77.35 | 81.43 | 82.10 [79.79, 84.17] |
| TEST | F10 | 3 | SNOMED | DiagnosisCode | 390 | 55.13 [49.74, 59.85] | 85.13 | 87.69 | 87.95 [84.30, 91.08] |
| TEST | F10 | 3 | SNOMED | ProcedureCode | 492 | 35.77 [31.48, 39.67] | 70.53 | 75.81 | 77.03 [73.07, 80.35] |
| TEST | F10 | 3 | SNOMED | LabService | 144 | 43.75 [36.11, 51.08] | 75.69 | 79.86 | 79.86 [73.55, 85.72] |
| TEST | F10 | 3 | SNOMED | EncounterType | 97 | 44.33 [36.17, 53.75] | 78.35 | 82.47 | 82.47 [76.29, 89.33] |
| TEST | F10 | 3 | SNOMED | Device | 78 | 52.56 [41.89, 63.29] | 83.33 | 87.18 | 88.46 [80.60, 95.00] |
| TEST | F10 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F10 | 3 | Combined | Primary minimum | — | 44.80 [42.03, 47.91] | 77.35 | 81.43 | 82.10 |
| TEST | F01 | 3 | ICD | All | 326 | 65.03 [60.00, 69.85] | 85.58 | 86.50 | 86.50 [82.81, 90.52] |
| TEST | F01 | 3 | ICD | DiagnosisCode | 326 | 65.03 [60.00, 69.85] | 85.58 | 86.50 | 86.50 [82.81, 90.52] |
| TEST | F01 | 3 | ICD | ProcedureCode | 0 | NA | NA | NA | NA |
| TEST | F01 | 3 | SNOMED | All | 1201 | 50.46 [47.39, 53.15] | 76.85 | 79.77 | 80.02 [77.75, 82.19] |
| TEST | F01 | 3 | SNOMED | DiagnosisCode | 390 | 61.54 [56.34, 66.59] | 84.36 | 86.67 | 86.67 [83.12, 89.93] |
| TEST | F01 | 3 | SNOMED | ProcedureCode | 492 | 42.28 [38.08, 46.01] | 71.34 | 73.98 | 74.19 [70.27, 77.62] |
| TEST | F01 | 3 | SNOMED | LabService | 144 | 44.44 [36.60, 52.89] | 72.92 | 76.39 | 76.39 [70.00, 83.02] |
| TEST | F01 | 3 | SNOMED | EncounterType | 97 | 48.45 [39.17, 58.70] | 79.38 | 82.47 | 82.47 [76.09, 89.11] |
| TEST | F01 | 3 | SNOMED | Device | 78 | 60.26 [48.75, 70.79] | 78.21 | 84.62 | 87.18 [78.57, 94.29] |
| TEST | F01 | 3 | SNOMED | MedicationProfileMedication | 0 | NA | NA | NA | NA |
| TEST | F01 | 3 | Combined | Primary minimum | — | 50.46 [47.58, 53.32] | 76.85 | 79.77 | 80.02 |

## Paired final-flow changes

Percentage points. These are paired grouped-bootstrap differences on identical rows.

| Set | Comparison | ICD delta (95% CI) | SNOMED all delta (95% CI) | DiagnosisCode delta (95% CI) | Primary minimum delta (95% CI) |
|---|---|---|---|---|---|
| DEV | F10-F00 | +3.98 [-0.31, 7.99] | +2.11 [0.38, 3.82] | +0.78 [-1.83, 3.31] | +2.11 [0.38, 3.82] |
| DEV | F11-F01 | +6.12 [2.42, 10.29] | +0.77 [-1.02, 2.40] | -0.52 [-3.35, 2.34] | +0.77 [-1.02, 2.40] |
| DEV | F01-F00 | +7.95 [3.42, 12.61] | +7.40 [5.12, 9.71] | +5.96 [2.36, 9.43] | +7.40 [5.12, 9.71] |
| DEV | F11-F10 | +10.09 [5.55, 14.54] | +6.05 [4.05, 8.04] | +4.66 [1.42, 7.85] | +6.05 [4.05, 8.04] |
| DEV | F11-F00 | +14.07 [8.96, 19.18] | +8.17 [5.74, 10.52] | +5.44 [1.47, 9.14] | +8.17 [5.74, 10.52] |
| TEST | F10-F00 | +9.20 [5.76, 12.93] | +2.08 [0.58, 3.63] | +3.59 [0.77, 6.61] | +2.08 [0.58, 3.63] |
| TEST | F11-F01 | +3.37 [0.00, 6.96] | +0.25 [-1.13, 1.92] | +0.51 [-2.33, 3.65] | +0.25 [-1.13, 1.92] |
| TEST | F01-F00 | +11.96 [7.37, 16.72] | +7.74 [5.36, 9.97] | +10.00 [6.05, 13.88] | +7.74 [5.36, 9.97] |
| TEST | F11-F10 | +6.13 [2.50, 10.29] | +5.91 [3.74, 7.92] | +6.92 [3.30, 10.40] | +5.91 [3.74, 7.92] |
| TEST | F11-F00 | +15.34 [11.04, 20.69] | +7.99 [5.61, 10.37] | +10.51 [6.38, 14.54] | +7.99 [5.61, 10.37] |

## Final unseen / rare / common chart performance

Unseen = zero occurrences of the code in training; rare = 1–5; common = more than 5.

| Set | Flow | Stage | System | Category | n | Accuracy = R@1 (95% CI) | R@10 | R@32 | R@64 (95% CI) |
|---|---|---|---|---|---:|---|---:|---:|---|
| DEV | F00 | 3 | ICD | common | 78 | 52.56 [41.77, 62.96] | 74.36 | 82.05 | 83.33 [73.97, 91.03] |
| DEV | F00 | 3 | ICD | rare | 168 | 62.50 [55.62, 69.50] | 89.88 | 91.67 | 91.67 [87.57, 95.71] |
| DEV | F00 | 3 | ICD | unseen | 81 | 33.33 [21.69, 44.58] | 70.37 | 76.54 | 76.54 [66.26, 85.19] |
| DEV | F00 | 3 | SNOMED | common | 63 | 53.97 [42.86, 65.57] | 87.30 | 87.30 | 87.30 [78.46, 95.24] |
| DEV | F00 | 3 | SNOMED | rare | 468 | 57.91 [53.29, 62.67] | 85.90 | 89.10 | 89.32 [86.37, 91.96] |
| DEV | F00 | 3 | SNOMED | unseen | 510 | 28.24 [24.45, 32.39] | 67.45 | 76.08 | 77.65 [73.89, 81.31] |
| DEV | F10 | 3 | ICD | common | 78 | 56.41 [44.44, 67.95] | 91.03 | 94.87 | 94.87 [89.33, 98.75] |
| DEV | F10 | 3 | ICD | rare | 168 | 65.48 [58.28, 72.73] | 91.67 | 93.45 | 93.45 [89.65, 96.91] |
| DEV | F10 | 3 | ICD | unseen | 81 | 39.51 [27.84, 51.16] | 72.84 | 81.48 | 81.48 [71.95, 90.00] |
| DEV | F10 | 3 | SNOMED | common | 63 | 63.49 [53.03, 75.00] | 93.65 | 93.65 | 93.65 [87.10, 98.48] |
| DEV | F10 | 3 | SNOMED | rare | 468 | 60.04 [55.53, 64.63] | 89.96 | 92.95 | 92.95 [90.44, 95.29] |
| DEV | F10 | 3 | SNOMED | unseen | 510 | 29.41 [25.57, 33.53] | 69.22 | 76.47 | 78.24 [74.40, 81.80] |
| DEV | F01 | 3 | ICD | common | 78 | 60.26 [49.40, 70.24] | 82.05 | 83.33 | 83.33 [73.97, 91.03] |
| DEV | F01 | 3 | ICD | rare | 168 | 71.43 [65.17, 77.84] | 92.26 | 92.26 | 92.26 [88.24, 95.88] |
| DEV | F01 | 3 | ICD | unseen | 81 | 39.51 [27.85, 51.33] | 70.37 | 76.54 | 76.54 [66.26, 85.19] |
| DEV | F01 | 3 | SNOMED | common | 63 | 66.67 [56.25, 77.78] | 87.30 | 87.30 | 87.30 [78.46, 95.24] |
| DEV | F01 | 3 | SNOMED | rare | 468 | 63.46 [59.39, 67.69] | 87.39 | 89.32 | 89.32 [86.37, 91.96] |
| DEV | F01 | 3 | SNOMED | unseen | 510 | 36.67 [32.40, 41.12] | 70.59 | 77.25 | 77.65 [73.89, 81.31] |
| DEV | F11 | 3 | ICD | common | 78 | 73.08 [62.81, 82.50] | 93.59 | 94.87 | 94.87 [89.33, 98.75] |
| DEV | F11 | 3 | ICD | rare | 168 | 73.81 [67.27, 80.36] | 93.45 | 93.45 | 93.45 [89.65, 96.91] |
| DEV | F11 | 3 | ICD | unseen | 81 | 46.91 [35.00, 58.98] | 76.54 | 81.48 | 81.48 [71.95, 90.00] |
| DEV | F11 | 3 | SNOMED | common | 63 | 69.84 [59.02, 80.00] | 93.65 | 93.65 | 93.65 [87.10, 98.48] |
| DEV | F11 | 3 | SNOMED | rare | 468 | 65.17 [60.77, 69.80] | 91.24 | 92.74 | 92.95 [90.44, 95.29] |
| DEV | F11 | 3 | SNOMED | unseen | 510 | 36.27 [31.85, 40.64] | 70.39 | 77.84 | 78.24 [74.40, 81.80] |
| TEST | F00 | 3 | ICD | common | 93 | 54.84 [44.44, 65.27] | 87.10 | 90.32 | 90.32 [84.21, 95.70] |
| TEST | F00 | 3 | ICD | rare | 156 | 57.05 [48.38, 65.00] | 86.54 | 89.10 | 89.10 [84.05, 94.16] |
| TEST | F00 | 3 | ICD | unseen | 77 | 42.86 [32.43, 54.29] | 75.32 | 76.62 | 76.62 [66.26, 86.11] |
| TEST | F00 | 3 | SNOMED | common | 86 | 65.12 [55.43, 76.47] | 87.21 | 88.37 | 88.37 [81.61, 95.06] |
| TEST | F00 | 3 | SNOMED | rare | 494 | 55.67 [51.38, 60.32] | 84.62 | 87.65 | 87.65 [84.75, 90.47] |
| TEST | F00 | 3 | SNOMED | unseen | 621 | 29.31 [25.32, 32.83] | 64.73 | 71.50 | 72.79 [68.90, 75.97] |
| TEST | F11 | 3 | ICD | common | 93 | 68.82 [58.69, 77.89] | 91.40 | 92.47 | 92.47 [86.81, 96.94] |
| TEST | F11 | 3 | ICD | rare | 156 | 73.72 [66.19, 80.63] | 94.23 | 95.51 | 95.51 [91.67, 98.73] |
| TEST | F11 | 3 | ICD | unseen | 77 | 57.14 [45.83, 67.57] | 79.22 | 81.82 | 81.82 [72.22, 90.54] |
| TEST | F11 | 3 | SNOMED | common | 86 | 67.44 [52.88, 81.58] | 93.02 | 93.02 | 93.02 [87.88, 97.59] |
| TEST | F11 | 3 | SNOMED | rare | 494 | 63.16 [58.79, 67.51] | 89.07 | 91.50 | 91.50 [88.76, 93.65] |
| TEST | F11 | 3 | SNOMED | unseen | 621 | 38.49 [33.76, 42.21] | 67.63 | 72.62 | 73.11 [69.23, 76.40] |
| TEST | F10 | 3 | ICD | common | 93 | 63.44 [53.84, 72.92] | 87.10 | 92.47 | 92.47 [86.81, 96.94] |
| TEST | F10 | 3 | ICD | rare | 156 | 66.67 [58.66, 74.03] | 91.03 | 95.51 | 95.51 [91.67, 98.73] |
| TEST | F10 | 3 | ICD | unseen | 77 | 51.95 [40.84, 63.64] | 76.62 | 81.82 | 81.82 [72.22, 90.54] |
| TEST | F10 | 3 | SNOMED | common | 86 | 63.95 [53.84, 74.70] | 91.86 | 93.02 | 93.02 [87.88, 97.59] |
| TEST | F10 | 3 | SNOMED | rare | 494 | 58.10 [53.86, 62.65] | 88.26 | 91.50 | 91.50 [88.76, 93.65] |
| TEST | F10 | 3 | SNOMED | unseen | 621 | 31.56 [27.71, 34.99] | 66.67 | 71.82 | 73.11 [69.23, 76.40] |
| TEST | F01 | 3 | ICD | common | 93 | 67.74 [58.16, 77.18] | 89.25 | 90.32 | 90.32 [84.21, 95.70] |
| TEST | F01 | 3 | ICD | rare | 156 | 67.31 [58.82, 74.50] | 88.46 | 89.10 | 89.10 [84.05, 94.16] |
| TEST | F01 | 3 | ICD | unseen | 77 | 57.14 [45.78, 68.12] | 75.32 | 76.62 | 76.62 [66.26, 86.11] |
| TEST | F01 | 3 | SNOMED | common | 86 | 69.77 [58.69, 82.28] | 88.37 | 88.37 | 88.37 [81.61, 95.06] |
| TEST | F01 | 3 | SNOMED | rare | 494 | 60.93 [56.49, 65.20] | 86.44 | 87.65 | 87.65 [84.75, 90.47] |
| TEST | F01 | 3 | SNOMED | unseen | 621 | 39.45 [34.51, 43.46] | 67.63 | 72.30 | 72.79 [68.90, 75.97] |

## Dictionary probes, all stages

These are separate alias/noise diagnostics, not chart accuracy.

| Set | Flow | Stage | System | Category | n | Accuracy = R@1 (95% CI) | R@10 | R@32 | R@64 (95% CI) |
|---|---|---|---|---|---:|---|---:|---:|---|
| PROBE | F00 | 1 | ICD | All | 3000 | 12.23 [10.47, 13.97] | 43.43 | 56.53 | 84.23 [82.50, 86.00] |
| PROBE | F00 | 1 | SNOMED | All | 3000 | 43.53 [41.10, 45.80] | 78.93 | 87.77 | 91.87 [90.43, 93.17] |
| PROBE | F00 | 2 | ICD | All | 3000 | 17.33 [15.60, 19.20] | 58.00 | 77.77 | 84.23 [82.50, 86.00] |
| PROBE | F00 | 2 | SNOMED | All | 3000 | 39.60 [37.43, 41.83] | 82.80 | 90.20 | 91.87 [90.43, 93.17] |
| PROBE | F00 | 3 | ICD | All | 3000 | 15.07 [13.43, 16.87] | 52.03 | 74.30 | 84.53 [82.83, 86.33] |
| PROBE | F00 | 3 | SNOMED | All | 3000 | 51.60 [49.00, 53.90] | 88.03 | 91.07 | 91.87 [90.43, 93.17] |
| PROBE | F10 | 1 | ICD | All | 3000 | 13.23 [11.47, 14.97] | 44.67 | 58.07 | 85.70 [84.03, 87.43] |
| PROBE | F10 | 1 | SNOMED | All | 3000 | 41.97 [39.67, 44.10] | 78.00 | 87.57 | 91.90 [90.47, 93.23] |
| PROBE | F10 | 2 | ICD | All | 3000 | 17.40 [15.63, 19.30] | 58.40 | 78.97 | 85.70 [84.03, 87.43] |
| PROBE | F10 | 2 | SNOMED | All | 3000 | 39.13 [36.97, 41.47] | 82.67 | 90.13 | 91.90 [90.47, 93.23] |
| PROBE | F10 | 3 | ICD | All | 3000 | 22.13 [20.20, 24.23] | 71.67 | 83.97 | 85.80 [84.17, 87.50] |
| PROBE | F10 | 3 | SNOMED | All | 3000 | 56.70 [54.30, 58.90] | 88.63 | 91.40 | 91.90 [90.47, 93.23] |
| PROBE | F01 | 1 | ICD | All | 3000 | 12.23 [10.47, 13.97] | 43.43 | 56.53 | 84.23 [82.50, 86.00] |
| PROBE | F01 | 1 | SNOMED | All | 3000 | 43.53 [41.10, 45.80] | 78.93 | 87.77 | 91.87 [90.43, 93.17] |
| PROBE | F01 | 2 | ICD | All | 3000 | 71.90 [69.73, 74.17] | 82.23 | 83.83 | 84.23 [82.50, 86.00] |
| PROBE | F01 | 2 | SNOMED | All | 3000 | 76.67 [74.57, 78.63] | 88.97 | 91.27 | 91.87 [90.43, 93.17] |
| PROBE | F01 | 3 | ICD | All | 3000 | 60.97 [58.90, 63.24] | 82.87 | 84.87 | 85.20 [83.57, 86.97] |
| PROBE | F01 | 3 | SNOMED | All | 3000 | 77.00 [75.07, 78.97] | 90.37 | 91.60 | 91.87 [90.43, 93.17] |
| PROBE | F11 | 1 | ICD | All | 3000 | 13.23 [11.47, 14.97] | 44.67 | 58.07 | 85.70 [84.03, 87.43] |
| PROBE | F11 | 1 | SNOMED | All | 3000 | 41.97 [39.67, 44.10] | 78.00 | 87.57 | 91.90 [90.47, 93.23] |
| PROBE | F11 | 2 | ICD | All | 3000 | 73.43 [71.33, 75.60] | 83.57 | 85.37 | 85.70 [84.03, 87.43] |
| PROBE | F11 | 2 | SNOMED | All | 3000 | 76.27 [74.03, 78.20] | 89.07 | 91.33 | 91.90 [90.47, 93.23] |
| PROBE | F11 | 3 | ICD | All | 3000 | 67.03 [65.03, 69.27] | 84.30 | 85.73 | 86.03 [84.40, 87.70] |
| PROBE | F11 | 3 | SNOMED | All | 3000 | 77.20 [75.23, 79.07] | 90.63 | 91.70 | 91.90 [90.47, 93.23] |

### Final clean versus noisy probes

| Set | Flow | Stage | System | Category | n | Accuracy = R@1 (95% CI) | R@10 | R@32 | R@64 (95% CI) |
|---|---|---|---|---|---:|---|---:|---:|---|
| PROBE | F00 | 3 | ICD | clean | 1500 | 15.93 [14.20, 17.80] | 52.80 | 74.07 | 84.73 [83.00, 86.60] |
| PROBE | F00 | 3 | ICD | noisy | 1500 | 14.20 [12.47, 15.93] | 51.27 | 74.53 | 84.33 [82.60, 86.14] |
| PROBE | F00 | 3 | SNOMED | clean | 1500 | 53.87 [51.27, 56.27] | 88.93 | 91.67 | 92.27 [90.80, 93.53] |
| PROBE | F00 | 3 | SNOMED | noisy | 1500 | 49.33 [46.67, 51.73] | 87.13 | 90.47 | 91.47 [90.00, 92.87] |
| PROBE | F10 | 3 | ICD | clean | 1500 | 22.93 [20.87, 25.20] | 71.73 | 84.13 | 86.00 [84.27, 87.80] |
| PROBE | F10 | 3 | ICD | noisy | 1500 | 21.33 [19.33, 23.40] | 71.60 | 83.80 | 85.60 [83.93, 87.33] |
| PROBE | F10 | 3 | SNOMED | clean | 1500 | 58.60 [56.07, 60.87] | 89.73 | 91.80 | 92.33 [90.80, 93.67] |
| PROBE | F10 | 3 | SNOMED | noisy | 1500 | 54.80 [52.20, 57.20] | 87.53 | 91.00 | 91.47 [90.00, 92.87] |
| PROBE | F01 | 3 | ICD | clean | 1500 | 63.13 [60.93, 65.40] | 83.60 | 85.53 | 85.73 [84.00, 87.47] |
| PROBE | F01 | 3 | ICD | noisy | 1500 | 58.80 [56.60, 61.40] | 82.13 | 84.20 | 84.67 [82.93, 86.53] |
| PROBE | F01 | 3 | SNOMED | clean | 1500 | 77.73 [75.73, 79.67] | 90.87 | 92.07 | 92.27 [90.80, 93.53] |
| PROBE | F01 | 3 | SNOMED | noisy | 1500 | 76.27 [74.20, 78.33] | 89.87 | 91.13 | 91.47 [90.00, 92.87] |
| PROBE | F11 | 3 | ICD | clean | 1500 | 68.53 [66.40, 71.00] | 84.67 | 86.07 | 86.33 [84.67, 88.07] |
| PROBE | F11 | 3 | ICD | noisy | 1500 | 65.53 [63.27, 67.87] | 83.93 | 85.40 | 85.73 [84.06, 87.47] |
| PROBE | F11 | 3 | SNOMED | clean | 1500 | 78.07 [76.13, 79.94] | 91.13 | 92.27 | 92.33 [90.80, 93.67] |
| PROBE | F11 | 3 | SNOMED | noisy | 1500 | 76.33 [74.07, 78.40] | 90.13 | 91.13 | 91.47 [90.00, 92.87] |

## Other calculated diagnostics

These metrics are supplementary to exact-code accuracy. Full metrics include error-class counts, all recall cutoffs, risk/coverage, low-confidence rates, selection given the final pool, category/tag/top-level/core accuracy, over/under-coding and header counts.

| Set | Flow | Pooled local-label accuracy | DiagnosisCode minimum | Paired inputs n | Both correct (95% CI) | Low confidence % |
|---|---|---:|---:|---:|---|---:|
| DEV | F00 | 45.47 | 51.81 | 307 | 34.85 [29.67, 40.19] | 5.19 |
| DEV | F10 | 48.03 | 52.59 | 307 | 39.74 [34.51, 45.14] | 4.90 |
| DEV | F01 | 53.00 | 57.77 | 307 | 40.72 [35.20, 46.41] | 5.12 |
| DEV | F11 | 55.04 | 57.25 | 307 | 46.58 [41.17, 52.34] | 5.19 |
| TEST | F00 | 44.92 | 51.54 | 300 | 37.67 [32.11, 42.60] | 5.63 |
| TEST | F10 | 48.53 | 55.13 | 300 | 43.67 [37.84, 49.64] | 4.98 |
| TEST | F01 | 53.57 | 61.54 | 300 | 50.33 [43.93, 56.21] | 5.44 |
| TEST | F11 | 54.49 | 62.05 | 300 | 51.67 [45.67, 57.34] | 5.96 |

### Final hierarchy and specificity diagnostics

ICD category accuracy means the first three code characters match; it is different from a supplied subconcept category. SNOMED tag and top-level matches are soft diagnostic measures. Rates are percentages; header violations are counts.

| Set | Flow | System | ICD category accuracy | SNOMED tag accuracy | Top-level accuracy | Same core | Over-coding % | Under-coding % | Illegal headers |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | F00 | ICD | 66.97 | NA | NA | NA | 1.22 | 2.14 | 0 |
| DEV | F00 | SNOMED | NA | 66.76 | 71.57 | 44.38 | 0.10 | 0.38 | 0 |
| DEV | F10 | ICD | 71.56 | NA | NA | NA | 2.45 | 1.83 | 0 |
| DEV | F10 | SNOMED | NA | 68.59 | 72.91 | 46.88 | 0.19 | 0.58 | 0 |
| DEV | F01 | ICD | 74.01 | NA | NA | NA | 1.22 | 1.22 | 0 |
| DEV | F01 | SNOMED | NA | 71.76 | 75.60 | 51.78 | 0.10 | 0.29 | 0 |
| DEV | F11 | ICD | 77.98 | NA | NA | NA | 1.22 | 1.53 | 0 |
| DEV | F11 | SNOMED | NA | 73.49 | 77.14 | 52.74 | 0.10 | 0.29 | 0 |
| TEST | F00 | ICD | 65.64 | NA | NA | NA | 1.84 | 3.99 | 0 |
| TEST | F00 | SNOMED | NA | 65.03 | 69.94 | 43.88 | 0.42 | 0.33 | 0 |
| TEST | F10 | ICD | 71.78 | NA | NA | NA | 1.84 | 2.45 | 0 |
| TEST | F10 | SNOMED | NA | 67.86 | 72.44 | 46.13 | 0.33 | 0.33 | 0 |
| TEST | F01 | ICD | 73.31 | NA | NA | NA | 1.53 | 0.92 | 0 |
| TEST | F01 | SNOMED | NA | 71.19 | 75.35 | 51.46 | 0.25 | 0.17 | 0 |
| TEST | F11 | ICD | 77.30 | NA | NA | NA | 1.84 | 1.53 | 0 |
| TEST | F11 | SNOMED | NA | 72.52 | 76.10 | 51.62 | 0.25 | 0.25 | 0 |

### Final SNOMED error classes

Counts over supplied SNOMED gold labels; correctly predicted rows are excluded.

| Set | Flow | Near-miss | Same tag, different concept | Same top-level, different tag | Wrong top-level |
|---|---|---:|---:|---:|---:|
| DEV | F00 | 10 | 240 | 50 | 292 |
| DEV | F10 | 13 | 235 | 45 | 277 |
| DEV | F01 | 8 | 216 | 40 | 251 |
| DEV | F11 | 10 | 226 | 38 | 233 |
| TEST | F00 | 11 | 258 | 59 | 360 |
| TEST | F10 | 13 | 267 | 55 | 328 |
| TEST | F01 | 10 | 243 | 50 | 292 |
| TEST | F11 | 10 | 254 | 43 | 285 |

### Final risk at confidence-based coverage

Risk is the percentage of errors among the retained highest-confidence rows. This pooled supplied-label diagnostic is supplementary to the primary per-system minimum.

| Set | Flow | 10% coverage | 25% | 50% | 75% | 90% | 95% | 100% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | F00 | 9.49 | 19.59 | 31.73 | 44.15 | 50.81 | 52.85 | 54.53 |
| DEV | F10 | 8.03 | 14.91 | 27.92 | 41.33 | 47.73 | 50.08 | 51.97 |
| DEV | F01 | 7.30 | 13.74 | 24.27 | 35.58 | 42.69 | 45.00 | 47.00 |
| DEV | F11 | 7.30 | 11.70 | 22.08 | 34.11 | 40.58 | 42.46 | 44.96 |
| TEST | F00 | 5.88 | 17.80 | 32.46 | 44.59 | 51.35 | 53.27 | 55.08 |
| TEST | F10 | 2.61 | 13.35 | 27.49 | 39.88 | 47.27 | 49.21 | 51.47 |
| TEST | F01 | 4.58 | 11.26 | 23.43 | 34.64 | 41.60 | 43.97 | 46.43 |
| TEST | F11 | 3.27 | 8.64 | 22.64 | 34.55 | 40.51 | 43.07 | 45.51 |

### Offline TEST scoring time

These timings include both coding systems per input. Direct timings include worker/model startup; hosted timings exclude service startup. F10 records its slowest parallel shard and excludes launch staggering. These are not single-request or complete online latency benchmarks. Historical cached/client latency and VRAM fields in the full JSON must not be interpreted as such; vLLM client VRAM zero does not mean the hosted model uses zero GPU memory.

| Flow | Scoring minutes | Frozen decision application seconds |
|---|---:|---:|
| F00 | 22.65 | 8.26 |
| F11 | 10.93 | 8.00 |
| F10 | 11.54 | 7.96 |
| F01 | 2.72 | 8.12 |

## Backends, leakage checks and saved outputs

TEST F00/F10 both use original Qwen via Transformers on A6000/sm86 with identical per-input batching and scorer identity. TEST F01/F11 both use epoch-2 LoRA via vLLM 0.19.0 on Blackwell/sm120 with identical scorer identity. Matching cached token scores are reused only when model, prompt/packing, serving version and architecture match. Cross-reranker comparisons still include possible bf16 backend/GPU effects; they are not a pure training-effect estimate. DEV preserves the earlier F00/F10 Blackwell scores and uses the completed F01/F11 A6000 scores, as previously requested.

F10/F01 TEST were added after the initial F00/F11 report, using the same already frozen models, DEV fitting policy and decisions. TEST was not used to change weights, prompts, parameters or selection. All previous results remain unchanged. TEST and DEV groups are disjoint. Each input has predictions for both systems, but only its supplied gold system contributes a label. All twelve DEV/probe/TEST validators pass.

- DEV F00: metrics: `private_assets/evaluation/F00/dev_metrics.json`, predictions: `private_assets/evaluation/F00/dev_predictions.csv`, diagnostics: `private_assets/evaluation/F00/dev_details.parquet`, validator: `private_assets/evaluation/F00/dev_validation.json`.
- DEV F10: metrics: `private_assets/evaluation/F10/dev_metrics.json`, predictions: `private_assets/evaluation/F10/dev_predictions.csv`, diagnostics: `private_assets/evaluation/F10/dev_details.parquet`, validator: `private_assets/evaluation/F10/dev_validation.json`.
- DEV F01: metrics: `private_assets/evaluation/F01/dev_metrics.json`, predictions: `private_assets/evaluation/F01/dev_predictions.csv`, diagnostics: `private_assets/evaluation/F01/dev_details.parquet`, validator: `private_assets/evaluation/F01/dev_validation.json`.
- DEV F11: metrics: `private_assets/evaluation/F11/dev_metrics.json`, predictions: `private_assets/evaluation/F11/dev_predictions.csv`, diagnostics: `private_assets/evaluation/F11/dev_details.parquet`, validator: `private_assets/evaluation/F11/dev_validation.json`.
- TEST F00: metrics: `private_assets/evaluation/F00/test_metrics.json`, predictions: `private_assets/evaluation/F00/test_predictions.csv`, diagnostics: `private_assets/evaluation/F00/test_details.parquet`, validator: `private_assets/evaluation/F00/test_validation.json`.
- TEST F11: metrics: `private_assets/evaluation/F11/test_metrics.json`, predictions: `private_assets/evaluation/F11/test_predictions.csv`, diagnostics: `private_assets/evaluation/F11/test_details.parquet`, validator: `private_assets/evaluation/F11/test_validation.json`.
- TEST F10: metrics: `private_assets/evaluation/F10/test_metrics.json`, predictions: `private_assets/evaluation/F10/test_predictions.csv`, diagnostics: `private_assets/evaluation/F10/test_details.parquet`, validator: `private_assets/evaluation/F10/test_validation.json`.
- TEST F01: metrics: `private_assets/evaluation/F01/test_metrics.json`, predictions: `private_assets/evaluation/F01/test_predictions.csv`, diagnostics: `private_assets/evaluation/F01/test_details.parquet`, validator: `private_assets/evaluation/F01/test_validation.json`.

[Aggregate CSV](Final_all_flows_metrics.csv) contains the full flat table, including all saved recall cutoffs and bucket/subtype records. [Complete aggregate metrics JSON](Final_all_flows_full_metrics.json) retains every calculated metric and all ten paired comparisons. Neither contains clinical text or per-row predictions.


Per-input predictions, diagnostics and validation files named above are distributed only in the restricted private bundle. See [results provenance and private-file access](../docs/RESULTS_PROVENANCE.md).
