# Final results and artifact provenance

The authoritative submission results are [results/README.md](../results/README.md) and [the complete four-flow report](../results/Final_all_flows_DEV_TEST.md), imported from the finalized research results. All four flows are complete on DEV, internal TEST and separate dictionary probes. Earlier LoRA and two-flow TEST snapshots remain in the research repository and are not presented as current submission results.

## Evaluation scope

| Set | Supplied labels | Protocol |
|---|---|---|
| Training | 6,751 | Train-only learning, memory, terminology priors and type classifier |
| DEV | 1,368: 327 ICD, 1,041 SNOMED | Independently fitted nested grouped cross-validation for each flow |
| Internal TEST | 1,527: 326 ICD, 1,201 SNOMED | Unchanged frozen full-DEV decision models; no TEST fitting or checkpoint selection |
| Dictionary probes | 6,000 | 1,500 codes/system, each with a withheld clean alias and noisy variant |

Primary score is the minimum of ICD and SNOMED **all-types** exact-code accuracies. DiagnosisCode-only and paired-both-correct values are supplementary. Only supplied gold labels are evaluated; opposite-system predictions never become labels. Recall@64 measures candidate availability, and is kept separate from final top-1 accuracy. All observed labels and the single ICD ProcedureCode DEV row remain in their original denominators. Internal TEST is not an organizer-provided hidden evaluation.

Every flow uses the latest reviewed preprocessing. F00/F10 use the original Qwen reranker; F01/F11 use revised-input LoRA epoch 2. F10/F11 use selected per-system SapBERT-v2 query/memory encoders while alias and type-inference vectors remain original SapBERT. The bridge stays disabled. Every flow includes its own decision layer. There is no all-label retrain claim.

## Integrity checks

[results/MANIFEST.json](../results/MANIFEST.json) records source and packaged report hashes, the four frozen decision hashes and validation/metric/prediction hashes for all twelve flow/set outputs. All twelve saved validators pass; DEV validators confirm held-out-group isolation. Each copied prediction file matches its validator's SHA256, each final aggregate matches the corresponding saved metric file, and DEV metric/decision hashes match the epoch-2 freeze.

The 664-row CSV and both aggregate JSON exports are copied byte for byte. Every Markdown metric-table line is preserved exactly. Packaging changes navigation only: research-local prediction links become private-bundle paths, obsolete snapshot links are omitted, and the problem-statement link points to the submission requirements. The import opens no raw sealed partition, performs no model or decision fitting, and reruns no evaluation.

Backend provenance is retained: original-model DEV uses vLLM/Blackwell, revised-LoRA DEV uses vLLM/A6000; original-model TEST uses Transformers/A6000 and revised-LoRA TEST uses vLLM/Blackwell. Cross-reranker deltas can include bf16 backend/hardware effects. Warm latency is an F11-only, uncached-query sequential benchmark with loaded models/indexes; offline batch scoring durations are not per-input latency.

## Private predictions and validation

The private ZIP restores these files under `private_assets/evaluation/{F00,F10,F01,F11}/`:

- `{dev,probe,test}_predictions.csv`: validated final output codes in original input order.
- `{dev,probe,test}_metrics.json` and `_raw_metrics.json`: saved final/raw-reranker metrics.
- `{dev,probe,test}_validation.json`: output and leakage safeguards.
- `{dev,probe,test}_details.parquet`, `_diagnostics.jsonl` and `_prediction_rows.parquet`: per-input diagnostics and identities, where saved.

Clinical inputs and per-input diagnostics remain outside Git and Hugging Face uploads. Paths in the full report identify files after extracting the restricted private bundle; they are not public download links. The bundle includes frozen `private_assets/models/decision/F*.joblib`, the train-fitted type classifier and the portable release manifest.

## Remaining submission checks

The saved research evaluation is complete. The curated deployment still needs its index rebuild and allocated-GPU acceptance test; CPU/import checks do not replace that. The documentation PDF, demo recording, restricted Drive upload and repository/model access for judges also remain separate handoff steps. The 95% accuracy target is not met, and missing-code detection is weak in the reported gold-removal stress test; these limitations must remain visible in the final report.
