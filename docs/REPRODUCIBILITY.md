# Reproduction and datasets

The ZIP expands to ignored `private_assets/`; it is separate from Git. It includes raw dictionary workbooks, clean train/DEV/probe tables and train-derived priors/aliases; it excludes the raw full labelled CSV and sealed TEST. This preserves training/evaluation boundaries while supplying the actual inputs used. Dictionary hashes and per-file SHA256 hashes identify the source release.

## SapBERT dataset format

`datasets/pretraining/mention_examples.parquet` contains 6,749 valid train mention examples, system/string code, all dense text views, row/group IDs, rare-code weight, positive alias row IDs/text and masked alias row IDs. `alias_examples_epoch_01.parquet` through `_10.parquet` contain 3,374 sampled alias/noise examples each; `batch_order_epoch_*.parquet` preserves shuffled batch order. `{ICD,SNOMED}_frozen_alias_rows.parquet` defines the row identity for positives/masks. Other unmasked concepts supply full-index InfoNCE negatives; a small pair file would misrepresent this final method.

These are deterministic reconstructions from the saved pinned text generation, final sampler and seed 13, with counts checked against the recorded audit. The original run did not persist materialized epoch datasets. The manifest says this explicitly. Reference train/DEV/probe/alias records and query texts support restoration; DEV/probe are monitoring only.

To reproduce that training generation, run `scripts.pretraining.restore_inputs` with `configs/pretraining.yaml`; run `scripts.pretraining.prepare_index` through Slurm to rebuild lexical/dense C1 inputs at that pinned path using original SapBERT, then run the included audit and trainer through Slurm. Training configs have separate output directories, so distributed selected checkpoints are preserved. The finalized weights themselves require no retraining.

## Qwen dataset format

`datasets/posttraining/quizzes_epoch_1.parquet` and `_2.parquet` preserve 10,502 ordered groups each. Each row contains the candidate-code list, gold index, type/Seen visibility, exclusion identifiers and all 16 exact candidate payloads. Both epochs have 168,032 candidate judgments. Reconstruction imports the immutable training-runtime snapshot, verifies its frozen sources/inputs and checks the recorded epoch group count. No DEV/TEST example is added to the training quizzes.

The current source includes the same seeded quiz builder and LoRA loss/trainer. With the default config, build `scripts.retrieval.build_indexes --training-pools` through Slurm; then select `configs/posttraining.yaml`, run `scripts.posttraining.prepare_pools` on CPU, and use its training wrapper. Reproducing candidate mining requires prepared original-SapBERT train/alias pools; direct dataset inspection/replay does not require distributing the historical score caches.

## Inference assets

The refreshed ZIP already includes the frozen main-run decision files, original train-fitted type classifier, validated outputs and portable release manifest. Run a GPU-index smoke and full `scripts.retrieval.build_indexes` through Slurm after restoring it. This rebuilds disposable exact search tables from original alias vectors and selected per-system encoders; no model training occurs. Do not encode aliases with adapted checkpoints. Then complete the one-command prediction acceptance test on the target GPU.

The final handoff has been imported. To repeat it from the same research checkout, `python tools/import_final_handoff.py /path/to/HackLabs` verifies all twelve validators, saved prediction hashes, published aggregates and frozen decision/DEV hashes before copying the four-flow artifacts. It writes the portable release and results manifests without opening the raw sealed partition or fitting a model. Run `tools/repack_assets.py --output /path/to/private_artifacts.zip` after any new handoff to refresh the ZIP and published checksum.

The repository is a source/artifact handoff, not a second evaluation run. Final DEV/TEST results are imported with their exact model/backend/GPU provenance and unchanged metric values. A fresh deployment on different hardware is verified separately; previous metrics are not silently claimed as a new run.

All random operations use seed 13; torch deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, rounding to 1e-5 and code-string tie breaking are retained. Codes stay strings. Training memory/priors use train only; training rows hide their entire group, and probe aliases remain globally excluded in evaluation-time indexes/training.
