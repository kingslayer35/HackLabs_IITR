# Private artifact bundle

Local file: `HackLabs_IITR_private_artifacts.zip` in the parent folder of this checkout. Keep it outside ordinary Git history and upload it only to the restricted Drive location.

- Size: 2,506,795,652 bytes (about 2.51 GB).
- Assets: 226, plus the root manifest.
- SHA256: `ea3d2590050a4fdf0fdb72280d0e42700ed0920f9411b72c83570062f2bea420`.
- Integrity: ZIP CRCs verified; no duplicate entries; per-file checksums in `private_assets/MANIFEST.json`.
- Models: original SapBERT/Qwen bases; selected per-system SapBERT-v2 encoders; both revised Qwen LoRA epochs and selected epoch 2; tokenizers, portable configs and original training settings.
- Final handoff: all four frozen decision models, the original train-fitted type classifier, portable release manifest and twelve validated DEV/probe/TEST prediction/diagnostic exports.
- Data: clean train/DEV/probe and terminology assets; exact seeded training-example/quiz reconstructions with their provenance. Raw full labelled CSV, sealed TEST, logs, optimizer-resume state and score caches are excluded.

The selected epoch-2 adapter hash is `c92175a6aa9af5d30700067a7fb630b594ba3a27e449db3f45fc7f45bfe97410`. The selected `best` copy has the same hash. SapBERT epoch-1/epoch-2 snapshots were not saved by the v2 run; only its two selected checkpoints are available.

This bundle is refreshed with the final decision layers and evaluation handoff. GPU deployment acceptance remains separate. The four trained model repositories are public and ungated under `yobro4619`, with anonymous weight access and checksums verified in [MODEL_UPLOADS.json](MODEL_UPLOADS.json) and links recorded in ARTIFACT_LINKS.json and README. Visibility changes do not change checkpoint or ZIP bytes. The ZIP contains clinical datasets and must remain restricted. The private Drive and demo URLs remain pending; upload this refreshed ZIP and use its current checksum.
