# Final SapBERT P2-v2 pretraining

`trainer.py` is the final method: query-side adaptation against a frozen off-the-shelf alias index using full-concept InfoNCE. It replaces the abandoned v1 symmetric metric-learning recipe. A concept score takes the maximum over its alias vectors and the mention's dense views. Shared gold aliases and self aliases are masked appropriately; rare chart mentions receive inverse-square-root frequency weights. The complete negative universe is all other unmasked concepts, rather than a stored small negative-pair list.

The saved training generation is pinned independently of the later preprocessing generation. Exported material contains 6,749 chart-mention examples, 3,374 sampled alias/noise examples per epoch for 10 epochs, exact seeded batch orders, positive alias identities and masked alias rows. DEV/probe reference records are clearly monitoring-only. Reconstructed datasets are labelled as deterministic reconstructions because the training run did not save materialized epoch datasets.

Selected full query encoders: ICD step 1431 / epoch 9.0; SNOMED step 950 / epoch 5.975. Only these selected checkpoints were persisted. No SapBERT epoch-1/epoch-2 checkpoints or optimizer-resume state are invented. Original dictionary vectors remain unchanged at inference.

Settings: max 40 tokens, normalized CLS, batch 64, lr 2e-5, temperature 0.03, weight decay 0.01, warmup 10%, clip 1.0, evaluation every 50 steps, max 10 epochs, seed 13. Exact settings and selection metadata are in the private bundle. Commands are under `scripts.pretraining`; training writes to its separate training config. No training is launched as part of repository preparation.
