# Checkpoint inventory and loading

| Role | Included checkpoint | Kind | Selection |
|---|---|---|---|
| Original alias/type encoder | `private_assets/models/base_sapbert/` | Full original weights/tokenizer | Off-the-shelf SapBERT |
| ICD query/memory encoder | `private_assets/models/sapbert_v2/best_ICD/` | Full adapted weights/tokenizer | Step 1431, epoch 9.0 |
| SNOMED query/memory encoder | `private_assets/models/sapbert_v2/best_SNOMED/` | Full adapted weights/tokenizer | Step 950, epoch 5.975 |
| Original reranker base | `private_assets/models/base_qwen/` | Full original weights/tokenizer | Qwen3-Reranker-0.6B |
| Revised LoRA epoch 1 | `private_assets/models/reranker_lora/epoch_1/` | Adapter + tokenizer | Preserved for comparison |
| Revised LoRA epoch 2 | `private_assets/models/reranker_lora/epoch_2/` | Adapter + tokenizer | Explicit final choice |
| Selected reranker | `private_assets/models/reranker_lora/best/` | Exact epoch-2 weight copy | Same adapter bytes |
| Decision models | `private_assets/models/decision/F*.joblib` | Fitted non-neural parameters | Imported frozen full-DEV fits for all four flows; hashes match the evaluation freeze |
| Subconcept classifier | `private_assets/models/type_eval_*_C1.joblib` | Train-fitted classifier | Imported original-SapBERT classifier; shared across flows |

The portable `private_assets/models/RELEASE.json` verifies dictionary identity, retained settings, code and artifact hashes. It includes all twelve saved output validations. This import does not certify the separate curated-checkout GPU deployment acceptance test.

**Unavailable checkpoints:** P2-v2 saved the two best per-system encoders, not separate epoch-1/epoch-2 SapBERT snapshots. They cannot be reconstructed from the selected weights. They are not silently substituted with v1. Optimizer-resume files and per-DEV score caches are not model deliverables and are excluded.

The Qwen LoRA adapter requires its original base. PEFT loading uses the base plus adapter; it is not random initialization. The bundle includes that base, so a new network download is unnecessary. Portable adapter configuration names the public base model; the trained adapter tensor bytes are unchanged and hashed.

Selection metadata preserves both the original epoch-1 automatic choice and the explicit epoch-2 final override. The bundle manifest lists every file's SHA256. Hugging Face model cards explain original versus adapted and full weights versus adapter. The four model repositories are public and ungated; the clinical-data bundle stays restricted. No all-label model is claimed.
