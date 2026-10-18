# Revised-input Qwen LoRA post-training

`trainer.py` contains the two-epoch Qwen3-Reranker-0.6B recipe, seeded sampling and quiz construction. Each quiz includes one supplied gold code plus 15 candidate/structural negatives; candidate positions and type/Seen visibility are seeded. Candidate cards always include the applicable mention/context information; reference quiz payloads in the private dataset preserve exact packed-text inputs.

Train-split pools use original SapBERT, leave-group-out memory and self-only-abbreviation masking. Alias quizzes withhold their own name; held probe aliases are excluded. DEV is used for checkpoint diagnostics/selection, never for gradient updates. The bundle exports 10,502 quiz groups and 168,032 candidate judgments for each epoch, in exact deterministic order reconstructed using the immutable training runtime.

LoRA rank 16, alpha 32, dropout 0.05; lr 1e-4; effective 8 quiz groups; maximum 512 tokens; gradient clip 1; bf16 base; seed 13. Both epoch adapters and tokenizers are provided. The original automatic selector chose epoch 1 on its original DiagnosisCode criterion; the user explicitly chose epoch 2 for the final F01/F11 comparison. Both facts are preserved rather than rewriting history.

A LoRA adapter is not a standalone Qwen model. Load it on the original Qwen base model included in the private bundle, or the named public base. `reranker_lora/best` in the bundle is an explicit copy of epoch 2. No all-label retraining is claimed.
