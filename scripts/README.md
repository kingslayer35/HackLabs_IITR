# Stage commands

Activate the environment and restore the private assets first. All commands use `TERMMAP_CONFIG` or `configs/default.yaml`. This folder contains production stage entry points, not experiment recovery controllers.

| Stage | Entry points |
|---|---|
| Public resources | `python -m scripts.resources.download` (approved resources/models only) |
| Dictionaries and hierarchy | `python -m scripts.terminology.build`; `python -m scripts.terminology.enrich` |
| Final preprocessing | `python -m scripts.preprocessing.prepare_queries`; `python -m scripts.preprocessing.audit_rows` |
| Original/adapted retrieval | `python -m scripts.retrieval.build_indexes`; `python -m scripts.retrieval.confirm` |
| SapBERT P2-v2 | `python -m scripts.pretraining.restore_inputs`; `python -m scripts.pretraining.train`; `python -m scripts.pretraining.evaluate` |
| Qwen LoRA | `python -m scripts.posttraining.train`; `python -m scripts.posttraining.audit` |
| Transformers / hosted scores | `python -m scripts.reranking.score`; `python -m scripts.reranking.score_hosted` |
| Decision fitting | `python -m scripts.decision.fit --flows F00 F10 F01 F11` |
| Checkpoints and diagnostics | `python -m scripts.evaluation.checkpoints`; `python -m scripts.evaluation.diagnostics` |
| Verified prediction | `bash run.sh input.csv predictions.csv` |

GPU entry points run only inside the screen/Slurm procedure in [operations](../docs/OPERATIONS.md). Training commands use their dedicated configs; never overwrite the distributed selected checkpoints. Some measurement entry points require the corresponding audit inputs or prepared pools. Final results are supplied by the main evaluation rather than re-running them during packaging.
