# Code and evidence index

The canonical package layout is preserved so imports and relative artifact paths remain meaningful. No source modules were refactored during handoff.


## Runtime and tools

- [`__init__.py`](independent_agent/__init__.py)
- [`__main__.py`](independent_agent/__main__.py)
- [`common.py`](independent_agent/common.py)
- [`prepare.py`](independent_agent/prepare.py)
- [`agent.py`](independent_agent/agent.py)
- [`tools.py`](independent_agent/tools.py)

## Retrieval, recovery and preprocessing

- [`comparisons.py`](independent_agent/comparisons.py)
- [`recovery_tools.py`](independent_agent/recovery_tools.py)
- [`recovery_eval.py`](independent_agent/recovery_eval.py)
- [`selective_tools.py`](independent_agent/selective_tools.py)
- [`selective_eval.py`](independent_agent/selective_eval.py)
- [`preprocessing.py`](independent_agent/preprocessing.py)
- [`preprocessing_eval.py`](independent_agent/preprocessing_eval.py)
- [`dense_agent_tools.py`](independent_agent/dense_agent_tools.py)
- [`dense_agent_eval.py`](independent_agent/dense_agent_eval.py)
- [`dense_agent_finish.py`](independent_agent/dense_agent_finish.py)
- [`dense_agent_resume.py`](independent_agent/dense_agent_resume.py)
- [`dense_search_service.py`](independent_agent/dense_search_service.py)
- [`retrieval_cascade_eval.py`](independent_agent/retrieval_cascade_eval.py)

## Evaluation and analysis

- [`analyze_selection.py`](independent_agent/analyze_selection.py)
- [`selection_prompt_eval.py`](independent_agent/selection_prompt_eval.py)
- [`backend_ablation.py`](independent_agent/backend_ablation.py)
- [`backend_ablation_analysis.py`](independent_agent/backend_ablation_analysis.py)
- [`diverse500_prepare.py`](independent_agent/diverse500_prepare.py)
- [`diverse500_eval.py`](independent_agent/diverse500_eval.py)
- [`diverse500_analysis.py`](independent_agent/diverse500_analysis.py)
- [`agent_focus_audit.py`](independent_agent/agent_focus_audit.py)

## Selectors, reasoning and rescue

- [`selector500_eval.py`](independent_agent/selector500_eval.py)
- [`reasoning_eval.py`](independent_agent/reasoning_eval.py)
- [`terminal_rescue.py`](independent_agent/terminal_rescue.py)
- [`terminal_rescue_eval.py`](independent_agent/terminal_rescue_eval.py)

## Training-case memory

- [`context_memory.py`](independent_agent/context_memory.py)
- [`memory_eval.py`](independent_agent/memory_eval.py)
- [`semantic_memory.py`](independent_agent/semantic_memory.py)
- [`semantic_memory_eval.py`](independent_agent/semantic_memory_eval.py)

## Tests

- [test_agent.py](independent_agent/test_agent.py)
- [test_backend_ablation.py](independent_agent/test_backend_ablation.py)
- [test_comparisons.py](independent_agent/test_comparisons.py)
- [test_context_memory.py](independent_agent/test_context_memory.py)
- [test_dense_agent.py](independent_agent/test_dense_agent.py)
- [test_diverse500.py](independent_agent/test_diverse500.py)
- [test_memory_eval.py](independent_agent/test_memory_eval.py)
- [test_preprocessing.py](independent_agent/test_preprocessing.py)
- [test_reasoning.py](independent_agent/test_reasoning.py)
- [test_recovery.py](independent_agent/test_recovery.py)
- [test_selection_prompt.py](independent_agent/test_selection_prompt.py)
- [test_selective.py](independent_agent/test_selective.py)
- [test_selector500.py](independent_agent/test_selector500.py)
- [test_semantic_memory.py](independent_agent/test_semantic_memory.py)
- [test_terminal_rescue.py](independent_agent/test_terminal_rescue.py)

## Experiment reports

[SUMMARY.md](SUMMARY.md) gives the main results and conclusions. [FINAL_REPORT.md](FINAL_REPORT.md) consolidates all 19 original experiment reports and analyses, including negative results and limitations. Duplicate per-experiment Markdown files remain in the local research archive and are excluded from Git.

## Artifact groups

| Folder | Contents |
|---|---|
| `independent_agent/artifacts/agent_focus_audit_v1/` | Visibility inspection and literature/guidance snapshots |
| `independent_agent/artifacts/backend_ablation_v1/` | ICD-only and selective SNOMED |
| `independent_agent/artifacts/comparisons_v1/` | Matched 40-input models/tools |
| `independent_agent/artifacts/dense_agent_v1/` | Combined dense backend/paging agent |
| `independent_agent/artifacts/diverse500_v1/` | 500-input agent comparison and training split |
| `independent_agent/artifacts/gpu_retrieval_v1/` | Dense retrieval, cascades, preprocessing and remote snapshots |
| `independent_agent/artifacts/memory_context_v1/` | Lexical contextual/contrast memory |
| `independent_agent/artifacts/plateau_audit_v1/` | Cross-experiment residual/complementarity audit |
| `independent_agent/artifacts/reasoning100_v1/` | Off/high reasoning pilot |
| `independent_agent/artifacts/recovery_v1/` | 80-input recovery/comparison |
| `independent_agent/artifacts/runs/` | Original smoke experiment |
| `independent_agent/artifacts/selection_audit_v1/` | Baseline selection-error review |
| `independent_agent/artifacts/selection_prompt_v1/` | Example-equivalence prompt |
| `independent_agent/artifacts/selective_v1/` | Selective query recovery |
| `independent_agent/artifacts/selector500_v1/` | 500-input A/B/C selectors |
| `independent_agent/artifacts/semantic_memory_v1/` | E5 semantic index and matched pilot |
| `independent_agent/artifacts/terminal_rescue500_v1/` | Fallback-only guided finalization |

Root artifact files include the terminology SQLite database, original split manifest, leakage audit, filtered examples and dictionary statistics. Source snapshots and supplementary verifiers under artifact folders are historical evidence, separate from canonical runtime modules.
