# TermMap agentic solution — consolidated experiment and submission report

Prepared 18 October 2026. This subdirectory contains the independent agentic workstream alongside the team pipeline. The Git submission includes its source code, tests, summary and consolidated experiment record. Supplied input data, frozen artifacts, duplicate reports and original planning notes remain in the private local archive. This work does not replace the team pipeline.

## Executive result

The most extensive completed paired agent comparison used 500 diverse diagnosis inputs. Original text retrieval scored **67.4% ICD / 70.4% SNOMED**. Pretrained SapBERT for ICD with the original SNOMED retrieval scored **69.2% / 69.2%**. Its small headline gain was inconclusive. A subsequent fallback-only guided terminal rescue, evaluated as a controlled continuation of saved traces, reached **70.6% / 71.4%**, with both codes correct on **53.6%** of inputs. The rescue result is development replay evidence, not a fresh end-to-end or untouched confirmation result. The challenge's greater-than-95% target has not been achieved.

The deliverable is a working standalone search/inspection agent and an extensive, auditable experiment record. Negative and inconclusive results are included. No independent-agent weight training, full SNOMED ontology import, learned contextual reranker, or final integration into the team submission was completed. The latest proposed 60-case selector diagnostic remains a proposal, not a completed experiment.

## Architecture and implementation

The agent maps a supplied mention and surrounding chart text to one supplied code in each system. SQLite stores concepts, all supplied aliases, code-derived ICD structural links and filtered training examples. Bounded tools support search, inspection, permitted example lookup and ICD relatives. A validated finish must use dictionary-valid, previously visible codes. Deterministic fallbacks and every failed finalization remain in scoring.

Later modules add remote pretrained SapBERT search, deeper-page access, conservative query views, fixed-evidence selectors, reasoning ablations, terminal rescue, lexical contextual memory and CPU semantic memory. These are separate experimental modules; the baseline runtime was not silently replaced. The model was usually DeepSeek V4 Flash 0731; earlier experiments used DeepInfra and later ones CoreWeave through OpenRouter. Qwen3.5-9B was tested only in the early matched comparison. Providers, sample sizes and evidence settings must be read with each result.

The primary metric is exact code equality separately for ICD and SNOMED; the headline is their minimum. Both-correct is an additional paired metric. Dictionary recall@K is a candidate-retrieval measure and must not be described as final mapping accuracy.

## Main 500-input mapping results

| Configuration | ICD | SNOMED | Both correct | Interpretation |
|---|---:|---:|---:|---|
| Original text search/inspection | 337/500 (67.4%) | 352/500 (70.4%) | 266/500 (53.2%) | Full paired reference |
| SapBERT ICD + original SNOMED | 346/500 (69.2%) | 346/500 (69.2%) | 265/500 (53.0%) | Mixed result; not a proven overall upgrade |
| A: compact fixed-evidence selector | 327/500 (65.4%) | 348/500 (69.6%) | 244/500 (48.8%) | Not an agent replacement |
| B: A plus coding/task guide | 337/500 (67.4%) | 351/500 (70.2%) | 254/500 (50.8%) | Some improvement over A |
| C: B plus lexical candidate distinctions | 344/500 (68.8%) | 348/500 (69.6%) | 256/500 (51.2%) | ICD gain with SNOMED losses |
| Fallback-only B rescue after failed investigation | 353/500 (70.6%) | 357/500 (71.4%) | 268/500 (53.6%) | Controlled continuation; 465 accepted decisions preserved |

The rescue adds one terminal response on 35 budget-failure inputs, permitting six rather than five responses there. It corrects seven ICD and 13 SNOMED labels and regresses two SNOMED labels. Its new API cost was $0.02710027. This is not an equal-budget comparison or independent confirmation.

## Long-tail results on the 500-input development sample

| System / permitted training frequency | Inputs | Original | ICD variant | Variant plus rescue |
|---|---:|---:|---:|---:|
| ICD unseen | 114 | 64 | 67 | 68 |
| ICD rare (1–5 examples) | 261 | 182 | 188 | 191 |
| ICD common (6+) | 125 | 91 | 91 | 94 |
| SNOMED unseen | 241 | 141 | 138 | 145 |
| SNOMED rare | 229 | 184 | 182 | 185 |
| SNOMED common | 30 | 27 | 26 | 27 |

Unseen means absent from the permitted training-example store, not absent from the terminology or model pretraining. The sample covers 382 normalized context groups, 336 ICD codes, 431 SNOMED codes and 20 coarse domains. It is purposively diverse development evidence, not a population-prevalence estimate or untouched test. Training-only evidence contains 3,423 diagnosis-labelled rows; exact input/mention/context separation and near-duplicate quarantine were audited.

## 100-input selector pilots — compare within each block only

| Experiment / condition | ICD correct | SNOMED correct | Both correct |
|---|---:|---:|---:|
| Reasoning off, 8,192-token common cap | 60 | 73 | 45 |
| High reasoning, same evidence/cap | 58 | 63 | 45 |
| Mention-based lexical memory | 61 | 71 | 48 |
| Contextual lexical memory | 62 | 72 | 49 |
| Lexical contrast memory | 61 | 74 | 50 |
| Semantic experiment: canonical cached lexical control | 62 | 72 | 49 |
| Semantic case retrieval | 60 | 70 | 47 |

High reasoning truncated on 13 inputs and cost 2.63 times its matched off arm. It did not pass the expansion gate. Lexical-memory apparent small gains were materially confounded by repeated generation variation on identical requests. Semantic memory used canonical exact-request reuse to prevent that attribution error: 92 new calls and eight cached semantic responses, costing $0.05527407. No memory or high-reasoning 500-input inference expansion was started. These fixed-evidence pilots are not directly comparable with the full search agent's 500-input scores.

## Earlier experiments and retrieval results

The full reports below preserve all tables, settings, corrections/regressions and limitations. Main findings:

- **20-input smoke:** agent 70% / 70%; search-only 25% / 55%; both correct 55%.
- **40-input matched model/tools:** DeepSeek search/inspection 82.5% / 80%; Qwen3.5-9B search/inspection 57.5% / 67.5%. These small-sample results are not the later 500-input baseline.
- **80-input recovery:** baseline 65% / 71.25%; mandatory query recovery 63.75% / 67.5%; recovery plus comparison 57.5% / 67.5%. More retrieval did not yield a final accuracy gain.
- **80-input prompt and selective recovery:** example-equivalence instruction 62.5% / 68.75%; selective recovery 66.25% / 66.25%, neither establishing an overall upgrade.
- **80-input dense tools:** combined dense retrieval/paging 71.25% / 68.75%; ICD-only change 70% / 70%; selective deeper SNOMED review 65% / 66.25%. These motivated the larger paired comparison rather than immediate promotion.
- **500-input retrieval-only benchmark:** SapBERT recall@12 64.6% ICD / 77.8% SNOMED; exact-first cascade 63.6% / 80.6%. Cascade plus category children added no recall at the tested budgets. This retrieval sample is not identical to the later diverse agent sample.
- **Conservative preprocessing:** refined combined views retained 64.6% / 80.6% recall@12, with offsetting recoveries/losses. Rejected v1 and refined v2 are retained; v2 is explicitly development tuning.

## Combined inference and failure analysis

In the diverse agent comparison, ICD initial dictionary gold recall rose by 99 cases under SapBERT, while eventual gold exposure rose by only 18 and correct answers by nine. Later searches already repaired many original first-pass misses. Initial recall alone therefore overstated the end-to-end opportunity.

After rescue, residual errors divide into 77 ICD / 78 SNOMED visible-gold selection failures and 70 / 65 exposure misses. Perfect choice from those saved pools reaches only 86% / 87%. Gold-aware selection of any correct output across six saved configurations reaches only 76.2% / 79.2%, showing strongly shared failures. Confidence is heuristic, not calibrated. More reasoning, examples or generic instructions have not established a reliable fix.

Errors involve target focus, concept type, specificity, code completion, incomplete candidate coverage, noisy targets and annotation-policy ambiguity. Neither clinical correctness nor bad labels can be inferred solely from an exact-match disagreement. The source dictionaries include legacy ICD categories and multiple SNOMED semantic tags; full encounter billing rules and disorder-only filters would not match the task automatically.

The latest recommendation is to review clearly supported versus policy-dependent cases, then test a materially different selector on frozen evidence with correct controls. A learned description/context scorer and fold-safe hard negatives are stronger untested directions than another general memory mechanism. This should be coordinated with the team's already planned trained reranker. Graph traversal is a targeted later operation requiring verified relationships, not a demonstrated missing ingredient.

## Costs, reproducibility and scope

Major later new API spends: diverse500 $1.64027022; A/B/C selectors $1.17187671; reasoning pilot $0.18416906; terminal rescue $0.02710027; lexical memory $0.15156114; semantic memory $0.05527407. These are per-experiment increments; historical cached calls must not be billed again. They are not presented as a complete lifetime total. Individual reports retain earlier costs, timings and ledger details. Retrieval/preprocessing runs used local/Slurm compute rather than paid LLM calls.

All saved traces, labels, raw generations, manifests, usage ledgers, audits and source snapshots available locally are included. Earlier remote code/results are collected separately under `remote_workspace/` when available. No credentials, virtual environments, Python caches or SSH trust files are part of the package. Large pretrained model caches and duplicated remote databases are not needed to review the results and are excluded from that supplementary remote copy.

`tools/verify_package.py` validates a portable relative-path SHA-256 manifest. Historical experiment configurations intentionally retain their original absolute paths and source hashes. Their original verifier entry points may therefore need original-path resolution on a relocated machine; do not rewrite frozen manifests and claim an unchanged original experiment. `CODE_INDEX.md` maps the code and experiments. The original per-module README describes earlier stages and should be read with this final report.

The package preserves clinical data and evaluation golds for the authorized team's audit. Runtime evaluation labels remain separate from agent inputs. No new paid runs, training jobs, process termination, deployment or team-repository edits were performed during packaging.

## Complete original reports and analyses

The following sections include every independent result/analysis Markdown report, preserving its text. Earlier next-step statements are historical recommendations; the final plateau analysis supersedes them. Duplicate original files remain in the local archive and are excluded from Git; the source filenames below identify historical provenance.


---

## Appendix 1: SMOKE_RESULTS.md

Historical source: `independent_agent/SMOKE_RESULTS.md` (included below).

### TermMap independent agent smoke results

The first smoke experiment on 17 October 2026 achieved 70% exact-match accuracy on each coding system. Both predictions were correct on 55% of inputs. These results describe 20 held-out paired diagnosis inputs from 19 chunk groups; they do not estimate performance precisely enough to establish the challenge's 95% target.

| Metric | ICD-10-CM | SNOMED CT US |
| --- | --- | --- |
| Agent exact matches | 14/20 | 14/20 |
| Agent accuracy | 70% | 70% |
| Search-only accuracy on the same inputs | 25% | 55% |
| Gold available after initial search | 11/20 | 14/20 |
| Gold returned at some point during exploration | 18/20 | 17/20 |
| Valid final codes | 20/20 | 20/20 |
| Corrections relative to search-only | 9 | 4 |
| Regressions relative to search-only | 0 | 1 |
| Unseen-code exact matches | 3/3 | 4/9 |

The challenge's minimum-of-systems headline is 70%. Both codes were correct on 11 of 20 inputs. Nineteen inputs completed an agent selection; one exhausted the six-turn budget and used the deterministic search fallback. All 20 inputs, including that fallback, remain in the denominator.

#### Data separation

The training example database contains 3,466 valid diagnosis-labelled examples. Two invalid-code training rows were excluded. Test inputs were selected by a fixed seed and hash order, before model inference, from paired inputs in a reserved test split.

- Shared test/training input identifiers: 0.
- Shared normalized mentions: 0.
- Shared normalized context groups: 0.
- Two training groups were quarantined after an exhaustive comparison against the smoke chunks.
- Maximum remaining test/training three-word-shingle Jaccard similarity: 0.7353, below the 0.85 exclusion threshold.

This checks exact overlap and the stated near-duplicate criterion; it does not establish that all charts come from distinct patients, because the supplied file lacks patient/document identifiers. Evaluation golds and gold display names are kept out of the agent input and its accessible example store. The supplied dictionaries remain searchable in full.

#### Model and cost

The run used `deepseek/deepseek-v4-flash-0731`, pinned to DeepInfra, temperature 0, seed 13, reasoning disabled, two workers, at most six responses per input and 2,048 completion tokens per request.

- Successful model requests: 54, all served by DeepInfra, with 54 HTTP attempts recorded.
- Agent-issued investigation tool calls: 68, excluding automatic initial searches and training-example lookups.
- Input tokens: 257,450. Completion tokens: 12,303.
- API-reported smoke cost: **$0.01059978**, approximately 1.06 US cents.
- A separate infrastructure-only tool-call check cost $0.00004152; total observed API use including that check was **$0.01064130**. Credit-purchase fees are not included.
- Median per-input elapsed time: 8.518 seconds. P95: 13.895 seconds. Maximum: 23.771 seconds. These timings include local search and API calls for each input; they do not represent sequential total runtime.

The initial invocation stopped because two inputs had no initial lexical candidates. The startup path was repaired to permit LLM query reformulation with an empty pool. The 18 completed predictions were preserved byte-for-byte, and the two pre-inference failures were resumed. No test-label-driven prompt or model changes were made. Source hashes and preserved output hashes are recorded in the repair history.

#### What failed

Among the six ICD errors, two gold codes were never returned by any tool and four were returned but not selected. Among the six SNOMED errors, three gold codes were never returned and three were returned but not selected. One error in each system came from the same budget-exhausted input.

Observed failure patterns include excess specificity inferred from surrounding text, preference for a semantically related condition over the exact mentioned concept, inconsistent treatment of annotation conventions, and confusion between closely related SNOMED concepts. In the budget-exhausted case, exploration eventually returned both correct candidates but did not produce an accepted final selection within the allotted turns. The fallback then lost both answers.

The agent demonstrates candidate recovery on this small sample, but graph/search access does not by itself solve selection or annotation consistency. The next experiments should use development inputs and a fresh confirmation set. This smoke set is now consumed.

#### Verification and files

Seven checks passed, covering separation of the completed example store, dictionary and string-ID validation, rejection of gold fields, unknown-tool blocking, API-failure fallback and secret redaction, empty-search recovery, and relationship provenance.

Recompute the results offline from saved traces:

```bash
python3 -m independent_agent evaluate
```

The detailed outputs are in `artifacts/runs/smoke20_v1/`: `metrics.json`, `predictions.csv`, `errors_and_predictions.csv`, `config.json`, `repair_history.json`, and per-input `traces/`. Clinical text and generated traces are excluded from version control.


---

## Appendix 2: COMPARISON_RESULTS.md

Historical source: `independent_agent/COMPARISON_RESULTS.md` (included below).

### TermMap matched model and tool comparison

DeepSeek with search, inspection and permitted training-example lookup achieved the best headline score on 40 fresh paired inputs: 80%, with 33/40 ICD codes and 32/40 SNOMED codes correct. Additional relationship-tool access did not improve its aggregate accuracy. Qwen3.5-9B was the smaller comparator; its best headline result was 60%.

#### Results

Each condition used the same 40 inputs and the same initial candidates and training examples. The headline is the lower of the two system accuracies. All fallbacks remain in the denominator.

| Model | Configuration | ICD-10-CM | SNOMED CT | Headline | Both codes correct | Successful API responses | Recorded cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | One selection request | 26/40 (65%) | 25/40 (62.5%) | 62.5% | 20/40 (50%) | 40 | $0.00678 |
| DeepSeek V4 Flash 0731 | Search and inspection | 33/40 (82.5%) | 32/40 (80%) | **80%** | 30/40 (75%) | 92 | $0.01628 |
| DeepSeek V4 Flash 0731 | Search, inspection and relationships | 33/40 (82.5%) | 30/40 (75%) | 75% | 29/40 (72.5%) | 98 | $0.01711 |
| Qwen3.5-9B | One selection request | 23/40 (57.5%) | 23/40 (57.5%) | 57.5% | 17/40 (42.5%) | 40 | $0.01277 |
| Qwen3.5-9B | Search and inspection | 23/40 (57.5%) | 27/40 (67.5%) | 57.5% | 20/40 (50%) | 91 | $0.04171 |
| Qwen3.5-9B | Search, inspection and relationships | 24/40 (60%) | 28/40 (70%) | 60% | 21/40 (52.5%) | 104 | $0.05268 |

The literal search-only baseline scored 20% on ICD and 40% on SNOMED. It is a simple lexical baseline, not the team's adapted retrieval/reranking pipeline.

The completed six conditions consumed 465 successful model responses with $0.14734416 of reported usage. Including recorded usage from abandoned provider failures and two small transport checks, observed API charges were $0.148644648. Interrupted in-flight requests may have incurred usage not captured in a finished trace; a conservative additional $0.05 allowance was reserved. Observed usage plus that allowance is below $0.20 and within the $0.50 cap. Credit-purchase fees are excluded.

#### What the comparison supports

**Additional search and inspection helped DeepSeek on this sample.** Relative to its one-request selector, it corrected eight ICD predictions and regressed one, for a net seven additional correct answers. For SNOMED it corrected nine and regressed two, also a net seven. Each system gained 17.5 percentage points; both-correct accuracy gained 25 points. Initial-evidence selection alone did not explain all of the agent's performance.

**Relationship access showed no net gain for DeepSeek.** It corrected one ICD answer and regressed one; it corrected no SNOMED answers and regressed two. DeepSeek called the relationship tool six times on six inputs. One of those inputs improved on ICD relative to the search/inspection condition, but the aggregate result did not justify selecting the relationship-enabled configuration.

**Qwen did not demonstrate useful relationship-tool use.** Its relationship-enabled condition made zero relationship calls. Its small accuracy differences from the search/inspection condition therefore cannot be attributed to retrieved graph evidence. The extra tool description and policy text can change model behavior even when the tool is unused.

**The smaller model benefited less from tools.** Qwen's search/inspection condition had no net ICD improvement and a ten-point SNOMED gain over one-request selection. With the same search/inspection capability, DeepSeek scored 25 points higher on ICD and 12.5 points higher on SNOMED. This suggests a difference in how effectively the models use the available evidence, not merely a first-turn code-recall difference.

The best observed prototype choice is DeepSeek with search and inspection. Qwen3.5-9B remains a possible local training/distillation target, but these results do not support replacing the larger controller with it unchanged. Parameter count also did not predict hosted cost: Qwen's provider rates and response usage made its calls more expensive in this experiment. These API costs and latencies do not estimate local A6000 deployment costs or throughput.

#### Unseen codes and uncertainty

Among the fresh inputs, 14 ICD gold codes and 18 SNOMED gold codes had no examples in the permitted training store.

| Configuration | Unseen ICD correct | Unseen SNOMED correct |
| --- | --- | --- |
| DeepSeek one request | 7/14 | 10/18 |
| DeepSeek search and inspection | 10/14 | 15/18 |
| DeepSeek with relationships | 10/14 | 14/18 |
| Qwen one request | 6/14 | 9/18 |
| Qwen search and inspection | 7/14 | 11/18 |
| Qwen with relationships | 7/14 | 12/18 |

Paired bootstrap intervals over the 40 context groups put DeepSeek's search/inspection gain over its selector at approximately +5 to +32.5 points for ICD and +2.5 to +32.5 points for SNOMED. These are small-sample, descriptive intervals across exploratory comparisons. The relationship and smaller-model configuration differences are not stable enough to treat as established improvements. This is a development comparison, not a final benchmark or evidence of the challenge's 95% target.

#### Controls and limitations

- Forty fresh paired inputs came from 40 distinct normalized context groups in the reserved test partition. Original smoke groups, matching mentions and near-duplicate smoke chunks were excluded. No pair of fresh chunks exceeded the 0.85 shingle-Jaccard near-duplicate threshold.
- The separately filtered example store contains 3,464 training examples. Shared input IDs, normalized mentions and context groups between it and the fresh set are all zero. Maximum remaining test/training three-word-shingle Jaccard similarity is 0.8376.
- Every condition received an identical initial-evidence payload for each input. Parent, child, leaf and depth fields were stripped from initial cards and ordinary inspections. Only the relationship tool could expose explicit graph information.
- The graph condition provides derived ICD hierarchy and lexical SNOMED qualifier variants. It does not test a complete official SNOMED IS-A/attribute graph.
- The one-request selector could choose only codes already present in supplied candidates or examples. It was not an unconstrained test of memorized code generation. An offline diagnostic of raw first-finish proposals, before shortlist validation, gave DeepSeek 28/40 ICD and 25/40 SNOMED, and Qwen 23/40 ICD and 24/40 SNOMED. These are not the primary pipeline scores; their headline minima are unchanged.
- Three outputs in each one-request condition failed validation or parsing and used the deterministic fallback. One output in each Qwen agent condition also fell back. DeepSeek agent conditions completed all 40 selections. All 480 final system-level predictions are dictionary-valid, and no unresolved API transport error is counted as a model prediction in the final comparison.
- DeepSeek remained on DeepInfra. Qwen switched to Venice after upstream DeepInfra rate limiting. Per the user's instruction, all 16 successful completed runs were retained byte-for-byte, including one Qwen result per condition from DeepInfra. Provider/quantization differences prevent attributing every model gap solely to parameter count.
- Temperature was zero and reasoning disabled. DeepSeek requests used seed 13; new Qwen requests omitted the seed parameter because Venice does not support it. Data selection remained fixed at seed 13. Hosted inference is not guaranteed deterministic.
- The comparison protocol allows agents to finish immediately and reserves their last response for a finish-only request. Its starting cards and policy differ from the original 20-input smoke experiment, so the two experiments' percentages are not directly comparable.

#### Saved outputs

The comparison command and setup are in [README.md](README.md). Full outputs are in `artifacts/comparisons_v1/`: `results.json`, `all_predictions.csv`, `execution.json`, `leakage_audit.json`, `config.json`, `retained_runs.json`, and per-condition `runs/`. Initial evidence and per-response usage records are saved separately. The original 20-input smoke artifacts remain unchanged.

Thirteen checks passed before inference. After the run, source hashes, original-smoke hashes, retained-prediction hashes and the identical initial-evidence hashes across all 240 model/input runs were verified. Configuration selection used the aggregate results; no prompts were tuned against the fresh set during the run. The 40 comparison inputs are now consumed and should be excluded from future confirmation claims.


---

## Appendix 3: RECOVERY_RESULTS.md

Historical source: `independent_agent/RECOVERY_RESULTS.md` (included below).

### DeepSeek query recovery and candidate comparison

Three configurations were evaluated on the same 80 fresh paired inputs using DeepSeek V4 Flash 0731 through OpenRouter, pinned to DeepInfra. The headline score is the lower of ICD-10-CM and SNOMED accuracy. Every final output, including any model/workflow fallback, is included.

**Neither mandatory workflow improved end-to-end accuracy on this sample.** The existing search/inspection agent remains the preferred configuration: 65% headline accuracy, versus 63.75% with recovery and 57.5% with recovery plus comparison. Recovery made the correct code available on two additional inputs per system, but this was outweighed by incorrect final selections. The comparison workflow also had 12 fallback inputs (15%). These observations do not establish that selective query recovery, a trained set-wise selector, or a better scheduled comparison step would fail.

The earlier 80% result used a different set of 40 inputs. It is not the baseline for this experiment, and the fresh 65% baseline illustrates the uncertainty in the earlier small-sample estimate.

#### Accuracy

| Configuration | ICD-10-CM | SNOMED CT | Headline | Both codes correct |
| --- | --- | --- | --- | --- |
| Existing search and inspection | 52/80 (65%) | 57/80 (71.25%) | 65% | 42/80 (52.5%) |
| Query recovery | 51/80 (63.75%) | 54/80 (67.5%) | 63.75% | 40/80 (50%) |
| Query recovery + candidate comparison | 46/80 (57.5%) | 54/80 (67.5%) | 57.5% | 37/80 (46.25%) |

#### Paired changes

| Comparison | System | Corrections | Regressions | Net change | Paired bootstrap 95% interval |
| --- | --- | --- | --- | --- | --- |
| Query recovery versus Existing search and inspection | ICD10CM | 4 | 5 | -1.25 pp | -8.75 pp to +6.25 pp |
| Query recovery versus Existing search and inspection | SNOMEDCT_US | 2 | 5 | -3.75 pp | -10 pp to +2.5 pp |
| Query recovery versus Existing search and inspection | both | 4 | 6 | -2.5 pp | -10 pp to +5 pp |
| Query recovery + candidate comparison versus Query recovery | ICD10CM | 4 | 9 | -6.25 pp | -15 pp to +2.5 pp |
| Query recovery + candidate comparison versus Query recovery | SNOMEDCT_US | 3 | 3 | +0 pp | -6.25 pp to +6.25 pp |
| Query recovery + candidate comparison versus Query recovery | both | 3 | 6 | -3.75 pp | -11.25 pp to +3.75 pp |
| Query recovery + candidate comparison versus Existing search and inspection | ICD10CM | 3 | 9 | -7.5 pp | -16.25 pp to +0 pp |
| Query recovery + candidate comparison versus Existing search and inspection | SNOMEDCT_US | 3 | 6 | -3.75 pp | -11.25 pp to +3.75 pp |
| Query recovery + candidate comparison versus Existing search and inspection | both | 3 | 8 | -6.25 pp | -13.75 pp to +1.25 pp |

Intervals are descriptive, based on paired resampling of the 80 distinct context groups. Several related outcomes are compared without multiplicity correction. A small positive difference whose interval includes zero is not established evidence of an improvement.

A separate 10,000-resample paired bootstrap recomputed the minimum of the two system accuracies in each sample. Headline changes versus the baseline were -1.25 points for recovery (95% interval -8.75 to +5 points) and -7.5 points for recovery plus comparison (-15 to 0 points). These exploratory intervals do not support selecting either addition as an accuracy improvement. The calculations are saved in `workflow_diagnostics.json`.

#### Retrieval and final-selection gaps

| Configuration | System | Gold initially available | Gold ever retrieved | Never retrieved and wrong | Retrieved but final output wrong |
| --- | --- | --- | --- | --- | --- |
| Existing search and inspection | ICD10CM | 55/80 | 66/80 | 14 | 14 |
| Existing search and inspection | SNOMEDCT_US | 64/80 | 71/80 | 9 | 14 |
| Query recovery | ICD10CM | 55/80 | 68/80 | 12 | 17 |
| Query recovery | SNOMEDCT_US | 64/80 | 73/80 | 7 | 19 |
| Query recovery + candidate comparison | ICD10CM | 55/80 | 70/80 | 10 | 24 |
| Query recovery + candidate comparison | SNOMEDCT_US | 64/80 | 72/80 | 8 | 18 |

Initially available includes both initial dictionary candidates and initial training examples. Ever retrieved includes all evidence returned during the run. Retrieved-but-wrong counts final failures, including any workflow fallback; it is not a pure measure of clinical reasoning.

#### Calls, cost and latency

| Configuration | Successful model responses | Local investigation calls | Median latency | p95 latency | Recorded cost |
| --- | --- | --- | --- | --- | --- |
| Existing search and inspection | 187 | 206 | 8.27s | 18.56s | $0.03235 |
| Query recovery | 250 | 255 | 12.83s | 24.34s | $0.05669 |
| Query recovery + candidate comparison | 313 | 385 | 16.48s | 33.09s | $0.08100 |

Completed configurations used 750 successful model responses, with $0.17003868 recorded in their traces. The per-response usage ledger totals $0.17003868, including any recorded abandoned attempts. The additional conservative allowance for possibly billed failed requests is $0.00000000; it is a bound, not confirmed spending. The spending cap was $0.50. Credit-purchase fees are excluded.

Tool counts and statuses:

- **Existing search and inspection:** tools `{"find_training_examples": 3, "inspect_concepts": 132, "search_concepts": 71}`; statuses `{"model": 80}`.
- **Query recovery:** tools `{"find_training_examples": 4, "inspect_concepts": 144, "recover_queries": 82, "search_concepts": 25}`; statuses `{"model": 80}`.
- **Query recovery + candidate comparison:** tools `{"compare_candidates": 162, "find_training_examples": 11, "inspect_concepts": 89, "recover_queries": 82, "search_concepts": 41}`; statuses `{"fallback_budget_or_invalid": 12, "model": 68}`.

All 750 recorded HTTP attempts succeeded and used DeepInfra. The 12 comparison-workflow fallbacks were control/tool-compliance failures, not API transport outages: their last response requested investigation tools despite the finish-only schema and forced-finish request. Ten inputs still requested comparisons, one requested another search, and one requested recovery again. The harness rejected those unavailable calls and used the documented fallback. It did not grant extra turns or silently accept an uncompleted comparison.

Both recovery conditions also had one rejected supporting quote and corrected it. Four comparison calls supplied fewer than two distinct codes; these were rejected. The model successfully included selected codes in comparisons for all 68 accepted comparison-workflow finishes.

The new arms generated exactly identical first-recovery search names on only 24 of 80 inputs. Their differing instructions can affect recovery itself, so the combined arm's delta cannot be attributed exclusively to displaying candidate descriptions together.

#### Recommended next experiment

Keep the baseline as the current default. On development inputs, investigate selective recovery for inadequate candidate lists, retaining the original shortlist and checking for regressions. For comparison, reserve an explicit comparison stage before finalization rather than letting unconstrained search consume its response budget. Establish those decisions on development data before any new held-out confirmation. No additional paid evaluation was run after inspecting these outcomes.

#### What was built

- `recover_queries`: the controller supplies an interpretation, one standard name and up to two alternative names per coding system. The tool preserves the literal original mention and fuses searches over these names, returning at most 12 candidates per system. Quotes supporting generated qualifiers must occur verbatim in the input. Generated names remain hypotheses; quote presence does not prove clinical entailment.
- `compare_candidates`: side-by-side dictionary descriptions, up to eight stored aliases and literal description-word differences for two to six previously retrieved valid codes. No extra LLM call, generated medical definitions, relationship metadata or outside source is used by the tool.
- The baseline uses the previous search/inspection policy. Both new configurations must recover queries first. The comparison configuration must compare its final selected code against at least one other code when multiple valid candidates are available. Comparisons for both systems may be batched.

#### Controls and limits

- All three configurations receive identical initial evidence, the same filtered example database, model, provider, temperature zero, seed 13, disabled reasoning, maximum 2,048 completion tokens per request and maximum five model responses. Investigation calls can be batched. Relationships are disabled.
- Each new workflow has a different required minimum set of actions and prompt. The same response cap does not mean identical compute use. Recovery wording can differ between the two new configurations because their policy instructions differ; this measures full workflows rather than isolating comparison formatting under identical generated queries.
- Inputs are selected in seeded hash order from reserved-test paired inputs, one per normalized context group, excluding all prior 60 evaluated inputs, their groups, normalized mentions and near-duplicate chunks. No within-set chunk pair crosses the near-duplicate threshold.
- Agent inputs contain only input ID, mention and context. Gold labels remain separate; local tools expose only dictionaries and the filtered training store. No prompt changes were made from these fresh outcomes during the run.
- Matching mentions, groups and near-duplicate training chunks are excluded from the example store. The maximum remaining three-word-shingle Jaccard similarity is 0.75, below 0.85. There are 3464 allowed training examples.
- Nineteen checks passed before inference. Source, input, label, initial-evidence and example-store hashes, plus hashes of all protected prior evaluation artifacts, were checked after inference. Every arm used the expected identical initial-evidence payloads.
- Final dictionary validity, model/workflow fallbacks, unseen-code performance and paired changes are saved in the machine-readable report. Unresolved API/infrastructure errors prevent scoring rather than being silently treated as model errors.
- These 80 inputs are now consumed development evidence. They must not be used as independent confirmation for later prompt tuning. Compare configurations within this experiment; the earlier 40-case percentages used different inputs.

#### Saved outputs

Implementation: `recovery_tools.py` and `recovery_eval.py`; checks: `test_recovery.py`. Commands are in `README.md`. Full results are under `artifacts/recovery_v1/`: `results.json`, `all_predictions.csv`, `leakage_audit.json`, `config.json`, `execution.json`, frozen inputs/labels/evidence and the three `runs/` directories. Each trace records tool arguments, results, final selection, response usage and latency.


---

## Appendix 4: SELECTION_ERROR_ANALYSIS.md

Historical source: `independent_agent/SELECTION_ERROR_ANALYSIS.md` (included below).

### Baseline selection-error analysis

The strongest pattern to investigate is **transferring a similar training example's label without resolving differences in meaning, modifiers or annotation scope**. More inspection by itself is unlikely to address the whole problem: the agent explicitly inspected the gold code in 17 of the 28 selection errors. Several other disagreements depend on annotation conventions that cannot be replaced by one general instruction about specificity, negation or target-span scope.

This is an offline review of the saved search/inspection baseline on the existing 80 development inputs. No API calls, prompt edits, label repairs or new model evaluations were performed. The baseline remains the current agent. “Gold” means the supplied dataset label; an exact-match disagreement is not automatically an incorrect clinical interpretation.

#### Scope and verified counts

| Baseline outcome | ICD-10-CM | SNOMED CT |
| --- | --- | --- |
| Correct final code | 52/80 | 57/80 |
| Gold never retrieved, final code wrong | 14 | 9 |
| Gold retrieved, final code wrong | **14** | **14** |
| Selection errors with gold in initial evidence | 10 | 12 |
| Selection errors with gold explicitly inspected | 7 | 10 |
| Selection errors whose predicted code appeared in visible training examples | 7 | 9 |
| Selection errors whose gold code appeared in visible training examples | 5 | 1 |

The 28 system-level selection errors affect 23 distinct inputs. Every gold code in this review was visible **before the assistant response that submitted the accepted final answer**, rather than appearing only afterward or in an unread tool result generated alongside that answer. All baseline cases completed with model selections; no infrastructure or budget fallbacks explain these errors.

Twenty-two of the 28 gold codes were already in the initial candidates or examples. Seventeen were explicitly inspected. Six gold codes also appeared as labels in visible training examples; on three errors, both the gold and predicted codes had supporting example labels. This is evidence of a choice between competing interpretations, rather than simply too few candidates.

The examples are useful on many cases too: their labels align with 32 of the 52 correct ICD predictions and 21 of the 57 correct SNOMED predictions. Removing examples outright is not supported by this review. These are associations, not an experiment establishing what caused a prediction.

#### Recurring patterns

These are manually assigned primary patterns, with one assigned per error to make the counts add up. Concerns overlap, and a final explanation is evidence about the model's stated justification, not a reliable transcript of its internal reasoning.

| Primary review pattern | ICD | SNOMED | Total |
| --- | --- | --- | --- |
| Similar-example label transferred across a meaningful distinction | 5 | 5 | **10** |
| Specificity or context scope differs from the target label | 6 | 2 | **8** |
| Semantic type or target focus differs | 1 | 4 | **5** |
| Assertion/negation convention differs | 0 | 2 | **2** |
| Default interpretation differs | 1 | 1 | **2** |
| Encounter/code-completion choice differs | 1 | 0 | **1** |

##### Similar examples can override target distinctions

The predicted label occurs in visible examples in 16/28 selection errors; ten are assigned example transfer as their primary pattern. The distinction matters: presence in an example alone does not establish copying as the cause.

- **“Not sleeping at all.”** A training example for “not sleeping” maps to Sleeplessness/Insomnia. The model follows it, despite inspecting the gold **Cannot sleep at all**. The target's “at all” distinction disappears.
- **“Nasal tip ptosis.”** The model follows a training example for “ptosis of the nasal tip” into **Structural abnormality of tip of nose**. The target gold is **Drooping nasal tip**, also inspected. The training example's context itself says there is some ptosis of the nasal tip, so this is a genuine label-granularity tension rather than a clean linguistic error with an obvious universal fix.
- **“Low oxygen requirement.”** The note says the requirement is new, with none at baseline. The model follows a “need for supplemental oxygen” example into **Dependence on supplemental oxygen**. Gold is **Hypoxemia**, and a hypoxemia-labelled example was also supplied. The choice adds status semantics that deserve an explicit evidence check.
- **“Asthma overlap with exacerbation.”** A COPD/asthma-overlap example leads to a COPD exacerbation code; the target gold is asthma exacerbation. The agent's explanation replaces the target focus with the combined contextual diagnosis.

These cases motivate checking semantic equivalence before transferring an example's code, rather than treating lexical similarity as enough. They do not justify memorizing phrase-to-gold overrides from this development set.

##### Specificity mistakes point in both directions

- **“Generalized weakness” → Muscle weakness (generalized).** The model adds “muscle”; gold is the general Weakness code. The explanation describes the predicted candidate as the most specific match.
- **“Episodic hypotension.”** ICD gold is Other hypotension, while the model chooses unspecified hypotension. On the same input, SNOMED goes in the opposite direction: the model selects the more descriptive Hypotensive episode, while gold is broader Hypotension.
- **Spinal stenosis and facet arthropathy.** Surrounding L1–L2 or L2–L3 wording leads the model to lumbar ICD codes, while the targets use site-unspecified codes. The location is actually present in context; these are annotation-scope disagreements, not evidence that the model invented a location.

“Always choose the most specific code” and “always choose the broadest supported code” would each disagree with some of these golds. The relevant question is what meaning belongs to the labelled target and what the dataset expects context to supply.

##### Semantic tags and surrounding diagnoses can change the target

- **“Scab.”** The agent explicitly prefers **Scab of skin (finding)** over **Scab (morphologic abnormality)**; the latter is gold. The task does not require findings to outrank other tags.
- **“Lack of stride height.”** The agent selects **Gait abnormality**, following related gait examples, while gold is **Step height (observable entity)**.
- **“Eschar/necrosis” on a left lower-leg ulcer.** Here the model follows the highlighted words into Necrotic eschar, while gold selects the enclosing **Ulcer of left lower leg**. This creates the opposite pressure to the asthma example: a strict span-only policy would not reproduce this label.

The SNOMED diagnosis-labelled rows therefore need more than a finding/disorder filter, and target-span scope needs explicit annotation guidance.

##### Assertion and default conventions need a separate review

Two SNOMED selection errors point in opposite directions:

| Input context | Agent selected | Dataset gold |
| --- | --- | --- |
| “No conversational dyspnea” | Dyspnea | No breathlessness |
| “Pressure injury status:none” | No pressure injury | Pressure injury |

The dyspnea choice is supported by a real retrieved training example: **“no dyspnea or chest pain on exertion” → Dyspnea**. This means the current disagreement cannot be resolved by assuming that examples uniformly preserve, or uniformly ignore, assertion status.

Other policy-sensitive examples include tricuspid regurgitation (rheumatic-labelled gold versus a nonrheumatic-labelled prediction), MRSA (organism/infection labels versus detected/carrier labels), and AKI/CKD (acute-component gold versus an acute-on-chronic prediction supported by similar training examples). These deserve annotation/default review before being counted as straightforward prompt-fix opportunities. This audit does not establish official coding defaults or prove that any gold is erroneous.

##### Dictionary validity does not resolve code completion

For **right rotator-cuff injury**, gold is `S46.001A`; the model selects `S46.001`, explicitly saying the encounter is unspecified in the text. Both strings exist in the supplied dictionary. The model saw the gold initially but inspected only the stem. An encounter-variant check is worth considering, but one case does not justify always adding `A` or forcing every prediction to a leaf code.

#### Recommended next prompt experiment

Test **one final equivalence check before transferring an example's label**. Keep the current model, tools, retrieval, example store and response budget unchanged; do not add a comparison tool or a new required model turn.

Suggested focus:

> Before using a training example's code as the deciding evidence, check whether its mention and context express the same concept, modifiers and status as the target. State the decisive target evidence, including any meaningful detail your selected candidate adds or discards. A similar example is supporting evidence, not an override for a different target meaning.

This is a hypothesis to test, not an implemented prompt change or a projected accuracy gain. The existing task prompt already warns that examples are not guaranteed equivalents; the proposed experiment would make this one final selection check more explicit. No per-case gold mappings or hard-coded fixes should enter the prompt.

Track corrections and regressions over all 80 inputs, not only these 28 errors. In particular, watch correct predictions supported by examples, and the opposite-direction specificity and negation cases above. Keep annotation-policy-dependent cases separately identified so that an aggregate score change is not mistaken for a demonstrated improvement in clinical interpretation.

#### Practical limit of selection-only work

With the **existing retrieved-code sets held fixed**, an oracle that corrected every selection error would score 66/80 (82.5%) on ICD and 71/80 (88.75%) on SNOMED: an 82.5% headline minimum. This is a diagnostic ceiling for these saved traces, not an achievable forecast; changing an agent's choices can also change its subsequent searches. The remaining 14 ICD and 9 SNOMED retrieval misses would still require retrieval work.

#### Reproduction and case-level evidence

Run from the project root:

```bash
python3 -m independent_agent.analyze_selection
```

The manual review annotations are private experiment artifacts in `artifacts/selection_audit_v1/review_annotations.json`. The script regenerates `summary.json`, `cases.json`, `cases.csv` and `source_hashes.json`, checks agreement with the original metrics, checks pre-decision visibility of each gold, and verifies that its source artifacts did not change during the audit. It imports no inference client and makes no network requests.

`cases.csv` contains every reviewed error with gold/predicted codes and stored dictionary descriptions, candidate visibility, inspection status, example support, the model's final explanation, and a reviewer note/caution. `cases.json` adds aliases, visible examples and source-trace references. Stored dictionary names are not a certification of official preferred terms. The review categories are interpretive; the counts of outcomes and tool visibility are reconstructed from the logs.


---

## Appendix 5: SELECTION_PROMPT_RESULTS.md

Historical source: `independent_agent/SELECTION_PROMPT_RESULTS.md` (included below).

### Example-equivalence final-selection prompt experiment

One generic final-selection check was appended to the existing agent system prompt and evaluated on the same 80 development inputs. DeepSeek V4 Flash 0731 and DeepInfra were retained. All saved baseline traces were reused byte-for-byte; only the new condition incurred API calls.

**Keep the baseline; this prompt did not demonstrate an improvement.** Headline accuracy fell from 65% to 62.5%. It corrected one of the 28 previously reviewed selection errors, and recovered two previously missed gold codes through changed investigation, but regressed seven previously correct system-level predictions. All 80 runs completed with model selections and no tool-validation errors, so the observed decline is not explained by fallbacks or transport failures. The paired headline interval includes zero; this run provides no evidence for selecting the new prompt over the baseline.

#### Accuracy

| Configuration | ICD-10-CM | SNOMED CT | Headline minimum | Both codes correct |
| --- | --- | --- | --- | --- |
| Saved search/inspection baseline | 52/80 (65%) | 57/80 (71.25%) | 65% | 42/80 (52.5%) |
| Example-equivalence check | 50/80 (62.5%) | 55/80 (68.75%) | 62.5% | 41/80 (51.25%) |

#### Paired changes

| Outcome | Corrections | Regressions | Net change | Paired bootstrap 95% interval |
| --- | --- | --- | --- | --- |
| ICD10CM | 1 | 3 | -2.5 pp | -7.5 pp to +2.5 pp |
| SNOMEDCT_US | 2 | 4 | -2.5 pp | -8.75 pp to +3.75 pp |
| both | 2 | 3 | -1.25 pp | -6.25 pp to +3.75 pp |

Headline change: -2.5 pp; paired 10,000-resample bootstrap interval: -7.5 pp to +2.5 pp. These are exploratory development intervals, with no multiple-comparison correction. This experiment follows error analysis of the same cases and is not independent confirmation.

#### Previously reviewed errors and previously correct predictions

| System | Baseline group | Inputs | Correct under new prompt | Wrong under new prompt |
| --- | --- | --- | --- | --- |
| ICD10CM | correct | 52 | 49 | 3 |
| ICD10CM | selection_error | 14 | 0 | 14 |
| ICD10CM | retrieval_miss | 14 | 1 | 13 |
| SNOMEDCT_US | correct | 57 | 53 | 4 |
| SNOMEDCT_US | selection_error | 14 | 1 | 13 |
| SNOMEDCT_US | retrieval_miss | 9 | 1 | 8 |

The groups are defined by the saved baseline, not used to route or select runtime inputs. The prompt ran on all 80 cases. Corrections among retrieval misses can result from changed search behavior, so they should not be counted as proof of better selection on identical retrieved candidates.

| Reviewed primary error pattern | Baseline errors | Fixed by new prompt |
| --- | --- | --- |
| semantic_type_or_focus | 5 | 0 |
| specificity_context_scope | 8 | 0 |
| assertion_convention | 2 | 0 |
| example_overtransfer | 10 | 1 |
| coding_default_convention | 2 | 0 |
| code_completeness | 1 | 0 |

Review categories are manual interpretations and overlap conceptually; each error was assigned one primary category for counting. A fixed example-transfer case is not causal proof that the new instruction produced the correction.

The one reviewed selection correction was **“not sleeping at all”**: the model selected **Cannot sleep at all**, instead of following the shorter “not sleeping” example into generic Sleeplessness/Insomnia. This is consistent with the intended modifier check. The other 27 reviewed selection errors remained wrong. Overall corrections also included ICD mapping of “drooling” and SNOMED mapping of “mucusua”; those had been baseline retrieval misses, so they are not evidence of better selection from identical available candidates.

Previously correct outputs regressed for ICD degenerative disc disease, weight-bearing intolerance and pressure injury; and for SNOMED fall with head impact, rhonchi, spinal stenosis and left maxillary sinusitis. Case-level codes and final evidence statements are saved in `changed_predictions.csv`. These changes illustrate that a general instruction can shift specificity, status and target interpretation outside the cases it was intended to repair. This is an end-to-end prompt result, not proof that every individual regression was caused by a particular sentence in the new check.

| Previously correct code supported by a visible example | Baseline correct | Still correct | Regressed |
| --- | --- | --- | --- |
| ICD10CM | 32 | 30 | 2 |
| SNOMEDCT_US | 21 | 19 | 2 |

#### Retrieval and usage

| Configuration | Gold ever retrieved: ICD | Gold ever retrieved: SNOMED | Model responses | Median latency | p95 latency | Recorded usage |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 82.5% | 88.75% | 187 | 8.27s | 18.56s | $0.03234660 |
| example_equivalence | 81.25% | 91.25% | 187 | 9.77s | 25.15s | $0.03266994 |

New reported API usage: $0.03266994, within a $0.20 cap. Baseline usage above is historic and was not charged again. Conservative allowance for potentially billed failed requests: $0.00000000; it is not confirmed spending. Credit-purchase fees are excluded.

New-condition statuses: `{"model": 80}`. Tool-validation errors: `{}`.

#### Exact intervention and controls

The existing system task and search/inspection policy are retained verbatim. The only appended instruction is:

> Before using a retrieved training example's code as the deciding evidence for your final choice, check whether its mention and context express the same concept, modifiers and status as the target. Similar wording alone is insufficient. In the short final evidence statement, identify the decisive target evidence, including any meaningful detail your selected candidate adds or discards. A similar example is supporting evidence, not an override for a different target meaning. This final selection check requires no additional model response or tool call.

- All 80 inputs, initial candidates/examples, filtered example database and cached baseline traces are byte-identical to the prior experiment. Labels are separate; only input ID, mention and context enter the agent. No reviewed case IDs, code mappings or phrase-specific exceptions were added to the prompt.
- The tool schemas exactly match the baseline on every response: search, inspection and permitted training-example lookup; finish only on response five. No recovery, relationship or candidate-comparison tools, extra required phases, schema fields or model responses were added.
- Model/provider, temperature zero, seed 13, disabled reasoning, 2,048-token completion cap and five-response maximum match the baseline. Baseline inference was performed earlier; hosted nondeterminism remains a limitation.
- The retrieval implementation is unchanged, but the new prompt can change queries, tool use and candidate exposure. This is an end-to-end prompt comparison rather than a fixed-evidence selection replay.
- All 29 checks passed before inference. Initial evidence, data, example-store, source and retained-baseline hashes, plus protected prior evaluation/audit artifacts, are verified after inference. Model/workflow fallbacks remain in the denominator; transport failures prevent final scoring.
- The prompt was frozen before inference and was not revised from outcomes. The same reviewed development cases were intentionally reused. Any promising result requires further development validation; this run alone does not establish generalization.

#### Saved outputs

Implementation: `selection_prompt_eval.py`; checks: `test_selection_prompt.py`. Commands are in `README.md`. Full outputs are under `artifacts/selection_prompt_v1/`: `results.json`, `all_predictions.csv`, `changed_predictions.csv`, `selection_diagnostics.json`, `config.json`, `execution.json`, retained-file hashes, new usage ledger and baseline/new-condition traces.


---

## Appendix 6: SELECTIVE_RESULTS.md

Historical source: `independent_agent/SELECTIVE_RESULTS.md` (included below).

### Selective query recovery on the same 80 inputs

DeepSeek V4 Flash 0731 on DeepInfra was evaluated with optional, system-specific query recovery. All 80 baseline traces were reused byte-for-byte from the previous experiment. Only the selective configuration incurred new API calls. These are the same previously inspected development cases, as requested.

**Keep the existing agent.** The selective policy produced a small headline increase, from 65% to 66.25%, but lost four correct SNOMED answers overall. Its paired headline interval includes both gains and losses. Recovery was attempted twice on one input and succeeded once; it supplied no previously missing gold code. The observed changes do not demonstrate an accuracy benefit from the recovery tool.

#### Accuracy

| Configuration | ICD-10-CM | SNOMED CT | Headline minimum | Both codes correct |
| --- | --- | --- | --- | --- |
| Saved search/inspection baseline | 52/80 (65%) | 57/80 (71.25%) | 65% | 42/80 (52.5%) |
| Previous mandatory recovery (reference) | 51/80 (63.75%) | 54/80 (67.5%) | 63.75% | 40/80 (50%) |
| Selective recovery | 53/80 (66.25%) | 53/80 (66.25%) | 66.25% | 43/80 (53.75%) |

#### Paired changes versus the saved baseline

| Outcome | Corrections | Regressions | Net change | Paired bootstrap 95% interval |
| --- | --- | --- | --- | --- |
| ICD10CM | 3 | 2 | +1.25 pp | -3.75 pp to +7.5 pp |
| SNOMEDCT_US | 1 | 5 | -5 pp | -11.25 pp to +0 pp |
| both | 3 | 2 | +1.25 pp | -3.75 pp to +6.25 pp |

The headline changed by +1.25 pp; the paired 10,000-resample bootstrap interval is -7.5 pp to +6.25 pp. These are exploratory development results, not confirmation on untouched data. Multiple related comparisons are not corrected for multiplicity.

#### Activation and mechanism

Recovery succeeded on 1/80 inputs, with 1 successful system-specific calls. It was unused on 79 inputs. Reasons: `{"meaning_mismatch": 1}`. Systems: `{"ICD10CM": 1}`.

| System | Successful recovery calls | Gold absent before call | Gold newly retrieved by recovery | Newly retrieved gold selected |
| --- | --- | --- | --- | --- |
| ICD10CM | 1 | 1 | 0 | 0 |
| SNOMEDCT_US | 0 | 0 | 0 | 0 |

Gold absence and recovery are reconstructed offline from the initial evidence and chronological tool results. Gold labels do not participate in runtime triggering. A missing gold is a diagnostic, not proof that an inadequacy judgment was semantically correct.

The sole activation concerned ICD mapping of “drooling.” The tool's queries for sialorrhea, drooling and excessive salivation did not return the gold code. A later ordinary search for the remembered code `K11.7`, together with salivary-gland wording, verified the supplied-dictionary entry “Ptyalism” and produced a correct final answer. This is a useful retrieval-vocabulary failure example, but it is not a successful recovery of the gold code by the new tool. The first recovery attempt incorrectly claimed an empty search; validation rejected it and the controller retried with a meaning-mismatch reason.

| Recovery activation | Inputs | System | Baseline correct | Selective correct | Corrections | Regressions |
| --- | --- | --- | --- | --- | --- | --- |
| recovery_used | 1 | ICD10CM | 0 | 1 | 1 | 0 |
| recovery_used | 1 | SNOMEDCT_US | 1 | 1 | 0 | 0 |
| recovery_unused | 79 | ICD10CM | 52 | 52 | 2 | 2 |
| recovery_unused | 79 | SNOMEDCT_US | 56 | 52 | 1 | 5 |

These subgroups are selected by the new policy, so their scores are descriptive. Changes on cases where recovery was unused cannot be attributed to newly retrieved recovery evidence: the extra policy/tool description and hosted inference variation can change ordinary search and selection.

#### Retrieval, cost and latency

| Configuration | Ever-retrieved gold ICD | Ever-retrieved gold SNOMED | Model responses | Median latency | p95 latency | Recorded usage |
| --- | --- | --- | --- | --- | --- | --- |
| Saved baseline | 82.5% | 88.75% | 187 | 8.27s | 18.56s | $0.03234660 |
| Selective recovery | 82.5% | 88.75% | 191 | 7.72s | 17.30s | $0.03490968 |

New recorded API usage was $0.03490968. Baseline usage in the table is historic and was not charged again. The new spending cap was $0.25; possibly billed failed-request allowance: $0.00000000. Credit-purchase fees are excluded. All inference used the pinned DeepInfra provider.

Selective statuses: `{"model": 80}`. Tool-validation errors: `{"icd10cm_code has not been retrieved; confirm it using a search or inspection tool": 1, "no_candidates requires an observed empty search for this system; use a meaning-based reason when results exist": 1}`. Tool calls: `{"find_training_examples": 6, "inspect_concepts": 130, "recover_queries": 2, "search_concepts": 72}`.

#### Implementation and controls

- The existing search/inspection task policy is extended with an optional recovery tool. The controller must explain an empty search, meaning mismatch or unresolved abbreviation. Mere ambiguity between plausible specificities is not an allowed reason. Empty-search assertions are verified against search history; the other two reasons remain model judgments, not a calibrated router.
- Recovery searches only the requested system. It preserves the original mention and all previously available candidates, deduplicates generated names after normalization and returns at most 12 new candidates. Supporting detail quotes must occur in the visible input; text presence does not prove entailment.
- At most one successful recovery per system is allowed, only in responses 1–3. Response 4 allows ordinary investigation; response 5 is finish-only. Immediate finish is valid. No mandatory recovery, candidate comparison or relationship tools are used.
- Model, provider, temperature zero, seed 13, disabled reasoning, 2,048 completion-token cap and five-response maximum match the saved baseline. Baseline inference was performed earlier rather than simultaneously; hosted inference is not perfectly deterministic.
- The 80 inputs, labels, 3,464-example filtered store, cached initial evidence and saved baseline traces are byte-identical to the preceding experiment. The previous mention/context/near-duplicate separation audit is retained. Only input ID, mention and context enter inference; neither gold labels nor baseline correctness are used to route cases.
- All 25 checks passed before inference. Source/config/input/example/initial-evidence hashes and all prior evaluation artifacts were protected and checked after inference. Final dictionary validity and workflow failures remain in the denominator.
- This experiment was designed after reviewing results on these same 80 cases. No prompt or trigger was changed midway from new outcomes, and no case was dropped or successful run retried. Any gain is a development result. This tests one selective policy together with per-system recovery, normalized query deduplication and an early deadline; it does not isolate selectivity from those implementation choices.

#### Saved outputs

Implementation: `selective_tools.py`, `selective_eval.py`; checks: `test_selective.py`. Commands are in `README.md`. `artifacts/selective_v1/` contains `results.json`, `all_predictions.csv`, `activation_diagnostics.json`, `config.json`, `execution.json`, `retained_inputs_and_baseline.json`, the separate new usage ledger and saved traces for both configurations.


---

## Appendix 7: RETRIEVAL_RECALL_RESULTS.md

Historical source: `independent_agent/RETRIEVAL_RECALL_RESULTS.md` (included below).

### SapBERT retrieval experiment

The unchanged 80 development inputs show a substantial ICD-10-CM candidate-recall improvement with pretrained SapBERT. Equal-weight hybrid retrieval performs best for SNOMED at 12 candidates. The existing agent is unchanged; these are dictionary retrieval results, not end-to-end agent accuracy.

| Retriever | ICD recall@12 | SNOMED recall@12 | ICD recall@30 | SNOMED recall@30 |
|---|---:|---:|---:|---:|
| Original lexical | 40/80 (50%) | 59/80 (73.75%) | 42/80 (52.5%) | 63/80 (78.75%) |
| SapBERT | 55/80 (68.75%) | 62/80 (77.5%) | 61/80 (76.25%) | 68/80 (85%) |
| Lexical + SapBERT RRF | 48/80 (60%) | 64/80 (80%) | 57/80 (71.25%) | 68/80 (85%) |

At 12 candidates, SapBERT recovers 15 ICD cases and loses none relative to lexical. For SNOMED it recovers eight and loses five; hybrid recovers six and loses one. Thus replacing lexical everywhere with dense retrieval would discard useful SNOMED candidates. Hybrid also dilutes the larger ICD improvement from dense retrieval alone.

#### Protocol

- Same frozen 80 inputs and labels as previous experiments; this is a repeatedly inspected development benchmark, not an independent generalization estimate.
- Original entity text only: no surrounding context, recovered queries, training demonstrations, fine-tuning, agent calls or paid LLM calls. Labels are used only for scoring and verifying dictionary coverage.
- Full existing dictionary: 159,306 ICD aliases across 98,222 codes and 984,585 SNOMED aliases across 382,124 codes.
- Pretrained `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`. CLS before pooler, L2 normalization, maximum 128 tokens, float16 encoding and float32 exact inner-product search with TF32 disabled.
- Code score is the highest matching alias score. Results are deduplicated by code before calculating recall.
- Equal-weight reciprocal-rank fusion uses constant 60, lexical top 30 and dense top 100; final output limits are identical across arms. This tests that particular fusion recipe, not an optimal hybrid.
- All 160 lexical top-12 lists exactly match prior cached results. Dictionary, input and label hashes match local sources. All 24 reported recall counts were independently recomputed from saved rankings.
- Cached alias embeddings are reusable. The initial encoding and evaluation took 188 seconds inside a 3-minute-15-second Slurm job, including lexical search and model loading. This does not measure production request latency.

#### Execution and artifacts

Slurm job `2700` completed with exit code 0 on an NVIDIA RTX A6000. All remote project files, environment, caches, model weights and outputs are under `/home/shivank_g/projects/pratham/termmap_adesh`. GPU computation ran only inside Slurm. Shell commands used the persistent SSH terminal; transfers reused its control socket.

Remote entry points: `code/benchmark_recall.py` and `code/recall.sbatch`. Results: `results/results.json`, `results/config.json`, `results/rankings.json`, detailed matched aliases and `results/per_input_recall.csv`. Local copies and the exact submitted source are under `independent_agent/artifacts/gpu_retrieval_v1/`.

#### Next experiment

Keep the current agent as the baseline. Evaluate an alternate search backend using dense ICD candidates and hybrid SNOMED candidates, while preserving training-example lookup, inspection, prompts and call budgets. That backend choice is selected on these 80 examples and must be reported as such. It still needs an end-to-end comparison: higher retrieval recall does not guarantee higher agent accuracy. No agent evaluation or backend replacement was performed in this experiment.


---

## Appendix 8: RETRIEVAL_CASCADE_500_RESULTS.md

Historical source: `independent_agent/RETRIEVAL_CASCADE_500_RESULTS.md` (included below).

### Exact-first retrieval cascade: 500-input evaluation

The fixed cascade improves top-12 code recall over the previous RRF recipe on both systems. SapBERT alone remains best for ICD-10-CM at this candidate budget; the cascade performs best for SNOMED. Adding category children does not improve recall@12 or recall@30. The existing agent remains unchanged, and no agent evaluation was run.

#### Results

Each cell is correct-code presence among the first K distinct candidate codes, with all 500 inputs in the denominator. These are candidate-recall measurements, not end-to-end mapping accuracy.

| Method | ICD recall@12 | SNOMED recall@12 | ICD recall@30 | SNOMED recall@30 |
|---|---:|---:|---:|---:|
| Existing lexical | 218/500 (43.6%) | 314/500 (62.8%) | 250/500 (50.0%) | 345/500 (69.0%) |
| Plain BM25 | 226/500 (45.2%) | 320/500 (64.0%) | 255/500 (51.0%) | 346/500 (69.2%) |
| SapBERT | **323/500 (64.6%)** | 389/500 (77.8%) | 353/500 (70.6%) | 412/500 (82.4%) |
| Existing RRF | 299/500 (59.8%) | 387/500 (77.4%) | 350/500 (70.0%) | 421/500 (84.2%) |
| Exact-first cascade | 318/500 (63.6%) | **403/500 (80.6%)** | **354/500 (70.8%)** | **423/500 (84.6%)** |
| Cascade plus category children | 318/500 (63.6%) | 403/500 (80.6%) | 354/500 (70.8%) | 423/500 (84.6%) |

At 12 candidates, the cascade recovers 31 ICD cases missed by RRF and loses 12, a net gain of 19 inputs or 3.8 percentage points. For SNOMED it recovers 29 and loses 13, a net gain of 16 inputs or 3.2 points.

The improvement is specific to candidate coverage. Cascade recall@1 is 21.8% ICD and 44.8% SNOMED, below RRF's 26.6% and 49.4%, and below SapBERT's 33.6% and 56.0%. Pinning exact and BM25 matches first can expose more correct codes within a list while putting an incorrect code at its head. This cascade should not automatically select its first candidate as the answer.

Category expansion triggered on 11 ICD inputs. It changes ICD recall@5 from 246/500 to 250/500, but leaves recall@12 and recall@30 unchanged. SNOMED is unchanged because no official SNOMED parent-child graph is available in the supplied dictionary. The present top-12 results do not justify adding this expansion to the default backend.

#### Fixed retrieval rules

All rules and quotas were fixed before scoring this set. Each system uses the same rules.

1. Put direct code matches first, followed by normalized exact alias matches. Preserve all matching codes up to the output budget; do not make an automatic final selection.
2. Add up to four additional distinct codes from BM25.
3. Fill remaining positions with SapBERT candidates, followed by any remaining BM25 candidates if needed.
4. Deduplicate by code and truncate to the output limit.

The category variant inserts up to two direct children of non-leaf exact matches after the exact matches, before the BM25 quota. Children must appear in dense top 100 and are ordered by dense rank. ICD relationships are the existing derived prefix hierarchy, not official tabular rules.

Plain BM25 uses the existing Porter-tokenized SQLite FTS5 index, stopword removal and up to 14 distinct normalized query words combined with OR. It considers the best 350 alias rows and deduplicates them by code in BM25 order. It omits the existing lexical retriever's separate AND pass, token-overlap bonus and exact-match boosts. Thus this experiment evaluates that concrete BM25 implementation, not every BM25 configuration.

The RRF baseline is unchanged: equal weights, reciprocal-rank constant 60, lexical top 30 fused with dense top 100. SapBERT uses the same pretrained model revision, CLS representation, normalization, maximum length 128 and cached dictionary embeddings as the earlier 80-input experiment. Dense retrieval uses exact float32 inner products with TF32 disabled and scores each code by its best alias. Candidate limits are identical when comparing recall at a given K.

#### Exact matching audit

| System | No exact alias match | Unique exact match | Unique match agrees with gold | Ambiguous exact match |
|---|---:|---:|---:|---:|
| ICD-10-CM | 464 | 36 | 25/36 (69.4%) | 0 |
| SNOMED | 381 | 116 | 111/116 (95.7%) | 3 |

All 11 ICD disagreements are exact category names whose expected label is more specific. For example, `heart failure` matches I50 while the gold is I50.9, and `shingles` matches B02 while the gold is B02.9. SNOMED also has five unique exact-match disagreements, including `difficulty hearing`, `acid reflux` and `thrush`. Agreement with this dataset's labels is being measured; these cases have not undergone independent clinical adjudication.

#### Sample and verification

- 500 paired DiagnosisCode inputs, selected by seeded SHA256 ordering from the existing reserved dev/test groups: 298 dev and 202 test. Eligible pool: 567 after excluding the prior 80 input IDs. No filtering by retrieval outcome or dictionary coverage.
- No input-ID or normalized-mention overlap with the prior 80. There are 499 distinct normalized mentions and 381 context groups within the new set. Seventeen inputs share normalized nonempty chart context with the prior 80, so the set is not completely independent at document level.
- Excluding those 17 inputs preserves the direction of the cascade-versus-RRF result: on the remaining 483, ICD recall@12 is 306/483 versus 287/483; SNOMED is 389/483 versus 374/483.
- Both gold codes exist in the supplied dictionary for all 500 inputs. All inputs remain in the denominator.
- Only the entity string is queried. No context queries, query recovery, demonstrations, training, agent calls or paid LLM calls. Source labels are used for scoring, not candidate construction.
- Rankings were saved before scoring. All 48 recall counts and all paired recovery/loss counts were independently recomputed from saved rankings. Input, label and submitted script hashes match the local frozen sources.
- This is one prespecified comparison on a larger development sample. Choosing a backend from these results is development tuning; no statistical significance or unseen deployment accuracy claim is made.

#### Execution and artifacts

Slurm job `2713` completed successfully with exit code 0 in 7 minutes 22 seconds. The scheduler assigned an NVIDIA RTX PRO 5000 Blackwell; GPU work ran only inside Slurm. The full dictionary embeddings were reused without re-encoding. Shell commands used the persistent SSH terminal, with transfers reusing its control socket.

All remote files are under `/home/shivank_g/projects/pratham/termmap_adesh`:

- Source: `code/retrieval_cascade_eval.py`; launch: `code/recall500.sbatch`.
- Frozen sample: `data/recall500_v1/{inputs,labels,selection_manifest}.json`.
- Outputs: `results/recall500_v1/`, including configuration, rankings, matched aliases, category expansions, per-input CSV, summary metrics and independent verification.
- Logs: `logs/recall500-2713.out` and `.err`.

Local implementation: `independent_agent/retrieval_cascade_eval.py`. Frozen upload and downloaded results: `independent_agent/artifacts/gpu_retrieval_v1/recall500_upload/` and `results/recall500_v1/`.

At a 12-candidate budget, the next agent-backend comparison should use SapBERT for ICD and the exact-first cascade for SNOMED, preserving the existing agent as its baseline. This backend choice is informed by these development results. Higher candidate recall still requires end-to-end agent validation; no backend replacement or paid evaluation was performed here.


---

## Appendix 9: PREPROCESSING_RESULTS.md

Historical source: `independent_agent/PREPROCESSING_RESULTS.md` (included below).

### Conservative preprocessing: 500-input ablation

The core query-view preprocessing described in `TermMap_Final_Plan.md`, Stage 1, is implemented independently of the team's pipeline. The refined combined pass does not improve top-12 candidate recall on the existing 500-input development benchmark. The existing agent and retrieval backend remain unchanged.

#### Implemented

- Keep the original mention and add optional views; never overwrite the source.
- Clean Unicode and quotes, HTML, whitespace, `header_<number>:` scaffolding and narrow HCC/POA artefacts.
- Mine abbreviation expansions from dictionary strings of the form `ABBR - expansion` and `expansion (ABBR)`. Retain up to three alternatives, ordered by dictionary occurrence counts, with no eval-label or training-label priors. Add `pna` and `doe` alternatives for the coverage gaps explicitly identified in the team plan.
- Require expansion initials to match the abbreviation, or a letter subsequence for single-word abbreviations. Block ordinary dictionary words and Roman numerals from expansion.
- Add cautious spelling/split views only for out-of-vocabulary alphabetic tokens of at least five characters. Permit a unique two-word split, missing-letter insertion, adjacent transposition or accidental repeated-letter removal. Avoid arbitrary substitutions and deletions. Protect numbers, known abbreviations, negation, laterality, type/stage and clinical qualifiers. Avoid repairing possessive names.
- Save a raw +/-300 character context window with mention markers and diagnostic cue flags. These are diagnostics only: they do not infer assertion scope and are not used in this retrieval experiment.

This is a conservative subset of the team's Stage 1, not its complete implementation. External abbreviation inventories, train-derived abbreviation priors, edit-distance-two correction, sentence/section-aware context windows and literal-context-code candidate injection are not included. The context remains separate from SapBERT's mention query.

#### Evaluation

Same 500 frozen inputs as the cascade experiment. No additional sample selection, training, agent calls or paid LLM calls. The fixed control is SapBERT for ICD and the exact-first cascade for SNOMED. Raw-only rankings exactly reproduce the prior 500-input run.

Each arm retains the raw query. `Abbreviation` and `repair` also include the clean view. Per-query backend rankings are fused with equal-weight RRF, constant 60; identical ranking lists vote once, and raw order breaks score ties. Output limits remain fixed. Consequently the result measures query views with this particular fusion rule, not preprocessing quality independently of candidate allocation.

| Refined v2 arm | ICD recall@12 | SNOMED recall@12 | ICD recall@30 | SNOMED recall@30 |
|---|---:|---:|---:|---:|
| Raw control | 323/500 (64.6%) | 403/500 (80.6%) | 353/500 (70.6%) | 423/500 (84.6%) |
| Clean views | 323/500 (64.6%) | 403/500 (80.6%) | 353/500 (70.6%) | 422/500 (84.4%) |
| Abbreviation views | 322/500 (64.4%) | 402/500 (80.4%) | 352/500 (70.4%) | 424/500 (84.8%) |
| Repair views | 323/500 (64.6%) | 404/500 (80.8%) | 354/500 (70.8%) | 423/500 (84.6%) |
| All views | 323/500 (64.6%) | 403/500 (80.6%) | 353/500 (70.6%) | 425/500 (85.0%) |

At 12 candidates, all views recover three cases and lose three cases in each system. Repair alone recovers one SNOMED case and loses none. These are small changes on a repeatedly inspected development sample; no generalization or significance claim is made. All views change top-1 counts from 168 to 167 ICD and 224 to 228 SNOMED, so the effect depends on rank budget.

Only 12 inputs produce a different clean string, 51 produce abbreviation views and 11 produce repair views. The refined pass needs 90 additional distinct dense and BM25 queries, with cached dictionary embeddings reused. Useful repair alternatives include `cardiomyopahty` → `cardiomyopathy`, `prexisting` → `preexisting`, and `withoutgangrene` → `without gangrene`.

#### Rejected first pass and refinement

The original v1 pass allowed unfiltered mined abbreviation senses and unique one-edit substitutions. Reviewing its generated views exposed false transformations: `heal` expanded into a procedure name, Roman `ii` into a procedure, and `weaker` into `weaver`. Raw text was always retained, but those alternatives were unsuitable.

V1 combined recall@12 was 321/500 ICD and 402/500 SNOMED, below the raw control. We rejected v1 and tightened the guards listed above. V2 was therefore developed after inspecting v1 views on these same 500 examples. Both runs, sources and rankings are retained; v2 is explicitly development tuning, not a fresh held-out evaluation.

#### Validation and artifacts

Six preprocessing tests cover raw preservation, formatting, ambiguous senses, Roman numerals, ordinary-word collisions, false spelling changes, real repairs and context marking. They pass with `python3 -m unittest independent_agent.test_preprocessing`.

For each run, all 40 recall figures and all paired recovery/loss figures were independently recomputed from saved rankings. Frozen sample and submitted source hashes match. The original sample's document-overlap limitations still apply; see `RETRIEVAL_CASCADE_500_RESULTS.md`.

Slurm jobs 2714 (v1) and 2715 (v2) completed with exit code 0, in 1 minute 22 seconds and 58 seconds respectively. GPU operations ran only inside Slurm. The persistent SSH terminal/control socket was reused. No model training or API spending occurred.

Local source: `independent_agent/preprocessing.py`, `preprocessing_eval.py`, and `test_preprocessing.py`. Local frozen source snapshots and results are under `independent_agent/artifacts/gpu_retrieval_v1/`.

All remote project files remain within `/home/shivank_g/projects/pratham/termmap_adesh`. Refined source is in `code/`; v1 source is preserved under `code/history/preprocessing500_v1/`. Detailed outputs are in `results/preprocessing500_v1/` and `results/preprocessing500_v2/`, including `query_views.json`, the mined abbreviation lexicon, per-query/per-input rankings, configuration, metrics and independent verification.

Keep the preprocessing module available, but do not enable every expanded view in the default retriever based on these numbers. The next useful analysis is abbreviation-sense and candidate-allocation errors, with context resolving ambiguous senses later. This experiment does not establish that broader preprocessing or the team's complete pipeline would fail to help.


---

## Appendix 10: DENSE_AGENT_RESULTS.md

Historical source: `independent_agent/DENSE_AGENT_RESULTS.md` (included below).

### Dense retrieval and deeper-page tools: agent experiment

The new backend raises the headline minimum accuracy from 65% to 68.75% on the existing 80-input development benchmark. ICD improves, while SNOMED regresses. This is a mixed result, not evidence for replacing the original agent wholesale. The original agent and prior evaluations remain unchanged.

#### Accuracy and retrieval exposure

| Measurement | Original ICD | New ICD | Original SNOMED | New SNOMED |
|---|---:|---:|---:|---:|
| Gold in initial dictionary top12 | 40/80 (50%) | 55/80 (68.75%) | 59/80 (73.75%) | 64/80 (80%) |
| Gold in initial evidence including examples | 55/80 (68.75%) | 64/80 (80%) | 64/80 (80%) | 67/80 (83.75%) |
| Gold visible before final decision | 66/80 (82.5%) | 72/80 (90%) | 71/80 (88.75%) | 71/80 (88.75%) |
| Final correct code | 52/80 (65%) | **57/80 (71.25%)** | **57/80 (71.25%)** | 55/80 (68.75%) |
| Wrong and gold never visible | 14 | 8 | 9 | 9 |
| Wrong despite visible gold | 14 | 15 | 14 | 16 |

Headline minimum: 65% → 68.75%. Both codes correct: 42/80 → 45/80. ICD corrects seven cases and regresses two; SNOMED corrects two and regresses four. Descriptive paired bootstrap intervals cross zero for both system accuracy differences. This reused development set and mixed-provider comparison do not establish a generalization improvement.

All 160 new system predictions are dictionary-valid. There are 79 accepted model finishes and one deterministic fallback after the five-response budget: on that case CoreWeave requested disabled investigation tools in the finish-only phase. The fallback stays in all denominators. There are no unresolved transport/infrastructure-error runs in the final score.

#### What was changed

- Same frozen 80 unlabelled inputs, context payload, filtered training-example store and training examples as the existing baseline. All 80 completed baseline runs were retained without additional baseline API requests.
- Same DeepSeek V4 Flash model ID, original task/policy prompt, seed13, temperature0, reasoning disabled, max2048 completion tokens and maximum5 successful model responses. Immediate finish remains permitted.
- ICD search uses pretrained SapBERT max-alias-score retrieval. SNOMED uses the fixed exact-first cascade: exact matches, four additional BM25 codes, then dense candidates.
- Each search retains a hidden pool of up to100 distinct codes. Initial output is12. Existing explicit search limits up to30 are preserved. The new `get_more_candidates` tool returns at most12 candidates from a prior pool using `search_id` and offset, without another retrieval search.
- The original query-level reciprocal-rank fusion rule is preserved for multiple queries: constant20, ranks starting0. No graph tools, preprocessing expansions, mandatory recovery or comparison phase are enabled.
- Hidden candidates are not marked visible or eligible for final selection. A new code must be returned before the model response that selects it; a search and finish emitted in the same response cannot manufacture pre-decision visibility.

The intervention combines a backend change with paging availability and revised search-tool descriptions. Their aggregate effects are not isolated.

#### Did the agent use the deeper pool?

Five paging calls were made on four inputs. All were ICD searches. One ICD gold was first exposed through paging, but the model still selected another code. Paging therefore demonstrated candidate recovery, but no direct final-answer correction on that first-exposure case. The agent did not page through SNOMED results.

Of the nine remaining SNOMED retrieval misses, five gold codes were already in the hidden initial100 pool but were never read. The other four were absent from that initial pool. All eight remaining ICD retrieval misses were absent from their initial100 pools. This separates two next steps: SNOMED needs better decisions about when to read deeper; ICD's remaining misses require improved query coverage or additional aliases, rather than simply paging the original query.

The initial100 pools contain 63 ICD and 73 SNOMED golds overall. These counts are not ceilings for the whole agent: training examples, reformulated searches and inspections expose additional codes.

There are now31 system-level selection errors versus17 retrieval misses. Improving retrieval still matters, but selection and search strategy are substantial bottlenecks. A larger graph is not established as the immediate solution by this experiment.

#### Provider change and cost

DeepInfra became rate-limited after34 successful new runs. At the user's explicit instruction and approval, all34 were preserved byte-for-byte and the remaining46 inputs used CoreWeave. Both endpoints advertise FP8 and support seed/tool calls. Backend/provider differences remain a material confound, even with the same model ID and sampling parameters.

CoreWeave rejected named forced-tool routing at the fifth response. For unfinished conversations only, the continuation used `tool_choice=auto` with `finish` as the sole allowed function. One case still requested disallowed functions and used the budget fallback described above. This compatibility difference weakens equivalence to the original forced-finish baseline and is retained in the report.

Eleven previously paid responses from partially completed conversations were replayed locally, with an assertion that the reconstructed conversation exactly matched the saved conversation before making a new request. They were not resent to the API. All72 predictions already completed before this continuation were also preserved byte-for-byte.

The 80 final new traces contain191 successful model responses versus187 for the cached baseline. The billing ledger contains192 successful response records, including one response from an abandoned partial attempt. Reported new inference cost, including that attempt: **$0.08580917**. Failed-request billing was not assumed to be zero: a conservative additional bound of $0.14238045 is retained, for a combined bound of $0.22818962, below the original $0.25 cap. Credit-purchase fees are excluded.

Current provider rates were checked through the public OpenRouter endpoint listing: [DeepSeek endpoints](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints). The price snapshots and conservative accounting calculations are saved in the experiment manifests.

#### Verification and artifacts

Six new tests pass for paging, visibility, invalid requests, unchanged response limits, label rejection and retained baseline files. An independent trace audit reconstructs the exact code evidence visible before the decision on all160 old/new runs. It reproduces the exposure/accuracy/error counts and verifies original baseline hashes, all34 original-provider success hashes, all72 pre-continuation success hashes and the cost ledger. No evaluation labels were passed to the agent or the GPU service. The original80 leakage audit and filtered example database were reused unchanged.

GPU search ran only in Slurm job2716 on an NVIDIA RTX PRO5000 Blackwell, using the cached SapBERT dictionary embeddings. The service was intentionally stopped after evaluation to release the GPU. The laptop orchestrated OpenRouter requests through the existing `.env`; no API key was copied to the remote machine. Remote commands reused the persistent SSH terminal, and retrieval used a loopback tunnel on the same SSH control socket.

Local implementation:

- `independent_agent/dense_search_service.py`: Slurm-only loopback search backend.
- `independent_agent/dense_agent_tools.py`: bounded search pages and visibility tracking.
- `independent_agent/dense_agent_eval.py`: preparation, original-provider orchestration and scoring.
- `independent_agent/dense_agent_resume.py`: user-approved remaining-runs provider switch.
- `independent_agent/dense_agent_finish.py`: partial-response replay and compatible final-turn continuation.
- `independent_agent/test_dense_agent.py`: six focused tests.

Artifacts: `independent_agent/artifacts/dense_agent_v1/`, including configuration, initial evidence, hidden pools, original/new traces, failed attempts, usage ledger, provider manifests, `results.json`, `all_predictions.csv`, `independent_verification.json` and `analysis/verify.py`.

All remote files remain under `/home/shivank_g/projects/pratham/termmap_adesh`. GPU-service configuration and query caches are in `results/agent_retrieval_v1/`; a local evaluation snapshot is stored alongside them. No existing baseline or team-pipeline implementation was replaced.

#### Recommended next step

Keep the original agent as the baseline. The improved ICD backend is promising, but first isolate it from the SNOMED change and provider difference before promoting it. For SNOMED, test a selective deeper-read policy on this development set and inspect the16 visible-gold selection errors. Broader graph expansion is lower priority than those measured gaps. No further paid ablation or default-backend replacement was performed.


---

## Appendix 11: BACKEND_ABLATION_RESULTS.md

Historical source: `independent_agent/BACKEND_ABLATION_RESULTS.md` (included below).

### ICD-only retrieval and selective SNOMED deeper reading

The ICD-only variant scored 70% on each system on the same 80 development inputs. Selective deeper reading did not improve accuracy. Keep the original agent as the stable baseline, retain ICD-only as a candidate for a larger paired evaluation, and leave this confidence-based paging policy disabled. Neither new arm is promoted to the default.

#### Results

| Configuration | ICD correct | SNOMED correct | Headline minimum | Both codes correct | Budget/workflow fallbacks |
|---|---:|---:|---:|---:|---:|
| Cached original search/inspection | 52/80 (65%) | 57/80 (71.25%) | 65% | 42/80 | 0 |
| Cached combined retrieval + paging | 57/80 (71.25%) | 55/80 (68.75%) | 68.75% | 45/80 | 1 |
| **New ICD-only retrieval change** | **56/80 (70%)** | **56/80 (70%)** | **70%** | **45/80** | **2** |
| New combined retrieval + selective SNOMED review | 52/80 (65%) | 53/80 (66.25%) | 65% | 42/80 | 3 |

All fallbacks remain included, and all 320 new code predictions are dictionary-valid. All 160 new inputs completed with no unresolved transport failures. The new arms made 418 successful model requests at a total reported cost of **$0.18696374**, below the shared $0.30 cap. There were no failed new HTTP attempts or uncertain billing allowances. Historic control inference was not repeated.

The ICD-only and cached combined configurations each get **112 of 160 individual codes correct**, with both codes correct on the same aggregate count of 45 inputs. The 1.25-point headline increase therefore reflects a more balanced pair of system accuracies, not an increase in total correct codes or both-correct inputs.

#### Paired corrections and regressions

Each entry compares the new configuration with the named saved control on identical inputs. Positive deltas favor the new configuration.

| New arm vs control | System | Corrections | Regressions | Delta | Paired bootstrap 95% interval |
|---|---|---:|---:|---:|---:|
| ICD-only vs original | ICD | 6 | 2 | +5 pp | -1.25 to +12.5 pp |
| ICD-only vs original | SNOMED | 3 | 4 | -1.25 pp | -7.5 to +5 pp |
| ICD-only vs combined | ICD | 2 | 3 | -1.25 pp | -6.25 to +3.75 pp |
| ICD-only vs combined | SNOMED | 3 | 2 | +1.25 pp | -3.75 to +6.25 pp |
| Selective review vs combined | ICD | 2 | 7 | -6.25 pp | -13.75 to 0 pp |
| Selective review vs combined | SNOMED | 2 | 4 | -2.5 pp | -8.75 to +3.75 pp |

The ICD-only headline gain versus the original is +5 pp, with a paired headline interval of -3.75 to +11.25 pp. Its headline gain versus combined is +1.25 pp, with an interval of -5 to +6.25 pp. Selective review changes the combined headline by -3.75 pp, with an interval of -11.25 to +2.5 pp. These are descriptive intervals on a repeatedly inspected development sample, without multiplicity correction; no improvement is established.

#### Where the remaining errors occur

Gold visibility is independently reconstructed from the evidence available **before** the deciding model response. Unread hidden pool entries do not count. Budget fallbacks remain included in the retrieved-but-wrong column, so it is not a pure clinical-reasoning measure.

| Configuration | ICD gold seen | ICD seen but wrong | ICD never seen and wrong | SNOMED gold seen | SNOMED seen but wrong | SNOMED never seen and wrong |
|---|---:|---:|---:|---:|---:|---:|
| Original | 66/80 | 14 | 14 | 71/80 | 14 | 9 |
| Combined | 72/80 | 15 | 8 | 71/80 | 16 | 9 |
| ICD-only | 70/80 | 14 | 10 | 73/80 | 17 | 7 |
| Selective review | 69/80 | 17 | 11 | 71/80 | 18 | 9 |

The ICD-only variant has 31 system-level failures despite visible golds, versus 17 where the gold was not exposed. Its existing-evidence oracle would score 87.5% ICD and 91.25% SNOMED, an 87.5% headline; that is a diagnostic ceiling for saved evidence, not an achievable forecast. Selection alone with those candidate sets cannot reach 95%.

Initial dictionary recall remains 55/80 ICD and 59/80 SNOMED for ICD-only, versus 55/80 and 64/80 for the combined backend. Nevertheless, ICD-only exposes more SNOMED golds over the whole run (73 vs 71), because its subsequent searches differ. This shows why initial recall is not the whole agent's ceiling, and why changing one system's retrieval can also alter the shared controller's behavior on the other system.

#### What selective deeper reading actually did

- Review activated on **16/80 inputs**. Six later proposed finishes were recorded as too late for review. The confidence threshold is a heuristic using joint confidence; it does not establish SNOMED shortlist inadequacy.
- One previously unseen gold was exposed by the review page: **“right leg issues.”** The model still selected a different code. Thus the policy demonstrated additional exposure but **zero newly exposed golds selected correctly**.
- Among the 16 reviewed inputs, seven proposed SNOMED selections were correct before review, and five final selections were correct. There were **zero proposed-wrong → final-correct changes** and **two proposed-correct → final-wrong changes**.
- The two changes were **“hardware failure”** and **“significant spinal canal or neural foraminal stenosis.”** The latter shifted from the dataset's general spinal-stenosis gold to a lumbar-specific concept. The former changed from **Device failure** to **Prosthetic joint mechanical failure**. These are exact-match disagreements with the supplied labels, not independently adjudicated clinical conclusions.
- All 16 reviewed runs ultimately finished normally. The three workflow fallbacks occurred on inputs without an activated review, so they are not direct evidence of the added review consuming those inputs' budgets.

These before/after observations are descriptive. They are not a randomized estimate of the page's causal effect, and the policy prompt can alter searches and decisions even when review never triggers. Overall provider-stratified changes are also saved: the selective arm's ICD declines by four correct cases on the 34 DeepInfra inputs and by one on the 46 CoreWeave inputs.

#### Scope for improvement and next decision

1. **Prioritize choosing among visible candidates.** New retrieval has already exposed many correct codes that the agent rejects. A trained candidate selector or the team's eventual reranker is a stronger next integration target than adding more mandatory investigation. Compare candidate selection with identical evidence so retrieval and selection effects can be separated.
2. **Learn dataset-specific granularity and target scope.** Reviewed regressions include choosing a drug-induced constipation concept instead of the generic target, using a broader ICD category instead of its expected child, and transferring an MRSA-positive example's label across a different context. Blanket rules about specificity, negation, or copying examples remain unsafe for exact-match accuracy.
3. **Improve true retrieval misses separately.** ICD-only still has ten ICD and seven SNOMED misses. Inspect those cases for missing terminology aliases, abbreviation ambiguity and wrong concept families before adding a large graph. A graph helps only where a useful entry point and authoritative edges exist; the current experiments do not establish a graph benefit.
4. **Do a larger paired evaluation next, rather than another policy tweak on these 80.** Compare the unchanged original agent with ICD-only on a frozen larger sample, using the same provider and filtered training examples for both arms. The current 500-input retrieval benchmark is available for a broader end-to-end *development* evaluation; it is already consumed retrieval evidence and should not be called untouched confirmation. Account for shared context groups in uncertainty estimates. Reserve a separate untouched set if available for a final generalization claim.

At the new arms' observed mean cost, 500 inputs across two arms would be roughly **$1.17** before retries or changed providers. A $1–$2 planning allowance is reasonable, but prices and the conservative request bound must be checked before that run. **No larger paid evaluation was started in this task.** The original agent, team pipeline and old predictions remain unchanged.

#### Frozen protocol

The same 80 previously inspected development inputs, filtered training-example database, original task prompt, DeepSeek V4 Flash 0731 model, temperature zero, seed 13, disabled reasoning, 2,048 completion-token cap, and five-response maximum are retained. All 160 cached baseline/combined-backend runs are reused without new baseline requests.

- **ICD-only:** retain the new SapBERT ICD retrieval and its paging capability, but restore the original lexical SNOMED search, including its original initial candidate cards. Training examples and the original task/policy prompt are unchanged. The search-tool description states the actual per-system backends. Compare against both cached controls; relative to the combined backend, this removes the SNOMED backend/paging change.
- **Selective SNOMED:** retain the combined SapBERT ICD / exact-BM25-SapBERT SNOMED backend. A proposed valid finish with heuristic overall confidence below 0.85 can return one unread 12-code SNOMED page during responses 1–3. A subsequent response must read that evidence before finishing. At most one review is permitted per input, with no review delay on responses 4–5. Confidence is not calibrated or system-specific. The extra instruction and controller gate are evaluated together.
- Each new input uses the same provider as its cached combined-backend run: 34 DeepInfra and 46 CoreWeave. The older lexical baseline used DeepInfra for all 80, so comparisons with that control retain a provider confound. CoreWeave uses automatic tool choice with a finish-only final menu, matching the compatible continuation used previously.
- A shared $0.30 new-usage cap accounts for both new arms. Successful completed runs are cached and never repeated. Transport failures stop further submissions and are preserved separately from model/workflow fallbacks.

#### Compute limitation

All four GPUs were allocated. Only our newly queued GPU job 2722 was cancelled; pre-existing jobs were untouched. Retrieval ran in CPU-only Slurm job 2725, using the existing dictionary embeddings, cached raw-80 rankings, and previous query cache. All 160 initial full search pools reproduce exactly, including scores and aliases. Re-encoding ten whitespace-equivalent queries on CPU preserved all ten top-12 and top-100 candidate sets, with one top-12 near-tie order swap. New CPU queries can therefore differ in fine ordering from previous GPU arithmetic; this is an additional attribution limitation. Job 2725 was stopped after completing the experiment; existing GPU jobs 2711, 2712 and 2721 were still running afterward.

#### Verification

Thirteen focused checks passed before inference, covering original SNOMED retrieval equivalence, ICD cache/paging use, review activation/deadline/one-page rules, phase menus, label rejection, and the requirement to read new evidence in a later model response. All 80 prepared inputs were checked for the intended candidate substitution and unchanged training examples. Source files, initial evidence and previous artifacts are protected by hashes.

The final independent audit recomputed raw-prediction counts, paired correction/regression direction, response limits, actual provider assignments and billing from saved traces. It verified all 1,750 protected old artifact files and all 160 retained new prediction hashes. The scorer also reconstructs pre-decision exposure for all 320 old/new runs. A scoring-only argument-order correction was recorded before publication: the frozen inference configuration, original draft and before/after scoring-source hashes are preserved; no runtime source, prompts, predictions or API requests changed.

Detailed outputs include `results.json`, `all_predictions.csv`, `changed_predictions.json`, `review_cases.json`, `case_diagnostics.json`, `independent_verification.json`, and `analysis/verify.py`. Per-response provider usage, all traces, CPU source/log snapshots and exact initial-search parity checks are retained.

Artifacts: `artifacts/backend_ablation_v1/`. Entry points: `backend_ablation.py` and `backend_ablation_analysis.py`.


---

## Appendix 12: DIVERSE500_RESULTS.md

Historical source: `independent_agent/DIVERSE500_RESULTS.md` (included below).

### Diverse 500-input agent comparison

Completed both configurations on the same 500 inputs. SapBERT for ICD raises the headline minimum from **67.4% to 69.2%**, but the gain is small and its paired uncertainty interval crosses zero. ICD unseen/rare scores improve slightly; SNOMED unseen/rare scores decline slightly. This does not establish a reliable overall or long-tail improvement. Keep the original agent stable while retaining the ICD variant as an experimental option.

#### Overall results

| Configuration | ICD | SNOMED | Headline minimum | Both codes correct | Workflow fallbacks |
|---|---:|---:|---:|---:|---:|
| Original: text search for both | 337/500 (67.4%) | 352/500 (70.4%) | 67.4% | 266/500 (53.2%) | 28/500 |
| SapBERT for ICD, text search for SNOMED | 346/500 (69.2%) | 346/500 (69.2%) | 69.2% | 265/500 (53.0%) | 35/500 |

All 2,000 final code predictions are dictionary-valid. Every input/configuration run is included, including workflow fallbacks. There are no unresolved transport failures. The 1,000 runs used **2,939 successful API responses**, costing **$1.64027022** in reported usage, below the shared $2 cap. There were no failed HTTP attempts or uncertain billing allowances. Credit-purchase fees are excluded. Inference took 31 minutes 44 seconds with four concurrent inputs; per-input median/p95 latency was 6.00/17.89 seconds for original and 5.89/18.45 seconds for the ICD variant. These are observed timings, not a controlled throughput claim.

The headline gain is **+1.8 percentage points**, with a paired context-cluster bootstrap 95% interval of **-2.12 to +3.78 points**. Across the two systems, the variant adds only three correct individual labels overall (692 vs 689 of 1,000) and loses one both-correct input. It should not be treated as a proven upgrade.

#### Long-tail results

These primary buckets use the exact 3,423-example filtered store available to both agents.

| System | Bucket | Inputs | Original correct | ICD variant correct | Delta |
|---|---|---:|---:|---:|---:|
| ICD | Unseen | 114 | 64 (56.1%) | 67 (58.8%) | +2.63 pp |
| ICD | Rare | 261 | 182 (69.7%) | 188 (72.0%) | +2.30 pp |
| ICD | Common | 125 | 91 (72.8%) | 91 (72.8%) | 0 pp |
| SNOMED | Unseen | 241 | 141 (58.5%) | 138 (57.3%) | -1.24 pp |
| SNOMED | Rare | 229 | 184 (80.3%) | 182 (79.5%) | -0.87 pp |
| SNOMED | Common | 30 | 27 (90.0%) | 26 (86.7%) | -3.33 pp |

For ICD unseen cases, the variant corrects eight and regresses five; its delta interval is -3.70 to +9.00 points. For ICD rare cases it corrects 15 and regresses nine, with an interval of -1.17 to +5.88 points. For SNOMED unseen cases it corrects nine and regresses 12, with an interval of -4.80 to +2.41 points. For SNOMED rare cases it corrects seven and regresses nine, with an interval of -4.33 to +2.54 points. None establishes a reliable improvement. The SNOMED common-code result contains just 30 inputs, so its one-case decline is especially imprecise.

Giving each distinct code equal weight yields overall macro accuracy of **65.55% → 68.49% ICD** and **68.62% → 68.27% SNOMED**. On unseen codes, macro accuracy is **56.07% → 59.35% ICD** and **58.30% → 57.71% SNOMED**. This supports investigating ICD retrieval, but not a claim that the two-system long-tail problem is solved.

#### Paired changes and robustness checks

| System | Corrections | Regressions | Net correct cases | Accuracy delta | Paired cluster 95% interval |
|---|---:|---:|---:|---:|---:|
| ICD | 27 | 18 | +9 | +1.8 pp | -0.63 to +4.44 pp |
| SNOMED | 16 | 22 | -6 | -1.2 pp | -3.56 to +1.18 pp |

On the 445 inputs without prior-agent context overlap, original scores 66.74% ICD and 70.11% SNOMED; the variant scores 68.76% on each. The direction is unchanged, and the older shared contexts do not account for the net gains or losses.

The within-sample near-duplicate audit found two high-similarity context pairs. Grouping those into connected components gives 380 resampling units instead of 382 and a headline delta interval of -2.05 to +3.78 points, leaving the conclusion unchanged. These groups are not verified patient identities.

Using frequencies in the **original full train partition**, before example-store filtering, reassigns seven ICD and five SNOMED inputs. ICD unseen accuracy remains a modest increase, 55.36% → 58.04% (112 inputs); SNOMED unseen remains a modest decline, 58.05% → 57.20% (236 inputs). All corresponding paired intervals still include zero. The primary conclusion is not an artifact of the quarantine/example-memory bucket definition.

#### Retrieval versus final choice

| Measurement | Original ICD | Variant ICD | Original SNOMED | Variant SNOMED |
|---|---:|---:|---:|---:|
| Gold in initial dictionary top 12 | 215 (43.0%) | 314 (62.8%) | 314 (62.8%) | 314 (62.8%) |
| Gold in initial evidence including examples | 315 (63.0%) | 375 (75.0%) | 351 (70.2%) | 351 (70.2%) |
| Gold seen before final decision | 412 (82.4%) | 430 (86.0%) | 432 (86.4%) | 435 (87.0%) |
| Final correct | 337 (67.4%) | 346 (69.2%) | 352 (70.4%) | 346 (69.2%) |
| Wrong despite visible gold | 75 | 84 | 80 | 89 |
| Wrong and gold never seen | 88 | 70 | 68 | 65 |

SapBERT adds **99 initial ICD gold candidates**, but only **18 additional eventual exposures** and **nine additional correct ICD answers**. The original agent often repairs its weaker first search through subsequent searches and examples. This is why first-pass recall alone overstates the expected end-to-end benefit.

The variant has **173 system-level failures despite visible golds**, versus **135 retrieval misses**. Of those 173 visible-gold failures, 37 occur in workflow-fallback outputs and 136 follow accepted model finishes. An oracle restricted to the variant's saved visible evidence would have an 86% ICD / 87% SNOMED ceiling. That is a diagnostic for these traces, not an achievable forecast; both retrieval and selection still need improvement to approach the problem statement's 95% target.

For unseen ICD specifically, gold visibility rises from 82/114 to 89/114 while final correctness rises from 64 to 67. For unseen SNOMED, gold visibility rises from 191/241 to 194/241 while final correctness falls from 141 to 138. Merely exposing additional candidates does not ensure the agent selects them.

#### Workflow and case-level findings

- **Finalization reliability particularly affects the long tail.** Original has 28 fallback inputs, 27 of which have unseen SNOMED golds. The variant has 35, including 29 with unseen SNOMED golds and 18 with unseen ICD golds. All fallback runs exhausted the response limit while still requesting investigation tools. Their failures are not transport outages, and they remain part of the primary score.
- **Paging was used but did not directly correct a newly discovered gold.** The variant paged on 54 inputs. One ICD gold was first exposed through paging, but was not selected. Sixteen of its 70 remaining ICD retrieval misses were already in the initial hidden 100-code pools; 54 were absent from those initial pools. This is not the ceiling for reformulated searches.
- **Target scope and annotation conventions remain difficult.** Examples among unseen SNOMED errors include mapping “cortical collapse” to a contextual avascular-necrosis diagnosis rather than the dataset's Collapse concept, and mapping historical “drug overdose” to a history-of-poisoning concept rather than the dataset's Overdose gold. These observations do not independently adjudicate clinical correctness or justify a universal rule about ignoring context/history.
- **Training examples can support a competing interpretation.** The model's explanation for the renal-cyst example follows a Simple renal cyst training convention while the gold is Renal cyst. Example/code co-occurrence is an association, not proof of causal copying. Blanket removal of examples is not supported by this comparison.
- **Exact identifier lookup deserves a small engineering check.** Ten valid literal ICD-code search queries occurred in the variant; one failed to return the queried gold code, later recovered by another tool. Original lexical search had no gold-code lookup misses in this diagnostic. This identifies a tool-interface gap, not a demonstrated final-answer loss or a measured accuracy improvement from a proposed fix.

#### Decision after this evaluation

Do not promote the ICD variant as an established improvement, and do not add a larger graph based on these results. Retain both frozen configurations and this benchmark for development comparisons.

The strongest measured next targets are **reliable finalization within the turn budget** and **better candidate selection on unseen codes**. Fixing finalization should be tested without changing retrieval; testing a stronger selector should hold evidence fixed. The team's trained reranker could be assessed later if its training split is compatible. Remaining retrieval misses deserve a separate alias/abbreviation/identifier-lookup audit. More prompt/tool complexity alone has not demonstrated a reliable gain.

No new training, graph expansion, prompt changes or additional paid experiments were performed after these results. The original agent and team pipeline remain unchanged.

#### What is compared

- **Text search for both systems:** the original search/inspection agent, using original lexical retrieval for ICD and SNOMED.
- **SapBERT for ICD, text search for SNOMED:** the previously tested ICD variant, including its ICD deeper-page interface. SNOMED search remains identical to the original.

Both configurations are freshly evaluated on all 500 inputs using DeepSeek V4 Flash 0731, pinned to CoreWeave with provider fallback disabled. They share the task prompt, examples, context payload, SNOMED initial candidates, temperature zero, seed 13, disabled reasoning, a 2,048-token completion cap and five-response limit. CoreWeave uses automatic tool choice with a finish-only menu on the final response for both arms. No query-recovery, relationship, candidate-comparison or selective-review policies are added. The shared new-usage cap is $2, across 1,000 input/configuration runs.

#### Sample diversity and long-tail definition

Selection considered all 507 paired diagnosis inputs in the existing dev/test partitions that were absent from earlier agent-evaluation input IDs. It selected 500 deterministically, covering chart groups first, then favoring code diversity, long-tail coverage and underrepresented broad ICD-derived clinical domains. Selection did not use retrieval outcomes or model correctness.

The set contains **382 normalized context groups**, **336 unique ICD codes**, **431 unique SNOMED codes** and **20 broad clinical domains**. Domain labels are derived from ICD ranges for diagnostics; they are not a claim of an official clinical adjudication. Symptoms/signs comprise 165 inputs, circulatory conditions 63 and musculoskeletal conditions 59; the remaining domains and challenge-feature counts are saved in the selection manifest. The data do not support uniform counts in every domain.

Frequency is measured in the exact filtered training-example store available to both agents:

| Bucket | ICD inputs | Unique ICD codes | SNOMED inputs | Unique SNOMED codes |
|---|---:|---:|---:|---:|
| Unseen: 0 permitted examples | 114 | 107 | 241 | 229 |
| Rare: 1–5 permitted examples | 261 | 179 | 229 | 183 |
| Common: 6+ permitted examples | 125 | 50 | 30 | 19 |

“Unseen” does not mean absent from the public dictionary or the LLM's pretraining. Both agents may search the full supplied dictionaries. Frequencies are frozen before inference and used only for selection/reporting, never passed as evaluation labels to the agent.

The sample is deliberately broad development evidence. None of its input IDs appeared in earlier agent evaluations, but **55 inputs share chart context with those experiments**, and **442 inputs** were included in the earlier retrieval-only 500-input benchmark. It is not untouched confirmation. Repeated contexts are retained as clusters in uncertainty estimates, and results on the 445 inputs without prior-agent context overlap are reported separately.

#### Training/example separation

The common store contains **3,423 diagnosis-labelled training examples**. Fourteen whole training groups were excluded, including ten near-duplicate groups. There are zero shared input IDs, normalized mentions, or identical normalized context groups between this store and the 500 inputs. Exhaustive test-to-training three-word-shingle Jaccard comparison has a maximum of 0.837607, below the 0.85 quarantine threshold.

The agent receives only input ID, mention and context. Gold labels, bucket assignments, sample domains and the evaluation file remain outside the runtime tool interface. No agent's previous evaluation predictions become memory.

#### Verification and interpretation

Eighteen focused checks passed before inference. Preparation checks all 500 paired evidence payloads for unchanged examples and SNOMED candidates across arms. Sources, labels, examples, prepared evidence and prior experiment files are protected by hashes. Successful completed runs are never repeated; infrastructure retries reconstruct and verify the conversation using saved paid responses before issuing another request. Model/workflow fallbacks remain in the score, while unresolved infrastructure failures prevent final scoring.

The report includes per-system accuracy and headline minimum, unseen/rare/common accuracy with denominators, macro accuracy across distinct codes, initial/ever-visible gold recall, selection errors versus retrieval misses, paired corrections/regressions, failures, cost and latency. Uncertainty uses 5,000 paired bootstrap resamples of complete normalized chart-context groups. Small subgroup counts, particularly the 30 common-code SNOMED inputs, limit precision. No alternative weighted score is substituted for the problem statement's headline metric. Intervals are descriptive development estimates without a multiplicity adjustment.

Retrieval ran in CPU-only Slurm job 2727 using the existing pretrained SapBERT revision and dictionary embeddings. Both configurations use fresh model requests on the same provider; the comparison is not against historical mixed-provider scores. Existing GPU allocations and queued jobs were untouched. Only our CPU service was stopped after the evaluation.

The independent audit reproduces all system/bucket counts and paired correction/regression directions from raw predictions, reconstructs pre-decision visibility for all 1,000 runs, verifies dictionary membership, actual provider assignments, response limits, unique billed generations, source/data/prior-artifact hashes and the cost ledger. It passed. All new requests succeeded on CoreWeave without transport retries or replayed paid prefixes.

Artifacts are under `independent_agent/artifacts/diverse500_v1/`, including `results.json`, `all_predictions.csv`, `changed_predictions.json`, `case_diagnostics.json`, `independent_verification.json`, `workflow_diagnostics.json`, both sensitivity reports, manifests, usage ledger and every input/configuration trace. Source entry points are `diverse500_prepare.py`, `diverse500_eval.py`, and `diverse500_analysis.py`. The `analysis/` artifact directory contains the independent verifier and diagnostic/sensitivity scripts. Remote source and execution logs are archived in `remote_execution.tar.gz`.


---

## Appendix 13: AGENT_NEXT_STEPS_ANALYSIS.md

Historical source: `independent_agent/AGENT_NEXT_STEPS_ANALYSIS.md` (included below).

### Agent priorities after the diverse 500-input evaluation

The next focus should be reliable finalization and evidence-grounded candidate selection, followed by context-aware training memory. Analyst guidance is worth testing as a small, candidate-specific evidence layer. A large semantic graph is a later experiment: the present evidence does not establish graph traversal as the missing component.

This analysis combines the completed 500-input comparison, a new read-only trace audit, the earlier 80-input error/prompt/tool experiments, and primary literature/official terminology guidance. No inference calls, weight training, changes to evaluation labels, or changes to the runtime agent were made for this analysis.

#### What the measurements establish

The original scores 67.4% ICD and 70.4% SNOMED; pretrained SapBERT for ICD with original SNOMED retrieval scores 69.2% on both. Its improvement is inconclusive under the paired context-cluster intervals. Both-correct accuracy is 53.2% versus 53.0%. The earlier selective SNOMED deeper-reading policy also failed to demonstrate improvement. Preserve these controls rather than treating either intervention as an established upgrade.

ICD initial dictionary recall@12 increases from 43.0% to 62.8%: 99 extra initial golds. But eventual ICD gold exposure increases by only 18, and correct ICD answers by nine. The original agent compensates for weaker first retrieval through later searches/examples. Initial recall therefore overstates the potential end-to-end difference.

The variant makes 308 wrong individual code predictions across 1,000 labels:

| Failure description | Code predictions | Interpretation |
|---|---:|---|
| Gold visible, accepted model selection wrong | 136 | Direct candidate-selection target; visibility does not establish that the model understood the candidate |
| Gold visible, output from budget fallback | 37 | Workflow reliability contributes; obtaining a model finish still would not guarantee correctness |
| Gold never visible, output wrong | 135 | Additional retrieval evidence required; includes both accepted and fallback outputs |

The 37 fallback predictions are system-level outputs, not 37 separate inputs. There are 35 fallback inputs in this arm. All exhausted the five-response budget while still requesting investigation. They disproportionately involve unseen SNOMED labels. The saved-evidence oracle ceiling is 86% ICD and 87% SNOMED. Selection work alone cannot reach 95% with these candidate sets.

The new audit reconstructs evidence before the accepted finish response, excluding unread results generated alongside it:

| Among accepted visible-gold errors | ICD | SNOMED |
|---|---:|---:|
| Total | 69 | 67 |
| Gold already in initial evidence | 49 | 47 |
| Gold explicitly inspected | 31 | 40 |
| Predicted code also present in visible training examples | 25 | 34 |
| Gold also present in visible training examples | 23 | 11 |
| Both alternatives have training-example support | 11 | 8 |
| Gold and prediction have different stored semantic tags | n/a | 24 |

Rows overlap. Example support is association, not proof of copying. A different semantic tag is a diagnostic, not automatic proof of clinical error. These counts demonstrate that many failures persist after discovery and inspection; they do not uniquely identify deficient schema knowledge as the cause.

Case evidence illustrates different mechanisms:

- **Specificity:** “upper pole left renal cyst” is linked to Simple renal cyst, following a training example; the supplied gold is Renal cyst. The target context specifies size/location but does not say “simple.” Both candidates were inspected. A useful comparison should expose the extra characteristic introduced by the selection.
- **Target focus:** “cortical collapse” is linked to avascular necrosis of the femoral head, supported by an example that explicitly mentions both collapse and avascular necrosis. The target only says there is no evidence of cortical collapse/flattening; gold is the morphology concept Collapse. This combines focus, example transfer, and assertion-policy questions.
- **Annotation convention:** “drug overdose” in a past medical history list is linked to H/O: poisoning, following a close historical training example; gold is Overdose. The model inspected the gold. This is not resolved by teaching the definitions of history and disorder alone.
- **Noisy target:** “premedication of other armsequo osteoarthritis” is interpreted as knee osteoarthritis, while gold is Premedication prescribed. The target itself is garbled and contains competing concepts. Do not label every such disagreement as simple model ignorance.

#### Analyst guidance: yes, with an explicit task policy

ICD and SNOMED guidance should be separate. Useful ICD evidence includes Alphabetic Index entry, Tabular inclusion/exclusion notes, other-specified versus unspecified, laterality/acuity, and applicable encounter characters. Official guidelines distinguish insufficient detail from detail for which no specific code exists; these are different reasons to select unspecified or other-specified. [CMS FY2027 guidelines, sections I.A.1–5 and I.A.9](https://www.cms.gov/files/document/fy-2027-icd-10-cm-coding-guidelines.pdf).

For SNOMED, provide the fully specified name, semantic tag, synonyms and available verified defining relationships, and compare the intended meaning. SNOMED's mapping guidance explicitly ties target scope and equivalence to the use case. Its methodology requires documented assumptions and rules for managing context. [Mapping to SNOMED CT](https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-mapping-guide/map-development/9-mapping-to-snomed-ct), [mapping methodology](https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-mapping-guide/introduction/3-mapping-methodology).

Our task links a highlighted mention to one supplied code per system; it is not full encounter billing. Supplied legacy ICD categories remain possible labels, and SNOMED diagnosis-labelled rows include situations, morphology, observable entities and substances. Blindly applying contemporary billable-leaf rules or filtering to disorders would conflict with the task. Match official resources to the supplied dictionary release before treating notes as authoritative for a candidate; the current FY2027 document is a reference, not proof of release compatibility.

Write a concise task-specific coding guide from the actual annotation instructions if available, otherwise from training-only reviewed examples. Distinguish documented rules, empirical conventions, and unresolved conflicts. Do not infer a universal convention from these 500 golds. Prior examples show opposing pressures on specificity and negation. A coherent analyst guide can clarify the task; unresolved annotation disagreements still require adjudication and should not be silently repaired.

The current prompt already contains basic specificity, semantic-tag, target-focus and example cautions. A previous generic example-equivalence addition regressed both systems from 65/71.25% to 62.5/68.75% on 80 inputs. Mandatory comparison/recovery and selective paging also failed to establish gains. Therefore the proposed new experiment must change the evidence representation and isolate selection; appending another general paragraph to the same end-to-end controller is weakly motivated.

For each shortlisted candidate, show a compact comparison: the concept meaning, what extra detail it requires, where that detail is supported in the target text, relevant official notes, and comparable labelled examples with their differences. Mark unavailable evidence as unavailable. Avoid invented necessary-and-sufficient definitions from a short display name. This is the specificity mechanism worth testing; an extra mandatory reasoning turn is unnecessary.

#### Training data and a useful version of the graph idea

Training data has already been used as demonstrations: the filtered store contains **3,423 labelled rows**, comprising 1,573 ICD rows across 714 codes and 1,850 SNOMED rows across 1,274 codes. It has not been used for independent-agent weight training or learned contextual prototype retrieval. Exact casefolded/whitespace-normalized mentions are unique within each system in this store; exact phrase memorization is not the main opportunity.

Start with contextual case memory, which can be represented as a small typed graph without a graph database:

```text
training case (mention + local context + source ID)
  --annotated_as--> supplied terminology code
  --has_evidence_for--> site / laterality / acuity / assertion / intent
  --similar_to--> another case, with a similarity score
code --has_description--> supplied dictionary alias / FSN / semantic tag
code --verified_is_a--> parent, only when an authoritative relationship exists
paired training case --coannotated_as--> ICD code and SNOMED code
```

The first two edges have different status: the label is observed; extracted attributes are hypotheses with text spans and confidence. Similarity, co-occurrence and coannotation are not synonymy, IS-A, or a rigid crosswalk. The current SNOMED workbook lacks ontology relationships; word overlap must not be relabelled as an official graph edge. The supplied dictionaries remain the full candidate universe.

Retrieve a few contextually comparable cases plus a close contrasting case, rather than several superficially similar positive examples. Compare target and example modifiers before transferring support. Candidate-specific example retrieval should expose both alternatives when examples exist. Preserve code diversity rather than filling memory with the most common label. For a code absent from training, use its dictionary meaning, compatible related cases and verified relatives; memory cannot retrieve a nonexistent labelled example of that code.

KRISSBERT provides evidence for context-encoded mention prototypes and retrieval against them, including incorporation of labelled mentions. Our proposed small supervised memory is inspired by this mechanism; it is not a reproduction of its large self-supervised training system. [Knowledge-Rich Self-Supervision for Biomedical Entity Linking](https://aclanthology.org/2022.findings-emnlp.61/).

Memory can also improve search: infer a bounded set of alternative terms from the target and compatible cases, preserving decisive modifiers, then union dictionary candidates with provenance. This targets the 135 true exposure misses. Do not insert evaluation gold aliases, require that every query variant consume another model turn, or discard the original mention on the basis of an uncertain interpretation.

#### What the literature changes in our priorities

- **Selection is a separate learning problem.** BeLink trains a set-wise multiple-choice reranker over candidate concepts and mention context. That supports a constrained selector experiment, rather than assuming another free-form agent instruction will suffice. Its biomedical benchmarks and trained Qwen models differ from TermMap and the fixed DeepSeek agent; its gains are not a forecast here. [BeLink](https://arxiv.org/html/2605.22501v1).
- **Surface form and context deserve distinct treatment.** PILOT uses separate surface/context scoring with fusion; BioELX separates alias retrieval from context-aware ranking. This agrees with our measured retrieval/selection gap. It does not justify putting the whole chart into the existing SapBERT query encoder. [PILOT](https://arxiv.org/abs/2608.04144), [BioELX](https://arxiv.org/abs/2605.27380).
- **Contextual memory is better supported than a speculative large graph as the first use of labels.** KRISSBERT offers a concrete prototype mechanism. Official SNOMED guidance supplies requirements for meaning and scope, but actual verified relationships are needed for authoritative traversal. A graph becomes useful when it supports a measured operation—candidate expansion or disambiguation—and should be assessed against a flat memory using the same evidence.

#### Recommended experiment sequence

1. **Finalization alone:** retain retrieval and model, reserve a terminal selection phase with no search dispatch, validate structured selection against observed candidates, and keep every failed finish in scoring. The current provider does not enforce the named forced-tool choice; implementation must handle that explicitly. Evaluate the phase on all inputs, including correct controls and no-gold cases. Do not claim the 37 errors will all be fixed.
2. **Selection with fixed evidence:** replay saved candidates/context/examples for all 500 inputs; use the same DeepSeek model and a compact selection input. No search in this diagnostic. Compare compact selector alone against the same selector plus candidate-specific coding guidance. Measure accuracy conditional on gold visibility, overall accuracy, both-correct, unseen/rare accuracy, regressions, valid completion, cost and latency. A fixed pool is a selection diagnostic, not an end-to-end replacement score. Never add a gold candidate just to make the task answerable.
3. **Training memory separately:** freeze the better supported selector, then compare current example retrieval against contextual/contrastive memory from the allowed training groups. Test memory for selection separately from memory-assisted query reformulation; changed evidence and changed selection must not be conflated. No extra mandatory comparison tool.
4. **Learn a selector if these tests still expose choice errors:** build training-only mention/context/candidate sets with difficult neighboring codes and conflicting semantic tags. Use held-out training groups for tuning, include code-disjoint validation, and randomize candidate order. Assess the team's reranker only if its training split is compatible. Keep this distinct from the current fixed-model agent ablations; no new model comparison is needed now.
5. **Then targeted retrieval/graph expansion:** audit abbreviation senses, aliases, literal identifier lookup and missing concept families. Compare flat memory with the same memory plus verified hierarchy expansion. Adopt graph traversal only if it adds useful candidates and final correct choices at acceptable cost, especially on unseen codes.

Use paired context-cluster uncertainty and explicit correction/regression counts. The 500 inputs are already development evidence, partly overlapping earlier retrieval/context exposure. Freeze rules/memory from training only; do not train on these traces or golds. Any final confirmation claim needs separately reserved inputs compatible with the training split. Avoid another broader evaluation before a candidate change demonstrates a plausible, isolated benefit.

#### Reproduction and limits

Run `python3 -m independent_agent.agent_focus_audit` from the project root. It imports no inference client, opens SQLite databases read-only, checks the 136-error and 3,423-example totals, and checks all 1,002 input source hashes before/after. Output is under `artifacts/agent_focus_audit_v1/`: `summary.json`, `cases.json`, and `source_hashes.json`. Official-guidance and KRISSBERT snapshots used during this review are stored there too. The audit passed.

The new audit flags are mechanically reconstructed; case interpretations remain reviewer hypotheses, not clinical adjudications or causal attributions. No full manual recategorization of all 308 wrong labels is claimed. The recommendation is a prioritized test plan, not evidence that analyst rules, memory or learned selection have already improved TermMap.


---

## Appendix 14: SELECTOR500_RESULTS.md

Historical source: `independent_agent/SELECTOR500_RESULTS.md` (included below).

### Fixed-evidence selector: 500 inputs, three conditions

**The coding guide improves the compact selector, and lexical candidate distinctions add ICD gains, but neither establishes an upgrade over the existing search agent. Keep the agent unchanged. The strongest next engineering target is a terminal selector only when investigation exhausts its budget, followed by a separate contextual-memory experiment.**

All 1,500 selections completed on the same fixed evidence sets. There were no transport errors, invalid outputs, or deterministic output fallbacks. Reported new API usage was **$1.17187671**, below the shared $2 cap. All requests returned the requested DeepSeek model on CoreWeave. No GPU or remote retrieval service was used, and no existing process was stopped.

#### Results

The saved reference below is the **existing agent with SapBERT for ICD and original lexical SNOMED retrieval**. It is not the original text-only agent. Its saved results are reused without new API calls.

| Configuration | ICD | SNOMED | Headline minimum | Both correct |
|---|---:|---:|---:|---:|
| Saved search agent, ICD variant | 346/500 (69.2%) | 346/500 (69.2%) | 69.2% | 265/500 (53.0%) |
| A: compact terminal selector | 327/500 (65.4%) | 348/500 (69.6%) | 65.4% | 244/500 (48.8%) |
| B: A + coding/task guide | 337/500 (67.4%) | 351/500 (70.2%) | 67.4% | 254/500 (50.8%) |
| C: B + lexical candidate distinctions | 344/500 (68.8%) | 348/500 (69.6%) | 68.8% | 256/500 (51.2%) |

C and the saved agent each get 692/1,000 individual labels correct, but C loses nine both-correct inputs. The unchanged original text-only control from the preceding experiment scored 67.4% ICD / 70.4% SNOMED / 53.2% both; it was not rerun here. Neither control is replaced by this experiment.

| Contrast | System | Corrections | Regressions | Delta | Paired context-cluster 95% interval |
|---|---|---:|---:|---:|---:|
| B versus A | ICD | 19 | 9 | +2.0 pp | 0.00 to +4.00 pp |
| B versus A | SNOMED | 13 | 10 | +0.6 pp | -1.22 to +2.47 pp |
| C versus B | ICD | 17 | 10 | +1.4 pp | -0.60 to +3.44 pp |
| C versus B | SNOMED | 13 | 16 | -0.6 pp | -2.73 to +1.51 pp |
| C versus A, secondary | ICD | 23 | 6 | +3.4 pp | +1.41 to +5.46 pp |
| C versus A, secondary | SNOMED | 16 | 16 | 0.0 pp | -2.23 to +2.22 pp |

The headline B-minus-A interval is **-0.21 to +3.90 pp**; C-minus-B is **-1.49 to +3.16 pp**. Both primary headline intervals include zero. C-minus-A also includes zero for the headline (-0.21 to +5.13 pp), despite the positive ICD-only interval. These are descriptive development intervals across multiple comparisons, not multiplicity-adjusted confirmation.

Replacing the saved agent with A produces 15 ICD corrections and 34 regressions (-3.8 pp; ICD interval -6.54 to -1.04 pp). This is evidence against replacing the whole workflow with this compact selector. The contrast changes presentation and terminal protocol, so it does not isolate the causal effect of removing the old fallback mechanism.

#### Long-tail performance

Buckets are unchanged from the preceding evaluation and use the permitted 3,423-row example store. Unseen means no permitted example for that code, not absence from the dictionary or model pretraining.

| System | Bucket | Inputs | Saved agent | A: compact | B: guide | C: distinctions |
|---|---|---:|---:|---:|---:|---:|
| ICD | Unseen | 114 | 67 (58.8%) | 62 (54.4%) | 64 (56.1%) | 62 (54.4%) |
| ICD | Rare, 1–5 | 261 | 188 (72.0%) | 178 (68.2%) | 184 (70.5%) | 188 (72.0%) |
| ICD | Common, 6+ | 125 | 91 (72.8%) | 87 (69.6%) | 89 (71.2%) | 94 (75.2%) |
| SNOMED | Unseen | 241 | 138 (57.3%) | 141 (58.5%) | 140 (58.1%) | 144 (59.8%) |
| SNOMED | Rare, 1–5 | 229 | 182 (79.5%) | 180 (78.6%) | 185 (80.8%) | 178 (77.7%) |
| SNOMED | Common, 6+ | 30 | 26 (86.7%) | 27 (90.0%) | 26 (86.7%) | 26 (86.7%) |

C's net ICD improvement over A comes entirely from rare/common inputs; unseen ICD remains 62/114. Its SNOMED unseen gain is offset by rare-code losses. There is no demonstrated two-system long-tail upgrade. Bucket-specific paired counts/intervals are in `results.json`; the small common SNOMED group remains imprecise.

#### Selection with visible gold

Every condition has exactly the same candidate IDs. Gold is visible for **430 ICD** and **435 SNOMED** inputs, matching the saved source trace evidence. None of these selectors can repair the 70 ICD and 65 SNOMED exposure misses without a later retrieval change.

| Configuration | ICD correct / gold visible | SNOMED correct / gold visible |
|---|---:|---:|
| Saved agent | 346/430 (80.47%) | 346/435 (79.54%) |
| A: compact | 327/430 (76.05%) | 348/435 (80.00%) |
| B: guide | 337/430 (78.37%) | 351/435 (80.69%) |
| C: distinctions | 344/430 (80.00%) | 348/435 (80.00%) |

Instructions and literal comparisons recover some selection performance lost by compacting the source evidence, but they do not solve the original selection gap. All output codes are dictionary-valid and belong to the corresponding visible candidate sets; validity is not clinical or exact-match correctness.

#### Finalization and preservation of correct decisions

The saved agent has 35 budget-fallback inputs. New selectors finish validly on every input, including those 35, but replacing previously accepted decisions introduces regressions:

| Subgroup | System | Saved correct | A correct | B correct | C correct |
|---|---|---:|---:|---:|---:|
| Source fallback, 35 inputs | ICD | 0 | 6 | 8 | 7 |
| Source fallback, 35 inputs | SNOMED | 6 | 18 | 19 | 17 |
| Source accepted finish, 465 inputs | ICD | 346 | 321 | 329 | 337 |
| Source accepted finish, 465 inputs | SNOMED | 340 | 330 | 332 | 331 |

This supports preserving accepted source decisions while investigating a terminal selector for budget failures. The route is observable at runtime and does not need a gold label. It is not evidence for overriding accepted decisions whenever another selector disagrees.

As a **post-hoc offline composite only**, retaining all 465 accepted source decisions and substituting B on the 35 fallback inputs yields **354/500 ICD (70.8%)**, **359/500 SNOMED (71.8%)**, and **270/500 both correct (54.0%)**. These numbers reuse existing outputs and cost no additional calls. This composite was not independently executed as an end-to-end agent, was considered after inspecting results, and is not a promoted configuration or a confirmation claim. A controlled terminal-phase implementation should be evaluated separately before adoption.

#### Case checks

- **Renal cyst:** all three conditions still select Simple renal cyst where gold is Renal cyst. Even explicit guide text and lexical “simple” comparisons do not reliably prevent unsupported or convention-driven specificity. This directly limits the hypothesis that a general specificity instruction suffices.
- **Historical drug overdose:** all three retain H/O: poisoning rather than the dataset's Overdose gold, citing the similar historical training example. Terminology definitions alone do not resolve the annotation-policy tension.
- **Cortical collapse:** C selects the SNOMED morphology gold while A/B select a contextual avascular-necrosis diagnosis. Its ICD remains wrong. This illustrates a useful correction without establishing a general mechanism from one example.
- **Garbled premedication/osteoarthritis target:** all three keep the knee-osteoarthritis interpretation and miss the SNOMED Premedication prescribed gold. Noisy target focus remains difficult.

These are exact-match observations with stored explanations, not independent clinical adjudications or proof of the model's internal reasoning. No case-specific correction was added to the prompts or training memory.

#### Cost and latency

| Condition, 500 selections | Reported cost | Median response latency | p95 |
|---|---:|---:|---:|
| A: compact | $0.23443123 | 1.31 s | 2.08 s |
| B: guide | $0.24733871 | 1.39 s | 2.30 s |
| C: distinctions | $0.69010677 | 2.14 s | 3.53 s |
| Total | **$1.17187671** | — | — |

C costs about **2.79 times B**. Its display comparisons and occurrence records substantially increase input size. That extra evidence does not establish an overall or long-tail benefit sufficient to justify adoption. Latencies are observed with eight concurrent requests; comparisons with the earlier four-concurrent, multi-response agent are not controlled throughput measurements. Conservative uncertain billing allowance is zero; no failed requests or retries occurred. Credit-purchase fees are excluded. The three successful pilot requests are part of the 1,500, not additional billed samples.

#### What was tested and verified

All conditions use DeepSeek V4 Flash 0731 on CoreWeave, seed 13, temperature zero, reasoning disabled, a 2,048-token output cap and one terminal response. The API JSON Schema constrains each code field to its actual candidate IDs. No search tools are available. Every completed paid response is stored before output parsing and retained on resume.

The pools include only codes read before the source deciding response. Uniform dictionary cards provide preferred name, FSN, semantic tag and up to four synonyms to all arms; existing visible training examples are retained. Some cards are richer than the source displays, so source-versus-A is not a pure finalization ablation. B adds a fixed general coding/task guide, not a full analyst manual or per-code official Tabular notes. C adds deterministic lexical distinctions against two similar candidate displays and literal text occurrences, not formal definitions or clinical entailment judgments. The experiment does not refute richer authoritative rule retrieval or learned contextual selection.

No evaluation gold labels are read by preparation/inference. No source predictions or assistant interpretations are supplied to the new selectors. No hidden initial100 entries, new example retrieval, graph edges, query rewriting or learned weights enter any arm. Six source emergency fallback codes were absent from the corresponding visible pools; their fallback choices were replaced consistently by a visible candidate. Since every response was valid, none of these deterministic fallbacks affected scores.

Ten protocol tests passed before inference, including candidate identity across arms, prompt isolation, code enums, invalid-output validation, literal offsets, and exclusion of unread co-generated/final-response evidence. Independent preparation checks passed for all 500 inputs. A separate final audit passed frozen hashes, candidate identity, training-example provenance, dictionary validity, raw response/prediction agreement, exact requested/returned model and provider, unique generation IDs, metrics, bucket counts, paired corrections/regressions, and billing. All 1,500 raw responses ended normally.

The inputs remain development evidence: 382 context groups, 336 distinct ICD and 431 distinct SNOMED codes, with previously documented chart/retrieval exposure. Paired intervals resample whole context groups 5,000 times. There is no fresh confirmation set or multiplicity correction. No new training, runtime-agent edits or model comparisons were performed.

#### Decision and next experiment

1. Preserve the original agent and experimental ICD backend controls; do not replace accepted choices with A/B/C.
2. Test a dedicated, bounded terminal phase only when investigation would otherwise exhaust its budget, preserving accepted finishes. Use the compact guide rather than automatically paying for C's larger comparison cards. The post-hoc composite motivates this test but does not replace it.
3. For the remaining semantic-choice problem, test contextual and contrasting training memory independently. The persistent renal-cyst/history errors make another generic instruction addition a weak next step. Keep retrieval fixed for the memory-selection experiment; assess memory-assisted querying separately.
4. Consider learned candidate selection after compatible train-only preparation, with code-disjoint validation. A larger graph remains contingent on verified relationships and measured candidate/selection benefit.

#### Artifacts and reproduction

All artifacts are under `artifacts/selector500_v1/`: exact frozen prompts/config, 500 prepared evidence sets, provider-price snapshot, 1,500 raw responses/results, metrics, paired intervals, predictions, subgroup diagnostics, changed cases, independent verification, post-hoc composite and execution logs. `PROTOCOL.md` documents the comparisons and official sources. Source entry points are `selector500_eval.py` and `test_selector500.py`.

```bash
python3 -m unittest independent_agent.test_selector500 -v
python3 -m independent_agent.selector500_eval verify
python3 -m independent_agent.selector500_eval analyze
python3 independent_agent/artifacts/selector500_v1/diagnostics.py
python3 independent_agent/artifacts/selector500_v1/verify.py
```

These verification/analysis commands make no inference calls. `run` resumes only incomplete selections and reuses any saved successful paid response.

#### Local disk cleanup before inference

Root filesystem free space increased from **556 MiB to about 4.7 GiB**, reclaiming roughly **4.1 GiB**, before creating this experiment. Cleanup used `uv cache clean`, `pip cache purge`, `npm cache clean --force`, and `pnpm store prune`. uv removed 47,619 files / 2.1 GiB; pnpm removed 40,339 files / 799 unreferenced packages. Package caches can be rebuilt or downloaded as needed.

Project source/data, old experiment traces, downloads, installed editor packages and running processes were preserved. The remaining large directories include editor installations, downloaded files and application packages; they were identified but not deleted. The new experiment occupies approximately 37 MiB. `disk_cleanup.json` records the cleanup actions and free-space observation. The filesystem remained around 4.6 GiB free after inference.


---

## Appendix 15: REASONING100_RESULTS.md

Historical source: `independent_agent/REASONING100_RESULTS.md` (included below).

### Reasoning off versus high on condition B: 100-input pilot

**High reasoning did not improve this bounded pilot. Keep reasoning off for the current selector, and do not expand high mode to 500 on the strength of these results.** With identical evidence, guide, schema and an 8,192-token output cap, ICD accuracy fell from 60% to 58% and SNOMED from 73% to 63%. High mode also failed to finish on 13 inputs and used substantially more time and tokens. This is evidence about high reasoning at this completion budget, not every possible thinking configuration.

All 200 requests completed for **$0.18416906**, below the shared $1 pilot cap. There were no transport failures, retries or provider switches. Reasoning activation was verified on every high response. The 13 truncated model outputs remain scored with the deterministic visible-code fallback, rather than being dropped or retried with a more favorable budget. No production/default configuration changed.

#### What was compared

Both arms use **condition B from the previous study**: the compact selector plus the same ICD/SNOMED/task guide. Each receives the same saved mention/context, dictionary candidate cards, visible training examples, candidate order and candidate-constrained JSON Schema. No new retrieval, graph, training memory or weight updates were introduced.

| Setting | Off arm | High arm |
|---|---|---|
| Model/provider | DeepSeek V4 Flash 0731 / CoreWeave | Identical |
| Reasoning object | `{"enabled":false}` | `{"effort":"high"}` |
| Maximum output tokens | 8,192 | 8,192 |
| Requested temperature/seed | 0 / 13 | Identical |
| Responses per input | One | One |
| Evidence, guide, schema | Frozen condition B | Identical |

Both controls were freshly run with the larger common cap; the older 2,048-token outputs were not substituted as the off arm. Fifteen protocol tests passed before inference, including a check that complete request bodies differ only in the reasoning object. Provider/model capability and price snapshots were fetched before freezing.

#### Sample and interpretation

The 100 inputs were selected deterministically from the existing 500-input development sample using coarse domain coverage, distinct contexts/codes and training-frequency balance. Selection did not read earlier predictions or correctness. It used gold code identity/frequency metadata to encourage diversity, but those labels never entered model requests.

The pilot spans **100 distinct normalized context groups, 100 distinct ICD codes, 100 distinct SNOMED codes and all 20 available coarse clinical domains**. ICD buckets are 23 unseen / 52 rare / 25 common; SNOMED buckets are 48 / 46 / 6. The previous outcome-independent near-duplicate audit also places these inputs in 100 separate components. These are not verified patient identities.

This purposively diverse pilot is not a prevalence-weighted estimate of the 500 cases. Its percentages should not be directly compared with the previous full-500 agent or selector scores. It remains previously exposed development evidence, not untouched confirmation.

#### Accuracy and paired changes

| Metric | Off | High | High minus off |
|---|---:|---:|---:|
| ICD exact match | 60/100 (60%) | 58/100 (58%) | -2 pp |
| SNOMED exact match | 73/100 (73%) | 63/100 (63%) | -10 pp |
| Headline minimum | 60% | 58% | -2 pp |
| Both codes correct | 45/100 | 45/100 | 0 |
| Valid model selections | 100/100 | 87/100 | -13 |
| Truncated/invalid terminal outputs | 0/100 | 13/100 | +13 |

| System | Corrections | Regressions | Net | Paired context-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| ICD | 7 | 9 | -2 | -10 to +6 pp |
| SNOMED | 2 | 12 | -10 | -17 to -3 pp |

The headline interval is **-10 to +6 pp**. The SNOMED decline is also present among valid high-mode finishes, rather than being explained entirely by truncation. Intervals use 5,000 paired complete-context resamples and are descriptive development intervals without multiplicity adjustment.

The correct code was in the frozen pool for 81 ICD inputs and 88 SNOMED inputs. Selection accuracy conditional on gold visibility was **60/81 (74.07%) → 58/81 (71.60%) ICD** and **73/88 (82.95%) → 63/88 (71.59%) SNOMED**. Reasoning could not fix the 19 ICD and 12 SNOMED missing-candidate cases without changing retrieval.

#### Long-tail results

Buckets use the same allowed 3,423-row example store as preceding studies. Unseen means no permitted example for that code, not absence from the dictionary or model pretraining.

| System | Bucket | Inputs | Off correct | High correct |
|---|---|---:|---:|---:|
| ICD | Unseen | 23 | 11 (47.8%) | 9 (39.1%) |
| ICD | Rare, 1–5 | 52 | 33 (63.5%) | 33 (63.5%) |
| ICD | Common, 6+ | 25 | 16 (64.0%) | 16 (64.0%) |
| SNOMED | Unseen | 48 | 31 (64.6%) | 27 (56.3%) |
| SNOMED | Rare, 1–5 | 46 | 38 (82.6%) | 33 (71.7%) |
| SNOMED | Common, 6+ | 6 | 4 (66.7%) | 3 (50.0%) |

There is no observed unseen-code benefit in this pilot. Small strata limit precision, especially the six common SNOMED inputs. Bucket-specific paired counts/intervals are retained in `results.json`.

#### Truncation does not explain every regression

All 13 invalid high outputs ended with `finish_reason=length`; no high output that ended normally failed schema/candidate validation. The off arm ended normally on every input. No invalid response was rerun with a larger cap.

Splitting the paired changes by whether high mode returned a valid answer gives:

| Subgroup | ICD corrections / regressions | ICD net | SNOMED corrections / regressions | SNOMED net |
|---|---:|---:|---:|---:|
| 87 valid high finishes | 6 / 4 | +2 | 1 / 9 | -8 |
| 13 high truncations, fallback scored | 1 / 5 | -4 | 1 / 3 | -2 |
| All 100 | 7 / 9 | -2 | 2 / 12 | -10 |

This is a diagnostic partition determined by the observed high response, not a randomized subgroup effect. It suggests some ICD distinctions benefit from additional thinking, while valid SNOMED choices still regress. Increasing the output cap might reduce failures; it is not established that doing so would repair the semantic-choice regressions or improve overall accuracy.

Among truncated inputs, the off arm was correct on six ICD labels and four SNOMED labels. The deterministic high fallback happened to preserve one of each and correct one previously wrong label in each system. Fallback successes are not attributed to high-mode clinical reasoning.

#### Case-level observations

Valid high-mode corrections include selecting the expected ICD child for generic respiratory failure, type 1 diabetes without documented complications, and shingles. These are examples of useful code-completion or specificity decisions.

Valid high-mode regressions include:

- **Posterior dislocation:** selecting a shoulder-dislocation disorder instead of the dataset's generic morphology concept, based on contextual rotation wording.
- **Onychomycosis:** selecting a toenail-specific concept using context, while gold is the general onychomycosis concept. This is a granularity disagreement; contextual specificity can be plausible without matching the annotation.
- **Open wounds to the left lower extremity:** selecting the encounter-unspecified ICD stem rather than the dataset's initial-encounter code.
- **Thrush:** replacing the oral-thrush gold with generic candidiasis because an oral site is not explicit in the target wording.

These examples show that more reasoning can move target scope, granularity and default interpretation in either direction. Stored final explanations are evidence of the stated justification, not reliable access to internal reasoning or independent clinical adjudication. No gold-specific corrections were added to the prompt.

#### Cost, latency and reasoning verification

| Metric | Off | High |
|---|---:|---:|
| Reported API cost | $0.05076399 | $0.13340507 |
| Median response latency | 1.40 s | 10.54 s |
| p95 response latency | 1.83 s | 44.82 s |
| Reported completion tokens, total | 9,948 | 300,988 |
| Reported reasoning tokens, total | 0 | 303,874 |
| Reported reasoning tokens, median | 0 | 1,785 |
| Responses with positive reasoning-token count | 0/100 | 100/100 |

High costs **2.63 times off** and has **7.51 times the median response latency** in this run. These are observed API timings with four concurrent requests, not controlled deployment throughput estimates. The first successful off/high compatibility checks are included in the 200 requests, not extra samples. Total reported cost is **$0.18416906**; conservative uncertain billing allowance is zero. Credit-purchase fees are excluded.

The provider reported a reasoning-token subtotal exceeding the reported completion-token total on **34 high responses**. Raw counters were preserved; reasoning is not added to completion tokens for a second charge. Two read-only generation-metadata examples are retained, including an affected response, but they do not fully resolve the counter inconsistency. Cost conclusions use the reported dollar charges, not a reconstructed bill from inconsistent token subtotals. Every high response also contains reasoning metadata, independently confirming activation. No hidden reasoning transcript is reproduced in this report.

#### Control stability and limits

On these same 100 inputs, the previous 2,048-token B run scored 60 ICD / 74 SNOMED; the fresh 8,192-token off control scores 60 / 73. Code agreement is 94/100 ICD and 98/100 SNOMED. This documents some repeated-call variation despite the requested temperature and seed, and is one reason to use the matched fresh control for the primary comparison.

The pilot tests one reasoning level and one bounded completion allowance. It does not establish results for low effort, larger high-mode budgets, reasoning during search, different models/providers, or learned selectors. Its context/candidate evidence is fixed and already exposed during development. No 95% target or broader generalization claim follows from this sample.

#### Decision

The frozen expansion rule required at least a two-point headline gain, no per-system accuracy loss, and no increase in invalid/truncated outputs. The pilot fails all relevant improvement/reliability conditions. **No larger 500-input high-mode run was started.**

Keep reasoning off for the existing condition-B selector. Resume the targeted terminal-rescue experiment for agent budget failures, then contextual/contrastive training-memory work. If continuing reasoning research, compare low effort or a separately controlled larger high-mode cap before broad deployment; a larger cap is not justified as an automatic accuracy fix by this pilot.

#### Verification and artifacts

Independent verification passed: frozen source/evidence hashes, sample uniqueness/domain coverage, dictionary/candidate validity, training-example provenance, raw response/prediction identity, requested/returned model/provider, high positive/off zero reasoning usage, output caps, metrics/buckets, paired corrections/regressions, response uniqueness and reported billing. The previous agent and all successful prior experiment artifacts remain unchanged. This run required no SSH connection, GPU allocation or retrieval service.

Source entry points: `reasoning_eval.py`, `test_reasoning.py`. Artifacts are under `artifacts/reasoning100_v1/`: exact config, selection manifest, source hashes, provider/model snapshots, all 200 paid responses and predictions, paired results, diagnostics/changed cases, context audit, token-counter examples, independent verification, protocol and execution logs.

```bash
python3 -m unittest independent_agent.test_reasoning independent_agent.test_selector500 -v
python3 -m independent_agent.reasoning_eval verify
python3 -m independent_agent.reasoning_eval analyze
python3 independent_agent/artifacts/reasoning100_v1/diagnostics.py
python3 independent_agent/artifacts/reasoning100_v1/verify.py
```

Verification and analysis make no inference calls. `run` resumes incomplete work while retaining and reusing completed paid responses.


---

## Appendix 16: TERMINAL_RESCUE_RESULTS.md

Historical source: `independent_agent/TERMINAL_RESCUE_RESULTS.md` (included below).

### Guided finalization only after agent budget failure

**The fallback-only guided terminal step improves this controlled continuation replay while preserving every accepted agent decision.** ICD rises from **69.2% to 70.6%**, SNOMED from **69.2% to 71.4%**, and both-correct from **53.0% to 53.6%**. All 35 fresh terminal requests produced valid selections, costing **$0.02710027**. The original 465 accepted finishes were unchanged.

This is a targeted workflow improvement on previously evaluated development traces. It is not a fresh end-to-end run or independent confirmation on new inputs. The trigger was motivated by earlier analysis of these cases; new terminal requests validate the implemented route and continuation behavior, not an untouched generalization claim.

#### Intervention and implementation

The source is the saved 500-input agent with pretrained SapBERT for ICD and original lexical SNOMED retrieval. It permits five model responses and previously ended with 465 accepted model finishes and 35 budget/invalid-output fallbacks. Infrastructure failures are not eligible for rescue.

`terminal_rescue.py` implements the route:

1. If the source agent has an accepted model finish, return its prediction unchanged and make no new request.
2. If the five-response investigation budget is exhausted without an accepted finish, assemble its visible candidate cards, target context and visible training examples.
3. Make exactly one guided terminal request using condition B, reasoning disabled, and a candidate-constrained JSON Schema. No search tools are available.
4. Accept a valid selection from the visible pools. If terminal output is invalid, retain the exact original fallback. Do not retry merely to obtain a better answer.

This adds one response **after** the failed five-response workflow, so the maximum becomes six on routed inputs. It does not replace the old fifth response or demonstrate equal-budget superiority. Both codes are selected together. The route uses observable completion status and response count, never gold correctness, confidence or semantic challenge labels.

The terminal model is DeepSeek V4 Flash 0731 pinned to CoreWeave, temperature zero, seed 13, reasoning disabled and a 2,048-token completion cap. Its prompt is exactly condition B's basic selector plus coding/task guide. It receives the same uniformly enriched dictionary cards used in that study, not the old assistant's prediction or reasoning. Candidate IDs and examples are reconstructed from material visible before the saved deciding response; hidden initial100 pool entries and unread co-generated results are excluded.

#### Controlled replay and fresh requests

All **1,461 saved source model responses** were verified against their original conversation-prefix hashes and stored assistant messages. The corresponding paid investigation prefixes were replayed rather than reissued. This preserves the original search behavior and avoids running a retrieval service or charging again for unchanged investigation.

The 465 accepted predictions were passed through the actual routing function and preserved. The 35 failed prefixes received **fresh terminal requests**; outputs from the previous selector study and the post-hoc composite were not substituted. New generation IDs were checked for uniqueness and non-overlap with both source responses and corresponding old condition-B generations.

The new terminal evidence was reconstructed from source initial evidence, saved tool results and the dictionary. Its payload matched the previously frozen condition-B evidence. Evaluation labels were not loaded by preparation or runtime; labels were only hashed for provenance and later loaded for scoring.

#### Overall results

| Metric | Saved source agent | Fallback-only guided finalization | Change |
|---|---:|---:|---:|
| ICD exact match | 346/500 (69.2%) | 353/500 (70.6%) | +7 / +1.4 pp |
| SNOMED exact match | 346/500 (69.2%) | 357/500 (71.4%) | +11 / +2.2 pp |
| Headline minimum | 69.2% | 70.6% | +1.4 pp |
| Both codes correct | 265/500 (53.0%) | 268/500 (53.6%) | +3 / +0.6 pp |
| Outputs from deterministic workflow fallback | 35/500 | 0/500 | -35 |
| Original accepted predictions preserved | — | 465/465 | All unchanged |

All 1,000 final code predictions are dictionary-valid. Eliminating workflow fallbacks does not mean all routed predictions are correct: among the 35 routed inputs, the new terminal selector gets **7 ICD and 17 SNOMED** labels right, versus 0 and 6 from the original deterministic fallbacks.

| System | Corrections | Regressions | Net accuracy delta | Paired context-cluster 95% interval |
|---|---:|---:|---:|---:|
| ICD | 7 | 0 | +1.4 pp | +0.41 to +2.50 pp |
| SNOMED | 13 | 2 | +2.2 pp | +0.79 to +3.78 pp |

The headline interval is **+0.58 to +3.23 pp**. These are descriptive development estimates using 5,000 paired resamples of entire context groups, without multiplicity adjustment. Positive intervals on this reused sample do not erase the adaptive choice of intervention or establish performance on fresh cases. Both SNOMED regressions occur on previously correct deterministic fallback outputs; no accepted source decision regressed.

#### Long-tail results

Frequency buckets use the same filtered 3,423-row example store. Unseen means no permitted example for the gold code, not absence from the dictionary or LLM pretraining.

| System | Bucket | Inputs | Source correct | With rescue correct | Delta |
|---|---|---:|---:|---:|---:|
| ICD | Unseen | 114 | 67 (58.8%) | 68 (59.6%) | +0.88 pp |
| ICD | Rare, 1–5 | 261 | 188 (72.0%) | 191 (73.2%) | +1.15 pp |
| ICD | Common, 6+ | 125 | 91 (72.8%) | 94 (75.2%) | +2.40 pp |
| SNOMED | Unseen | 241 | 138 (57.3%) | 145 (60.2%) | +2.90 pp |
| SNOMED | Rare, 1–5 | 229 | 182 (79.5%) | 185 (80.8%) | +1.31 pp |
| SNOMED | Common, 6+ | 30 | 26 (86.7%) | 27 (90.0%) | +3.33 pp |

Unseen SNOMED has nine corrections and two regressions; its descriptive delta interval is +0.41 to +5.74 pp. Unseen ICD gains one case, with an interval of 0 to +2.78 pp. Other small bucket intervals also touch zero. The common SNOMED bucket contains only 30 inputs. This is a useful workflow gain, not a solution to long-tail semantic selection.

#### What remains difficult

The candidate sets are unchanged, so the original **70 ICD and 65 SNOMED exposure misses** remain. The 136 original accepted-model wrong selections with visible gold are also unchanged by design.

For the 35 routed inputs, gold was visible for 15 ICD and 28 SNOMED labels; the terminal step chooses 7/15 and 17/28 correctly. This leaves 19 visible-gold errors in the rescued subgroup, in addition to its missing-gold cases. Across all inputs, the replay has 155 visible-gold errors and 135 retrieval misses, versus 173 and 135 before. More reliable finalization reduces errors but does not resolve concept granularity, annotation conventions or target focus.

The preceding post-hoc composite reused older B outputs and suggested 70.8% ICD / 71.8% SNOMED. This controlled continuation uses fresh requests and scores 70.6% / 71.4%. The small difference is reported rather than selecting the better historical outputs. Identical requested temperature and seed do not guarantee identical generations.

#### Cost and latency

- **New paid requests:** 35, all successful and valid.
- **New reported API cost:** $0.02710027, below the shared $0.25 cap.
- **Uncertain billing allowance:** $0; no failed requests or retries.
- **Historical source-prefix model cost:** $0.85219598, reused without a new charge.
- **Historical prefix plus measured continuation model cost:** $0.87929625, an accounting estimate for the combined workflow, not the amount newly spent here.
- **Terminal-call median/p95 latency:** 1.77 / 2.76 seconds, applying only to the 7% of inputs that routed to rescue.
- **Measured continuation time averaged across all 500 inputs:** 0.131 seconds, an arithmetic estimate of added serial latency; no fresh full-agent end-to-end latency run was performed.

The additional reported model cost is about 3.2% of the historical source model cost. These figures exclude credit-purchase fees and do not establish deployment throughput or retrieval-service operating cost. All new requests used the exact requested model/provider and reported zero reasoning tokens. No SSH connection, GPU allocation or search service was needed.

#### Decision and next focus

Retain this **experimental fallback-only wrapper** for the next development comparisons. Preserve the original agent and previous backend controls as frozen references. The implementation is available through `finalize_after_budget`; it has not silently changed an existing default or the team's separate training pipeline.

The next semantic experiment should improve contextual and contrasting training-example memory while holding retrieval fixed. That targets wrong choices among visible candidates, including the unchanged 136 original accepted selection errors. A later fresh-input end-to-end check is needed before treating the workflow gain as confirmed generalization. The wider retrieval problem remains separate.

#### Verification and reproduction

Eighteen checks passed before inference, covering immutable accepted decisions, no request on accepted finishes, correct budget/infra routing, at most one extra call, preservation after invalid output, unseen-code rejection, input identity, nonreasoning terminal settings, and the original selector invariants.

The independent audit passed frozen source/evidence hashes, all original request/response-prefix identities, exact preservation of the 465 accepted predictions, the 35 observable budget routes, fresh unique generation IDs, candidate/dictionary validity, training-example provenance, raw responses/predictions, reasoning/model/provider settings, all metrics/bucket counts, correction/regression directions and billing.

Artifacts are under `artifacts/terminal_rescue500_v1/`: exact config/source hashes, reconstructed terminal evidence, all 500 routed results, all 35 new paid responses, metrics/intervals, independent verification, provider snapshot and execution logs. Source entry points are `terminal_rescue.py`, `terminal_rescue_eval.py` and `test_terminal_rescue.py`.

```bash
python3 -m unittest independent_agent.test_terminal_rescue independent_agent.test_selector500 -v
python3 -m independent_agent.terminal_rescue_eval verify
python3 -m independent_agent.terminal_rescue_eval analyze
python3 independent_agent/artifacts/terminal_rescue500_v1/verify.py
```

These analysis/verification commands issue no inference requests. `run` resumes missing work and reuses saved successful terminal responses without recharging them. Old source artifacts and accepted agent decisions remain unchanged.


---

## Appendix 17: MEMORY_CONTEXT_RESULTS.md

Historical source: `independent_agent/MEMORY_CONTEXT_RESULTS.md` (included below).

### Contextual training-memory index and controlled pilot

**The training-memory index is built and audited, but this lexical context/contrast prototype does not establish a reliable accuracy improvement. Do not promote it or expand the inference run to 500.** Context memory adds one correct label in each system on the 100-input pilot; contrast adds three SNOMED labels versus mention control but no ICD/headline gain. Some apparent changes occur on identical requests and cannot be credited to memory selection.

All **300 fresh requests** completed with valid model selections for **$0.15156114**, below the shared $2 pilot/optional-expansion cap. There were no transport failures, retries, invalid outputs or provider switches. Reasoning remained off. Independent provenance, request/output, metric and billing checks passed. No existing agent default or team training pipeline changed.

#### Delivered memory index

`context_memory.py` builds `artifacts/memory_context_v1/memory.sqlite` from the same **3,423 permitted training rows** as earlier experiments. It preserves each observed annotation, mention, full source context, input/group provenance, and dictionary description/semantic tag. It adds:

- An exact local context substring, with its source offset and a flag indicating whether the mention was located literally.
- Literal cue occurrences for laterality, acuity, stage/type, history/negation wording, severity, intent and selected description modifiers, each with a source field and offsets.
- Description cues absent literally from the example snippet. This records missing literal wording, not a judgement that the training annotation is invalid or clinically unsupported.
- Separate mention/full-context and local-context FTS indexes.

No model weights were trained. No synthetic labelled examples, evaluation-derived case memory, crosswalk rules or authoritative ontology relationships were added. This is a lexical/context memory prototype, not a learned contextual encoder or a full semantic graph. It can support a later learned retriever while retaining the same provenance.

#### Offline audit and one preflight revision

All 500 available inputs were prepared and mechanically audited before paid inference. Checks passed for every selected training example, source snippet and cue offset, visible-code membership, example/byte caps, and **1,000 mention-control rank comparisons** against the original training database. The control's BM25 ranking matches the existing entity-weight-5/context-weight-1 mention-query recipe, with the same candidate restriction.

Nine cases were reviewed without reading evaluation gold or prior predictions. An initial context-only scorer promoted nearby conditions, including a thrombosis case for a cellulitis target. Before any paid calls, a bounded training-IDF-weighted overlap with the annotated example's mention was added. The unanchored preflight artifacts are retained in `preflight_unanchored/`, with revision notes. No per-case gold overrides were introduced; the final scorer was frozen before inference and not tuned afterward.

The repeated audit still found limitations: unrelated matches through words such as “burden,” confusion around abbreviations, and different-code examples that are not necessarily useful clinical contrasts. The audit certifies provenance and implementation invariants, not semantic or clinical correctness.

#### Three matched conditions

| Condition | Selection of training examples |
|---|---|
| Mention control | Existing mention-query BM25 over training entity/full context |
| Context memory | Mention/context rank fusion, bounded literal-cue features and annotated-mention overlap |
| Context plus contrast | Context memory with promotion of a sufficiently relevant, overlapping different-code example |

All three share the same frozen target/context, candidate IDs/cards, condition-B guide, common explanation of literal memory metadata, example formatting, and output schema. Up to **four examples per system**, **1,800 UTF-8 bytes per example**, and **6,000 serialized bytes per system** are allowed in every arm. Actual counts can differ when a query has fewer eligible examples.

The contrast condition keeps the best contextual case and may promote a different-code training case when it shares target content words and reaches 60% of the best contextual score. It is a possible alternative interpretation, not an invented negative or an official conceptual relationship. Retrieval scores and condition labels are not shown to the model.

Every example's code must already belong to that input's frozen candidate pool. Memory therefore cannot insert a new gold candidate in this experiment. All model-request fields other than selected examples are identical across conditions, verified by protocol tests. The new matched control has regenerated capped examples and common metadata; it is not identical to the earlier B display or the full agent with terminal rescue.

#### Pilot sample and protocol

The pilot reuses the previously frozen diverse 100-input IDs: 100 distinct normalized context groups, 100 distinct codes in each system, and 20 coarse ICD-derived clinical domains. These IDs were selected before reasoning/memory outcomes, using domain/context/code diversity and frequency balance. They were not reselected based on this experiment's outcomes.

Buckets are ICD 23 unseen / 52 rare / 25 common, and SNOMED 48 / 46 / 6. Unseen means no permitted example for the code. The sample is deliberately diverse rather than prevalence-weighted; its scores should not be compared directly with earlier full-500 agent results. It is previously exposed development evidence, not untouched confirmation.

All arms use DeepSeek V4 Flash 0731 pinned to CoreWeave, requested seed 13 and temperature zero, reasoning disabled, a 2,048-token completion cap and one response. Four requests run concurrently with input-specific deterministic condition ordering. Twenty-two protocol checks passed before freezing. Successful paid responses are stored before parsing and reused on resume.

#### Pilot results

| Condition | ICD | SNOMED | Headline minimum | Both correct | Cost |
|---|---:|---:|---:|---:|---:|
| Mention control | 61/100 | 71/100 | 61% | 48/100 | $0.05010245 |
| Context memory | 62/100 | 72/100 | 62% | 49/100 | $0.04945626 |
| Context plus contrast | 61/100 | 74/100 | 61% | 50/100 | $0.05200243 |

| Contrast | System | Corrections | Regressions | Delta | Paired context-bootstrap 95% interval |
|---|---|---:|---:|---:|---:|
| Context versus mention | ICD | 3 | 2 | +1 pp | -3 to +5 pp |
| Context versus mention | SNOMED | 2 | 1 | +1 pp | -2 to +5 pp |
| Contrast versus context | ICD | 1 | 2 | -1 pp | -5 to +2 pp |
| Contrast versus context | SNOMED | 2 | 0 | +2 pp | 0 to +5 pp |
| Contrast versus mention | ICD | 1 | 1 | 0 pp | -3 to +3 pp |
| Contrast versus mention | SNOMED | 3 | 0 | +3 pp | 0 to +7 pp |

Headline intervals are **-3 to +5 pp** for context minus mention, **-4 to +2 pp** for contrast minus context, and **-3 to +3 pp** for contrast minus mention. These use 5,000 paired complete-context bootstrap draws, without multiplicity adjustment. All primary headline estimates remain inconclusive.

The gold is present in the fixed pool for 81 ICD and 88 SNOMED inputs. Corresponding selection accuracy is 61/81, 62/81 and 61/81 for ICD; 71/88, 72/88 and 74/88 for SNOMED. The 19 ICD and 12 SNOMED exposure misses cannot be fixed by this memory-only selection study.

#### Long-tail results

| System | Bucket | Inputs | Mention | Context | Contrast |
|---|---|---:|---:|---:|---:|
| ICD | Unseen | 23 | 12 | 11 | 12 |
| ICD | Rare, 1–5 | 52 | 33 | 34 | 33 |
| ICD | Common, 6+ | 25 | 16 | 17 | 16 |
| SNOMED | Unseen | 48 | 30 | 30 | 31 |
| SNOMED | Rare, 1–5 | 46 | 37 | 38 | 39 |
| SNOMED | Common, 6+ | 6 | 4 | 4 | 4 |

Context does not improve unseen SNOMED and loses one unseen ICD case. Contrast gains one unseen SNOMED case versus mention control. These small counts do not establish a long-tail benefit, particularly with only six common SNOMED cases. All bucket-specific paired counts/intervals are retained in the results artifact.

#### Identical requests expose repeated-call variation

The exact full-request fingerprint includes the prompt, selected examples, schema, settings and candidates. Where two conditions generate identical requests, changes between their outputs cannot be attributed to different memory evidence.

- Context versus mention has **27 identical full requests**. Four of those produce different code pairs. One ICD correction and one SNOMED correction occur there; those equal the entire net +1/+1 aggregate advantage of context. On changed requests, the net change is zero in each system.
- Contrast versus context has **79 identical full requests**. Nine produce different code pairs. All two SNOMED corrections, and the net one-case ICD loss, occur on identical requests. On requests actually changed by the contrast rule, exact-match counts do not improve in either system.
- Contrast versus mention has **25 identical full requests**. Two of its three SNOMED corrections occur there. Its ICD net change is zero.

This is not a new stochastic sampling benchmark, but it demonstrates variation despite the requested temperature/seed. It materially limits attribution of the apparent memory gains. No advantageous historical output was selected or rerun to increase scores.

#### How much did retrieval change?

In the pilot, context changes the example order on 65 ICD and 62 SNOMED inputs, but changes the example set on only 39 and 25. It supplies more examples on 13 ICD and nine SNOMED inputs, with none losing examples. Total examples rise from **653 to 696** across the 200 input/system combinations, within the same maximum budgets. This is a coverage-and-ranking intervention, not a fixed-count pure reranker test.

Contrast changes order relative to context on only 14 ICD and 16 SNOMED inputs, and changes the example set on only **three ICD inputs and one SNOMED input**. Most of its intervention is reordering existing cases. It is too weak a change to treat as a test of an effective semantic contrast generator.

The full 500-input offline audit also passed before inference. Mention/context empty-example counts fall from 26 to six ICD and 22 to 14 SNOMED. Context selects a different ordered list on 364 ICD and 315 SNOMED inputs, but these are retrieval diagnostics, not unperformed full-500 accuracy results.

#### Case observations

- **Tinea with “oval” wording:** annotated-mention anchoring moves the tinea training case ahead of a PFO case in retrieval. The irrelevant PFO case remains available; this is partial lexical disambiguation, not certified conceptual equivalence.
- **Cellulitis:** context favors cellulitis examples over a neighboring thrombosis case. The contrast rule can re-promote the thrombosis example, illustrating that a different label is not automatically a useful contrast.
- **Minimal plaque burden:** context adds vascular cases but still retains irrelevant “burden” matches. A lexical local-context index does not solve the semantic gap by itself.
- **Renal cyst:** all conditions receive the same example evidence. The contrast run happens to select the generic gold while the other two choose Simple renal cyst, despite identical requests; this cannot be credited to contrastive memory. Its final explanation still describes the target as simple, so exact match alone does not demonstrate improved specificity reasoning.
- **Thrush:** changed contextual examples accompany corrections in both systems versus mention control, while other changed-memory cases regress, including an anemia other-specified/unspecified distinction and target-focus decisions. These are small descriptive observations, not causal or clinical adjudications.

#### Cost, latency and verification

Total new cost is **$0.15156114**. Condition median/p95 response latencies are **1.53/2.47 seconds** mention, **1.61/2.36 seconds** context, and **1.59/2.33 seconds** contrast. All 300 responses are valid, use the requested model/provider, and report zero reasoning tokens. There are no transport failures, retries or uncertain billing allowances. Credit-purchase fees are excluded. No GPU, SSH connection or live search service was required.

Independent verification passed all protected hashes, all indexed source records/snippets/cue offsets, exact training-input/context separation, unchanged base evidence and candidate pools, real example provenance, count/byte caps, raw response/prediction agreement, model/provider/reasoning settings, all metrics/buckets, paired corrections/regressions, unique paid generation IDs and billing. Frozen request tests independently establish that only examples vary across conditions.

#### Decision and next work

The frozen expansion rule required at least a two-point headline improvement, no per-system loss, no unseen loss of more than one case, and no additional invalid outputs. Neither variant passed. **No 500-input inference run was started**, and retrieval was not retuned after pilot results.

Keep the existing agent and experimental fallback-only rescue policy as the controls. Keep the audited memory store/provenance infrastructure, but do not adopt this lexical ranking or present the contrast result as an improvement in agent understanding.

A stronger next memory experiment needs genuinely different evidence: contextual semantic ranking of the training cases or a compatible learned candidate selector, with difficult real alternatives and training-only preparation. The first prerequisite is an offline relevance audit showing that it changes the case sets meaningfully and reduces neighboring-condition matches. If a later experiment again produces identical requests across conditions, reuse identical frozen requests within that input rather than paying for repeated identical calls; separately estimate model variation if needed. That is a future protocol improvement, not a retrospective alteration of this pilot.

Increasing graph size or appending more generic instructions is not supported by this result. Memory-assisted candidate expansion should remain a separate experiment because the current candidate universe was deliberately fixed.

#### Artifacts and reproduction

Source: `context_memory.py`, `memory_eval.py`, `test_context_memory.py`, `test_memory_eval.py`. Artifacts are under `artifacts/memory_context_v1/`: the training index, exact config/hashes, all 500 prepared evidence sets, offline audit and review notes, preserved unanchored preflight, 300 raw paid responses/predictions, paired pilot results, repeat-variation diagnostics, changed cases, independent verification and execution logs. `PROTOCOL.md` records the frozen rules and limitations.

```bash
python3 -m unittest independent_agent.test_context_memory independent_agent.test_memory_eval independent_agent.test_selector500 -v
python3 -m independent_agent.memory_eval verify
python3 -m independent_agent.memory_eval analyze --phase pilot
python3 independent_agent/artifacts/memory_context_v1/verify.py --phase pilot
python3 independent_agent/artifacts/memory_context_v1/diagnostics.py --phase pilot
```

Analysis/verification issue no inference requests. Successful paid results are retained and never repeated on resume. The expansion entry point checks the frozen pilot criterion before allowing additional requests.


---

## Appendix 18: SEMANTIC_MEMORY_RESULTS.md

Historical source: `independent_agent/SEMANTIC_MEMORY_RESULTS.md` (included below).

### Contextual semantic memory pilot

Semantic ranking changed the retrieved examples and sometimes improved their relevance, but did **not** improve selector accuracy on the existing 100-input development pilot. Keep the existing agent and fallback rescue unchanged. Do not expand this variant to 500 API inputs or integrate it as the default.

| Condition | ICD10CM | SNOMED CT | Both correct |
|---|---:|---:|---:|
| Lexical contextual memory, canonical cached control | 62/100 | 72/100 | 49/100 |
| E5 semantic contextual memory | 60/100 | 70/100 | 47/100 |

ICD had 2 corrections and 4 regressions; SNOMED had 0 corrections and 2 regressions. The paired descriptive 95% intervals for the accuracy differences were −7 to +2 percentage points for ICD and −5 to 0 for SNOMED. This is a small, previously exposed development sample; the result does not establish that semantic retrieval in general is ineffective.

#### What changed

The prototype embeds all 3,423 permitted training rows and the 500 existing evaluation inputs locally with `intfloat/e5-small-v2`, pinned revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`. It uses the official quantized ONNX model on CPU, attention-mask mean pooling, L2 normalization, 384 dimensions, and a 512-token maximum. No input was truncated. Model loading and all 4,896 unique embedding texts took about 31 seconds; clinical texts were not sent to an external embedding API. See the [official model card](https://huggingface.co/intfloat/e5-small-v2) for the encoder's documented usage. This is a general-purpose English encoder, not a clinical model.

Each training/query case has two views: the target mention alone and the target mention plus its exact local context window. Similarity is an equally weighted sum of their cosine similarities. Code labels and descriptions do not enter the embedding text. Candidate-code labels only mask which training examples may be shown, as in the lexical control. Search ranks all eligible training rows rather than a lexical shortlist.

The experiment preserves the original candidate cards and their ordering, coding guide, example formatting, four-example/6,000-byte per-system budgets, 1,800-byte per-example budget, model/provider, reasoning off, temperature 0, seed 13, and 2,048 output-token limit. The underlying SapBERT retrieval and original agent/rescue are unchanged. These scores compare two memory selector conditions, not a new end-to-end agent against the previous 500-input agent run.

#### Offline gate and execution

The gate and weights were recorded before retrieval review or accuracy analysis. Semantic retrieval changed example sets on 73/100 ICD inputs and 48/100 SNOMED inputs. Across all 500 offline inputs, those counts were 398 and 289. Ranking order changed on 88/100 and 76/100 pilot inputs.

A qualitative audit of top examples on 12 inputs × two systems used a fixed 0/1/2 relevance rubric with retrieval-system names hidden. It improved four comparisons and worsened none, gaining four total points and meeting the gate. The audit was performed by the coding assistant, not a clinical adjudicator; some contexts were already known. Several semantic examples remained off-topic or only broadly related. The threshold was met exactly; assigning zero rather than one to the marginal ventilatory/hemodynamic example would have failed it. This gate was weak evidence for relevance, not evidence of code accuracy.

The inference pilot used the same 100 inputs selected for the preceding memory experiment. Exact request fingerprints reuse one canonical historical response, in fixed context → mention → contrast priority, without examining correctness. All 100 lexical controls were cached. Semantic used 92 fresh calls and eight cached responses; seven semantic requests were identical to their lexical control and therefore shared its generation and prediction. The remaining cache hit matched another prior condition. This avoids the earlier confound of different generations for identical requests, though changed requests still contain generation variability even at temperature zero.

All 200 records were valid model outputs. New API spend was **$0.05527407**, with no transport failures or uncertain charges. Fresh semantic and historical control latency summaries should not be treated as a simultaneous provider speed benchmark. No GPU allocation or other process was stopped.

#### Long-tail and evidence coverage

| System / training-code frequency | Inputs | Lexical correct | Semantic correct |
|---|---:|---:|---:|
| ICD unseen | 23 | 11 | 11 |
| ICD rare | 52 | 34 | 31 |
| ICD common | 25 | 17 | 18 |
| SNOMED unseen | 48 | 30 | 30 |
| SNOMED rare | 46 | 38 | 36 |
| SNOMED common | 6 | 4 | 4 |

Gold codes were in the fixed candidate pool for 81 ICD and 88 SNOMED inputs; memory cannot solve the other 19 and 12 under the visible-code constraint. Semantic retrieval increased inputs with a displayed gold-labeled example from 53 to 58 for ICD and 42 to 44 for SNOMED. Correct predictions on those subsets stayed at 46 and 38 respectively. Thus additional gold-code demonstrations did not translate into more correct selections. These subsets differ between arms and are descriptive, not a causal intervention.

Unseen codes have no same-code training demonstrations by definition. No benefit appeared on those inputs. Training-case memory alone is therefore not supported as a solution to the unseen-code problem here.

#### Changed decisions

All statements below describe agreement with the dataset gold, not independent clinical coding adjudication. Raw evidence and responses are retained in `artifacts/semantic_memory_v1/decision_changes.json`.

| Mention | System | Change relative to gold | Observation |
|---|---|---|---|
| gynecologic primary | ICD | C57.9 → C80.1, correction | Replaced top osteoarthritis/“primary” word matches with carcinoma and gynecologic-origin context. The explanation still argued for a female-genital primary despite choosing the dataset's generic cancer code: agreement does not establish a sound explanation. |
| wet anus | ICD | K62.5 → K62.89, correction | New decision focused on discharge rather than nearby bleeding. No directly matching discharge demonstration was added; do not claim a proven mechanism. |
| intentional drug overdose | ICD | T50.902A → T50.902, regression | ICD examples were identical and in the same order; the joint request changed through SNOMED examples. This is not evidence that ICD retrieval itself caused the error. |
| wounds | ICD and SNOMED | T14.90XA → T14.90; 416462003 → 13924000, regressions | Gold-labeled demonstrations remained visible. The outputs shifted toward an ICD parent and SNOMED morphology rather than the dataset's disorder code. |
| kidney stones | ICD | N20.0 → N20.9, regression | Broader kidney-disease/urolithiasis examples accompanied an explanation incorrectly treating the explicit kidney site as unspecified. |
| thrush | ICD and SNOMED | B37.0 → B37.9; 79740000 → 78048006, regressions | Same per-system example sets, reordered. Gold oral-candidiasis examples remained first. Generic-vs-site-specific ambiguity remains and requires task-policy review before drawing a clinical conclusion. |

Both systems are predicted in one request. A decision change may reflect evidence in either system, example order, or generation variability. This experiment cannot isolate those effects per system.

#### Decision and next focus

Preserve the semantic index as optional experimental infrastructure and retain the existing agent and fallback rescue controls. Do not promote this retrieval rule or run its full 500-input inference extension.

The next useful design target is candidate discrimination: explicitly compare the strongest visible alternatives for target site, concept type, and level of specificity, and support that comparison with trusted terminology relationships or adjudicated task policy. The evidence supports testing that hypothesis, not assuming it will improve accuracy. First review the injury parent/child, SNOMED disorder/morphology, and ambiguous thrush cases against dataset policy. Avoid deriving universal coding rules from these eight outcomes or tuning repeatedly on this pilot. Any later accuracy claim needs an untouched evaluation sample, with separate reporting for retrieval misses and visible-gold selection failures.

#### Reproduction and validation

New modules: `semantic_memory.py`, `semantic_memory_eval.py`, and `test_semantic_memory.py`. Existing frozen experiment modules were not changed. Model/runtime requirements are in `semantic_embedding_requirements.txt`; CPU ONNX dependencies were installed in an isolated `/tmp/termmap-semantic-venv`, leaving the project's global NumPy unchanged. This prototype retrieves among precomputed query vectors; an online agent integration would additionally need query encoding and has not been implemented.

```bash
python3 -m unittest independent_agent.test_semantic_memory -v
python3 -m independent_agent.semantic_memory_eval verify
PYTHONPATH=. python3 independent_agent/artifacts/semantic_memory_v1/verify.py
```

All 13 unit checks passed. The independent verifier checks protected hashes; all training provenance and exact context spans; no training/evaluation input or group overlap; 3,423 training and 500 query vector dimensions and norms; label-free embedding-input reconstruction; all 500 candidate masks and display budgets; 200 raw prediction records; canonical response reuse; bucket/outcome totals; unique fresh generations; and billing. Frozen configuration, audit ratings, results, coverage counts, raw calls, and execution logs are retained under `artifacts/semantic_memory_v1/`.


---

## Appendix 19: PLATEAU_ANALYSIS.md

Historical source: `independent_agent/PLATEAU_ANALYSIS.md` (included below).

### Why progress has stalled, and what to establish next

We have reached a plateau in this particular unadapted agent and its prompt/memory variants. We have not established a ceiling for TermMap, a learned reranker, or a model with different capabilities. Continuing to add instructions or demonstrations without diagnosing the residual errors is weakly justified.

This review combines the problem statement, original project plan, completed experiment reports, a fresh offline audit of the 500-input experimental agent with terminal rescue, and primary literature. No new inference, training, evaluation-label edits, or team-pipeline changes were performed. New model cost is zero.

#### 1. Locate the plateau accurately

The full 500-input experimental agent with fallback rescue scores 353/500 ICD and 357/500 SNOMED (70.6% / 71.4%). The recent semantic-memory experiment scores 60% / 70% on a deliberately difficult 100-input selector pilot; its matched lexical-memory control scores 62% / 72%. That is a local negative result, not an end-to-end decline from 70.6% to 60%.

| Completed change | What it established | What it did not establish |
|---|---|---|
| Pretrained SapBERT for ICD | Initial dictionary gold recall@12 rose from 43.0% to 62.8%. Eventual exposure rose from 82.4% to 86.0%. | A corresponding 19.8-point final-accuracy gain. Later search already compensated for many initial misses. |
| Compact selector + coding guide | Some ICD benefit over its unguided compact control. | Reliable superiority over the investigation agent, or mastery of the task's annotation policy. |
| Lexical candidate distinctions | Some further ICD gains, with SNOMED losses. | Formal ontology reasoning; distinctions were generated from display words. |
| High reasoning, bounded at 8,192 tokens | More cost, 13 truncations, lower aggregate accuracy. | That all stronger models or all reasoning configurations are ineffective. |
| Terminal rescue | A targeted workflow gain while preserving accepted decisions. | A solution to the large semantic-choice and retrieval gaps. |
| Lexical/context/contrast memory | An auditable training-case store; apparent small gains were confounded by repeated-call variation. | A learned disambiguation rule. |
| E5 semantic memory | Different and sometimes more relevant cases, but no selector gain on the matched pilot. | Failure of contextual biomedical training or of every possible memory design. |

The central limitation across most recent tests is unchanged: the same base selector receives additional prose or examples, but its weights are not trained to distinguish the actual competing labels.

#### 2. The remaining errors are almost evenly split

The new audit reconstructs the rescue predictions against the source trace's pre-decision visible candidates. Rescue selects only from that same pool, so the visibility decomposition remains valid.

| Current experimental workflow, 500 inputs | ICD | SNOMED |
|---|---:|---:|
| Correct | 353 | 357 |
| Wrong, gold present in saved evidence | 77 | 78 |
| Wrong, gold absent from saved evidence | 70 | 65 |
| Perfect-choice ceiling using these saved pools | 430/500 = 86.0% | 435/500 = 87.0% |
| Current correctness when gold is present | 82.1% | 82.1% |

These are two separate engineering problems. A better selector cannot select a missing candidate. A higher-recall retriever does not ensure the selector will choose a newly supplied candidate.

To reach **95%**, ICD needs 122 additional correct predictions and SNOMED needs 118. Even perfect choice on every currently visible-gold error would still leave gaps of 45 and 40 cases respectively. The problem statement asks for **greater than** 95%; on this 500-input sample that would require at least 476 correct, one more than the 475 used for the arithmetic above. The current saved pools cannot support either target.

Under a candidate-constrained pipeline, accuracy decomposes as recall × conditional selection accuracy. For illustration, 98% recall and 97% conditional selection yield 95.06%. Both components need substantial improvement; this arithmetic is not a forecast.

Long-tail failures also span both components. Unseen SNOMED has gold present on 194/241 inputs but only 145 correct selections: 47 exposure misses and 49 visible-gold errors. Unseen ICD has 89/114 visible and 68 correct: 25 misses and 21 visible-gold errors. “Add more examples” cannot provide a same-code demonstration for a code with zero training examples.

#### 3. Existing variants share many failures

We compared saved predictions from six configurations: original text agent, ICD SapBERT variant, that variant with rescue, and compact selectors A/B/C.

| Gold-aware diagnostic upper bound | ICD | SNOMED |
|---|---:|---:|
| Rescue alone | 353 | 357 |
| Correct whenever rescue or any A/B/C selector is correct | 370 | 384 |
| Correct whenever any of all six configurations is correct | 381 | 396 |
| Wrong in every one of the six | 119 | 104 |

Even a perfect chooser among all six saved answers reaches only 76.2% / 79.2%. This does not bound a new model or retrieval system, but it argues against expecting a large gain from voting over these outputs. Actual routing would score below its gold-aware oracle unless it perfectly identified the right answer.

Confidence is insufficient as a rescue trigger by itself. Among 235 inputs with joint self-reported confidence ≥0.9, ICD is correct on 201 (85.5%) and SNOMED on 202 (86.0%). There are still 34 and 33 errors in that high-confidence group. The value is one heuristic for both systems and is not calibrated per system; it cannot establish 95% precision.

#### 4. “The agent does not understand the schema” is a hypothesis, not a complete explanation

There is evidence for concept-type and specificity problems. Among the rescue workflow's 143 SNOMED errors, 57 have a different stored semantic tag from gold; 31 of those occur with gold already visible. However, semantic-tag disagreement is not a clinical adjudication, and most errors do not involve different tags.

For ICD, only 16/147 residual errors have a direct ancestor/descendant relationship under the stored code-derived parent chain: 14 choose an ancestor of gold and two choose a descendant. The remaining 131 have neither relationship. A blanket rule to descend to a leaf would therefore address only part of the observed disagreement and could create new errors; the dictionary also contains accepted legacy categories.

Earlier trace inspection found that the agent explicitly inspected gold in 31 of 69 accepted visible-gold ICD errors and 40 of 67 such SNOMED errors. The current prompt already discusses target focus, modifiers, semantic tags, legacy codes, and example non-equivalence. The tools already expose ICD relatives and SNOMED names/tags. Repeating those resources is not a meaningful new intervention unless their content or representation changes.

At least four mechanisms remain plausible:

1. **Discrimination failure:** the target explicitly supplies a distinction, but the model chooses an incompatible alternative. The kidney-stone regression illustrates loss of an explicit site.
2. **Annotation-policy mismatch:** a clinically plausible interpretation disagrees with the supplied target's convention. History vs disorder, generic vs oral thrush, and morphology vs disorder need case-specific adjudication. More medical detail does not uniquely determine the intended label.
3. **Target/context interference:** nearby diagnoses or OCR noise shift attention away from the highlighted target. Context is necessary in some cases and distracting in others.
4. **Insufficient candidate evidence:** names and aliases may not explain the decisive distinction; verified definitions, relationships, or release-compatible notes could help particular cases.

We have not measured a clinically adjudicated prevalence for these four categories. Previous assistant reviews are useful hypotheses, not proof that the dataset is noisy or a numerical upper bound on its quality. The source problem statement itself acknowledges convention-dependent SNOMED granularity and gives a historical hypertension example mapped to the condition, supporting the need for task-specific policy rather than unmodified encounter-coding rules.

#### 5. Why memory was an incomplete intervention

The permitted diagnosis store contains 1,573 ICD rows across 714 codes and 1,850 SNOMED rows across 1,274 codes. There are 421 singleton ICD codes and 975 singleton SNOMED codes. At most five examples exist for 656/714 ICD codes and 1,251/1,274 SNOMED codes. This is sparse case coverage, particularly for varied contexts and subtle boundaries.

Semantic memory was an off-the-shelf general English encoder plus fixed cosine fusion. It was not trained on clinical equivalence or the dataset's competing labels. Its candidate mask could show only examples of already available codes, so it did not test candidate recovery. It also coupled the two systems in one selector response; changing SNOMED evidence could change an ICD decision even if the ICD examples stayed identical.

The offline gate was weak: four qualitative relevance gains, no losses, on 24 top-example comparisons. It passed exactly at the threshold, and one marginal rating would reverse the gate. Top-example relatedness also says little about the full example set's usefulness for the final choice. We should not describe that gate as strong validation of clinical semantic retrieval.

In the pilot, semantic memory increased gold-labeled demonstration coverage from 53 to 58 ICD inputs and 42 to 44 SNOMED inputs, yet selections correct on those subsets remained 46 and 38. That is a reason to stop treating demonstration retrieval as a sufficient substitute for learning the decision boundary, while recognizing the subsets differ and the counts are descriptive.

#### 6. What the literature actually supports

The relevant papers distinguish mechanisms that our small prototype did not reproduce:

- **SapBERT** learns entity representations through biomedical synonym alignment. This supports its retrieval role, not an expectation that its similarity alone resolves annotation policy. [Primary paper](https://arxiv.org/abs/2010.11784).
- **KRISSBERT** trains a contextual encoder using ontology-derived self-supervision and then links to contextual prototypes. Our unadapted E5 memory experiment did not implement that training. Its negative result does not refute the paper's mechanism. [Primary paper](https://aclanthology.org/2022.findings-emnlp.61/).
- **BeLink** instruction-tunes a generative model for constrained set-wise candidate selection. It separates learning the selection task from retrieving candidates and uses a compact answer format. This is a materially different intervention from adding examples to the unchanged model. [Primary paper](https://arxiv.org/html/2605.22501v1).
- **PILOT** combines neighborhood-aware retrieval with separate surface/context ranking and fusion. The relevant lesson is to measure these components separately rather than assume one similarity score captures both. [Primary paper](https://arxiv.org/abs/2608.04144).

These studies concern different datasets and resources. They motivate experiments; their reported gains are not predictions for TermMap. This review re-opened the primary sources above. The older ClinicalNLP citation in the project plan could not be fetched during this check, so its numerical claim is not used here.

#### 7. Recommended next step: diagnose the selector's limits before another feature

Do not run another broad prompt, graph, high-reasoning, or memory expansion. Prepare a small diagnostic that answers whether better discrimination is available from the current evidence.

**First, establish a reviewed diagnostic set.** Sample 60 system-level cases from the existing development traces: 30 visible-gold errors and 30 correct controls, balanced across systems and covering unseen/rare codes and distinct context groups. Selection within these strata should be deterministic and outcome-category based, not cherry-picked by a new model's success. It is an enriched diagnostic, not an estimate of deployment accuracy. Review target, context, and candidate meanings before revealing gold where practical; mark whether the supplied gold is clearly supported, policy-dependent, or unresolved. Qualified clinical/coding review is needed to call these adjudications; an assistant review must remain labeled as such. Do not silently relabel the benchmark.

**Then compare candidate discrimination on frozen evidence.** Use the current selector and one independently selected stronger model or compatible trained reranker. For each case, test the full saved candidate pool and a two-candidate diagnostic. In erroneous cases, the diagnostic pair is gold vs the original wrong choice; in correct controls, gold vs a deterministic strong alternative from the original ranking. Randomize option order deterministically and hold the guide, context, and candidate descriptions fixed. Initially evaluate systems separately to avoid cross-system evidence interference.

The two-candidate task deliberately uses gold to construct an easier, oracle-provided pool. Its score must never be presented as deployable mapping accuracy, new retrieval recall, or a fair full-pool benchmark. Its purpose is to identify whether the semantic distinction can be made at all when candidate overload is removed. Changing one joint prediction into separate systems also means this is a diagnostic rather than a controlled replication of the prior joint selector.

| Diagnostic result | Consequence |
|---|---|
| Both models choose well on two candidates, poorly on full pools | Prioritize learned reranking/shortlisting and candidate presentation; preserve recall while reducing distractors. |
| Stronger model improves both binary and full-pool decisions on clearly supported golds | Capacity or task adaptation is a credible target; quantify gains and regressions before training/integration. |
| Both struggle on pairs, with policy-dependent reviewer judgments | Clarify annotation policy or missing evidence before adding a graph or more demonstrations. |
| Errors are clearly resolved only by authoritative definitions/relationships | Test those specific resources on the affected slice against the same model without them. |
| No meaningful gain on clear cases | Reconsider the independent selector route rather than expanding an unproductive variant. |

Report corrections, regressions, validity, candidate-order stability, and outcomes on correct controls. Do not accept a variant for improving only an error-selected subset. A balanced diagnostic with 60 cases is for choosing a direction, not detecting a one- or two-point population gain. Predefine the next validation criterion before seeing its outputs; any later performance claim needs compatible untouched groups, because the existing 500 and 100 are development data.

#### 8. Where engineering effort should go if the diagnostic supports it

The strongest untested investment is **learning to rank candidate meanings against the highlighted mention and its context**, with difficult negatives and validation on unseen codes. This should score descriptions and context, not classify among a fixed list of training code IDs.

Use training-only groups to construct positive/negative candidate sets. Negatives should include retrieved confounders, ICD relatives where available, and relevant SNOMED concept-type/specificity alternatives. Avoid treating every clinically plausible neighboring concept as a proven negative when policy is unresolved. Vary candidate order and demonstration presence; prevent same-case or near-duplicate examples from leaking into training queries. Keep a grouped validation set and a code-disjoint diagnostic to reveal code memorization. Synthetic examples need separate validation and cannot replace adjudicated supervision.

The team plan already includes an adapted bi-encoder and trained reranker. Do not duplicate or modify that work without checking its status and split compatibility. The independent contribution can be this residual taxonomy, fold-safe hard-negative cases, and a fixed-evidence benchmark for the team's scorer. Its actual performance has not been inspected in this review.

Retrieval remains a parallel requirement for the 70 ICD and 65 SNOMED exposure misses. Audit these for absent aliases, abbreviation senses, target corruption, identifier lookup and code-family gaps. Test expansions against fixed candidate budgets and final accuracy; increasing recall by flooding the selector with distractors is insufficient. A graph should be introduced only for a defined operation, such as recovering a missing ICD relative or supplying a verified SNOMED relation, and compared with the same evidence without traversal. Training co-occurrence is not an authoritative hierarchy.

#### 9. Decision

Preserve the original agent and the experimental rescue result as controls. Pause broad variants of the unchanged selector. First distinguish clear, answerable selection failures from annotation-policy and evidence problems; then test a materially different selector. Prefer coordinating with the team's trained ranker over building another general memory mechanism.

There is measurable headroom, but no evidence yet that 95% is achievable on these labels with the available resources. A reasonable next success is a repeatable gain on clearly supported visible-gold choices without losing correct controls, together with measurable retrieval recovery. It is not another small score fluctuation on the reused 100 inputs.

#### Reproduction

`artifacts/plateau_audit_v1/analyze.py` reads all 3,000 saved predictions across six conditions, the source visibility diagnostics, dictionary and permitted training store. It writes `summary.json`, `residual_cases.json`, and `source_hashes.json`, verifies source bytes did not change, asserts the 500-input rescue totals and pool ceilings, and makes no network or inference calls.

```bash
python3 independent_agent/artifacts/plateau_audit_v1/analyze.py
```

Mechanical flags are not a complete manual error taxonomy; confidence is joint and heuristic; complementarity ceilings use gold; stored ICD parent links are code-derived rather than release-verified official semantics. No pooled score combines the purposively selected 100-input pilot with the full 500-input results.
