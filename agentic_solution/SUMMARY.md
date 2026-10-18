# TermMap agentic solution — experiment summary

## What we built

A standalone agent that maps a clinical mention and its chart context to one ICD-10-CM code and one SNOMED CT code. It searches the supplied dictionaries, inspects concepts, retrieves training examples, and validates its final answers. We added experimental dense retrieval, query recovery, preprocessing, candidate selectors, reasoning modes, fallback rescue, and contextual memory. The team's main pipeline was kept separate.

## Main results: 500 diagnosis inputs

| Approach | ICD accuracy | SNOMED accuracy | Both correct |
|---|---:|---:|---:|
| Original search/inspection agent | 67.4% | 70.4% | 53.2% |
| SapBERT retrieval for ICD; original SNOMED search | 69.2% | 69.2% | 53.0% |
| A: compact selector over saved evidence | 65.4% | 69.6% | 48.8% |
| B: A plus coding/task instructions | 67.4% | 70.2% | 50.8% |
| C: B plus lexical candidate distinctions | 68.8% | 69.6% | 51.2% |
| ICD variant plus fallback-only guided finalization | **70.6%** | **71.4%** | **53.6%** |

The last row is a controlled continuation of saved agent traces: one terminal selection request was added only for 35 inputs that exhausted their investigation budget. All 465 accepted agent decisions were preserved. It is our strongest reported full-sample experimental workflow, but it is **not a fresh end-to-end confirmation**.

## Other experiments and outcomes

| Experiment | Result and decision |
|---|---|
| Initial smoke and model/tool checks | Agent scored 70%/70% on 20 inputs. On a separate 40 inputs, DeepSeek search/inspection scored 82.5%/80%, ahead of the smaller Qwen comparator. Small samples did not predict the later 500-input result. |
| Query recovery, mandatory comparison, extra prompt checks | On 80 inputs, none established an overall improvement over the 65%/71.25% baseline; some added failures or regressed correct choices. |
| Dense retrieval and selective deeper reading | ICD-only change reached 70%/70% on 80 inputs. Selective SNOMED review did not help. The subsequent 500-input comparison remained inconclusive. |
| Retrieval cascade and preprocessing | On a separate 500-input retrieval benchmark, SapBERT achieved 64.6% ICD recall@12; the exact-first cascade achieved 80.6% SNOMED recall@12. Combined preprocessing did not improve those figures. These are candidate-recall scores, not mapping accuracy. |
| Higher reasoning | On 100 inputs, off scored 60%/73%; high scored 58%/63%, with 13 truncated outputs and 2.63× the cost. No expansion. |
| Lexical contextual/contrast memory | On 100 inputs, mention/context/contrast scored 61%/71%, 62%/72%, and 61%/74%. Repeated-call variation on identical requests undermined attribution of small gains. |
| Semantic contextual memory | On the same 100-input pilot, semantic memory scored 60%/70% against a cached lexical control's 62%/72%. It retrieved different cases but did not improve selection. No expansion. |

Percentages are ICD/SNOMED in that order. Compare within each experiment; sample sizes, evidence and protocols differ.

## Main inferences

- **Retrieval and selection both need work.** After fallback rescue, 70 ICD and 65 SNOMED errors had no visible gold candidate; another 77 and 78 chose incorrectly despite gold being visible. Perfect selection from those saved pools would reach only 86%/87%.
- **Better initial retrieval does not translate directly into final accuracy.** SapBERT exposed 99 additional initial ICD gold candidates in the agent comparison, but produced only nine additional correct ICD answers; later searches already repaired many original misses.
- **Reliable finalization was the clearest targeted gain.** Rescue improved the failed-budget cases while preserving accepted decisions. More instructions, thinking, or demonstrations did not establish a broad semantic improvement.
- **Long-tail remains difficult.** With rescue, unseen-code accuracy was 59.6% ICD and 60.2% SNOMED. Training memory cannot supply same-code examples for unseen codes.
- **The residual problem includes specificity, concept type, target focus and annotation conventions.** More medically plausible detail does not always match the dataset's intended label. These disagreements need review rather than blanket coding rules.
- **Next priority:** review clear versus policy-dependent errors, then evaluate a materially stronger or trained candidate selector on fixed evidence with correct controls. Coordinate with the team's trained reranker; target retrieval misses separately.

## Submission status

The Git submission contains code, tests, this summary and the consolidated report. Saved predictions and audit artifacts are retained in the private local archive. No independent-agent weight training or integration into the team's main pipeline was completed. The greater-than-95% target was **not achieved**. Results are development evidence with reused samples, not untouched final-test estimates.

See [FINAL_REPORT.md](FINAL_REPORT.md) for the complete record and [CODE_INDEX.md](CODE_INDEX.md) for implementation navigation.
