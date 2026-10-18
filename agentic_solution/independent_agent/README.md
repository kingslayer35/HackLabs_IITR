# Independent TermMap terminology agent

This prototype maps a supplied clinical mention and its chunk context to one ICD-10-CM code and one SNOMED CT US code. It runs independently of the team pipeline. Terminology search and validation run locally; the initial LLM backend is DeepSeek V4 Flash 0731 through OpenRouter, pinned to DeepInfra with provider fallback disabled.

## Setup and execution

Run commands from the `agentic_solution/` directory (one level above this package). Python 3.10 or later and SQLite with FTS5 support are required.

```bash
python3 -m pip install -r independent_agent/requirements.txt
```

Set `OPENROUTER_API_KEY` in a private `agentic_solution/.env`. The example environment file is `independent_agent/.env.example`. Credentials are loaded at runtime and are not written to request traces. Source data, `.env` files and generated artifacts are excluded by the root `.gitignore`.

Place the supplied files in `data/`:

- `hackathon_train.csv`
- `icd10cm_codes.xlsx`
- `SNOMEDCT_US_codes_and_descriptions.xlsx`

Prepare the dictionaries, example store and fixed smoke split once:

```bash
python3 -m independent_agent prepare
python3 -m unittest independent_agent.test_agent -v
```

Run the complete 20-input experiment, including evaluation:

```bash
python3 -m independent_agent smoke
```

The default is two concurrent inputs, six LLM responses per input, at most 2,048 completion tokens per request, temperature zero and seed 13. Reasoning is disabled. The spending stop is $2 of reported API usage; concurrent requests can finish after that stop is reached. Provider or network errors have at most two retries. Checkpoint files allow completed inputs to be reused without another API request.

The model and provider can be changed using `--model` and `--provider`, with a new `--run-dir`. Use `--provider ''` for a compatible local endpoint. Set `TERMMAP_API_BASE` to that endpoint's `/v1` base URL. The current client sends OpenRouter-specific fields, so compatibility with another server must be checked before switching. Hosted inference can vary even with fixed settings; saved predictions are the reproducible artifact, not a promise of bit-identical fresh API runs.

Recompute metrics from saved traces without calling the API:

```bash
python3 -m independent_agent evaluate
```

## Terminology and tools

SQLite holds concepts, aliases, relationships and permitted training examples. A full-text index searches all supplied dictionary descriptions. Multiple rewritten queries are combined by reciprocal rank and deduplicated into concept IDs. Exact names and literal codes are also supported. There is no dense retriever in this first version; the LLM can repair spelling or expand abbreviations in a new query.

The four investigation tools are:

- `search_concepts(system, queries, limit)` searches the entire supplied dictionary, not just codes observed in training.
- `inspect_concepts(system, codes)` returns names, synonyms, semantic tags and available ICD children.
- `get_related_concepts(system, code, limit)` returns ICD parents, children and siblings. For SNOMED it returns lexical qualifier variants with the same extracted semantic tag, explicitly labelled as derived hints.
- `find_training_examples(query, system, limit)` returns mention/context examples and labels from the permitted training split only.

The `finish` tool returns both code strings, a short explanation and heuristic confidence. A programmatic validator checks dictionary membership and confirms that codes have been returned by tools. Category ranges cannot be final predictions. If the agent cannot finish, the deterministic initial-search predictions are used and the fallback is recorded explicitly.

If an initial search has no candidates, the agent can still reformulate the query. For emergency fallback only, an empty ICD search uses R69 and an empty SNOMED search uses 404684003, after checking dictionary membership. These generic outputs are not confident semantic matches. The first run required this empty-search startup repair: 18 completed predictions were retained byte-for-byte and two inputs that had stopped before inference were resumed. The original and repaired source hashes are in `repair_history.json`. No prompt or model-setting changes were made from test-label feedback.

ICD relationships are derived from supplied code prefixes and chapter-range entries. They do not include official exclusions, inclusion notes, encounter rules or the Alphabetic Index. SNOMED synonym links are available, but official SNOMED IS-A and attribute relationships are not present in the supplied workbook. This is a terminology graph with limited structural information, not a complete imported SNOMED ontology. `description_order=1` is the first stored description, not a claim of official preferred-term status.

## Split and leakage protection

Only diagnosis-labelled source rows are used for examples and evaluation. Identical normalized chunk texts form groups; empty contexts use the normalized mention instead. Groups are split 70/15/15 with seed 13. The smoke set is the first 20 paired test inputs in seeded hash order, without looking at model outcomes. Each has a label for both systems.

All test groups are absent from the example store. In addition, any candidate training group containing an identical normalized smoke mention is excluded. Every smoke chunk is compared exhaustively with every candidate training chunk using three-word-shingle Jaccard similarity. A whole training group is quarantined if similarity is at least 0.85. The completed SQLite example store is independently checked by tests.

The evaluator passes only `input_id`, `entity` and `chunk_text` to the agent. It does not pass the test `code`, `display`, `coding_system`, or gold mappings. Gold labels are read separately after predictions. The agent cannot access the test file through its tools. The entire supplied terminology dictionary is allowed; matching a test mention to a public/supplied dictionary name is not training-label leakage.

The audit guarantees these stated checks for the 20 smoke inputs, not the absence of every possible semantic relationship between charts. Near duplicates among other reserved development/test inputs have not yet been audited. The smoke inputs are considered consumed after this experiment; future model/prompt development should use development groups and a fresh confirmation set.

## Outputs and interpretation

Generated files are in `independent_agent/artifacts/`:

- `terminology.sqlite`: terminology, relationship and training-example stores.
- `terminology_stats.json`: counts, source hashes and structural limitations.
- `split_manifest.json` and `leakage_audit.json`: fixed grouping, selection and separation evidence.
- `smoke20.json`: the 20 held-out inputs and evaluation-only gold labels.
- `runs/smoke20_v1/config.json`: frozen model settings, source hashes and split hash.
- `runs/smoke20_v1/predictions.csv`: one row per input with both code strings.
- `runs/smoke20_v1/metrics.json`: accuracy, initial/ever-retrieved recall, frequency buckets, API usage and latency.
- `runs/smoke20_v1/errors_and_predictions.csv`: all 40 system-level comparisons and short model explanations.
- `runs/smoke20_v1/traces/`: prompts, tool calls, retrieved evidence, responses and usage for each input. These contain proprietary clinical text and belong only in the authorized prototype environment.

There is no `chunk_id` column in the supplied training CSV. The prototype creates an `input_id` from the mention and context; its prediction export uses that identifier. A production adapter must preserve the official evaluation file's chunk IDs.

Accuracy is exact code equality, reported separately for each system and jointly for paired inputs. The headline is the lower system accuracy. Search-only top-1 predictions provide a baseline on the same inputs. Confidence is not calibrated. Wilson intervals are descriptive; 20 inputs are too few for a general performance claim, and intervals do not account for dependencies between inputs sharing a chunk. Report actual failures and invalid labels rather than silently repairing them.

The first smoke run scored 14/20 (70%) on each system, with both codes correct on 11/20 (55%). It made 54 successful API requests at a reported cost of $0.01059978. One input exhausted the turn budget and used its search baseline; it is included in these scores. See [SMOKE_RESULTS.md](../FINAL_REPORT.md) for the aggregate report. To experiment with changed source code, use a new run directory rather than rewriting the original run provenance.

## Matched model and tool comparisons

The separate comparison experiment uses 40 fresh paired inputs, one per normalized chunk group, excluding the original smoke groups and their near duplicates. It creates a separately filtered training-example database and caches the same flat starting evidence for all six conditions. The shared terminology store and original smoke outputs remain unchanged.

```bash
python3 -m independent_agent.comparisons prepare
python3 -m unittest independent_agent.test_agent independent_agent.test_comparisons -v
python3 -m independent_agent.comparisons run --workers 6 --max-cost 0.50
python3 -m independent_agent.comparisons score
```

Both DeepSeek V4 Flash 0731 and Qwen3.5-9B are evaluated as a one-request selector, an agent with search/inspection/training-example tools, and an agent additionally allowed relationship tools. The first condition can select only from initial evidence. Agent conditions can finish immediately or use up to five responses, with the final response exposing only `finish`. Parent, child and leaf fields are absent from starting evidence and ordinary inspections in every condition; relationships are exposed only through the graph tool.

DeepSeek stays on DeepInfra. After repeated upstream Qwen rate limits, only pending/failed Qwen requests switch to Venice. Successful completed predictions are retained byte-for-byte, including three Qwen results already served by DeepInfra. The provider of every response is recorded; this is not a strictly controlled provider comparison. New Venice requests omit the unsupported sampling seed; temperature is zero for both models and data selection uses seed 13. Hosted runs are not guaranteed bit-identical.

Artifacts are in `artifacts/comparisons_v1/`. The `transport_attempts/` directory preserves the interrupted provider attempt, and `retained_runs.json` records hashes of successful predictions kept during recovery. A conservative five-cent allowance for interrupted in-flight usage is reserved within the 50-cent cap. Completed results are cached; the model, prompts, split and example database are frozen in the experiment configuration. The comparison set is consumed after these runs and should not become a prompt-tuning set for future confirmation claims.

## DeepSeek query recovery and candidate comparison

The separate `recovery_v1` experiment evaluates the existing search/inspection agent,
mandatory contextual query recovery, and query recovery followed by candidate comparison.
It uses DeepSeek V4 Flash 0731 on DeepInfra only. Each condition receives identical initial
evidence and the same filtered example store, with at most five model responses.

`recover_queries` receives standard clinical names and up to two alternative phrasings per
system from the controller. It retains the original mention, searches the local dictionaries,
and returns up to 12 candidates per system with query provenance. Supporting detail quotes
must occur in the visible input; this text-presence check does not prove clinical entailment.

`compare_candidates` displays descriptions, aliases and literal distinguishing description
words for two to six previously retrieved codes. It uses local dictionary data, not another
LLM call or a generated medical definition. In the comparison condition, the final selected
code must have been compared when at least two valid candidates were available. Comparisons
for the two systems can be batched into one model response. Relationship tools are disabled.

```bash
python3 -m independent_agent.recovery_eval prepare --count 80
python3 -m unittest independent_agent.test_agent independent_agent.test_comparisons independent_agent.test_recovery -v
python3 -m independent_agent.recovery_eval run --workers 4 --max-cost 0.50
python3 -m independent_agent.recovery_eval score
```

The 80 inputs are fresh paired reserved-test inputs from distinct context groups. All prior
60 evaluation inputs, their groups, normalized mentions and near-duplicate chunks are excluded.
The example store excludes matching mentions, groups and near-duplicate chunks. Gold labels
remain separate from the agent's input. Outputs, leakage audit, per-response usage ledger and
source/data/protected-artifact hashes are in `artifacts/recovery_v1/`. Completed model runs
are cached; only infrastructure failures may be retried, with failed traces preserved.

This tests complete mandatory workflows, including their prompt and minimum-action differences.
The common response cap does not imply equal numbers of calls or identical generated recovery
queries. Scores should be compared within this fresh experiment, not against the earlier 40 inputs.

The completed run found no aggregate accuracy improvement: headline scores were 65% for the
baseline, 63.75% for recovery and 57.5% for recovery plus comparison. Total reported usage was
$0.17003868. See [RECOVERY_RESULTS.md](../FINAL_REPORT.md) for paired changes, retrieval
diagnostics, workflow failures and limitations.

## Selective recovery on the existing 80-input development set

`selective_eval.py` reuses the exact inputs, filtered example database, initial evidence and
80 byte-identical baseline traces from `recovery_v1`. It runs only the new DeepSeek V4 Flash
configuration; the saved baseline is never sent to the API again.

The existing search/inspection policy gains an optional, system-specific `recover_queries`
tool. The controller must state one of three reasons: an observed empty search, a mismatch
between candidate meanings and the mention, or an unresolved abbreviation. Choosing between
two plausible candidates of differing specificity is not a recovery reason. Empty-search
claims are checked against search history; the other reasons are model judgments rather than
calibrated thresholds or deterministic guarantees. No labels or baseline correctness enter
the trigger.

Recovery keeps the original candidates, adds a search over the original mention plus up to
three generated clinical names, and returns up to 12 candidates for the requested system.
Names are deduplicated after normalization. One successful recovery per system is allowed,
only during the first three responses, leaving an ordinary investigation response before
the fifth, finish-only response. Recovery and comparison are never required to finish;
candidate comparison and relationships are unavailable.

```bash
python3 -m independent_agent.selective_eval prepare
python3 -m unittest independent_agent.test_agent independent_agent.test_comparisons independent_agent.test_recovery independent_agent.test_selective -v
python3 -m independent_agent.selective_eval run --workers 4 --max-cost 0.25
python3 -m independent_agent.selective_eval score
```

Outputs are under `artifacts/selective_v1/`. The new usage ledger and execution accounting
contain only newly incurred API usage. Summary metrics also retain the baseline's historic
cost and latency for comparison. All earlier artifacts are protected by content hashes.
These 80 cases are explicitly a development benchmark. Policy changes and hosted inference
variation may change predictions even when recovery is unused, so gains on those cases cannot
be attributed to retrieved recovery evidence.

The completed selective run scored 53/80 (66.25%) on each coding system, compared with the
saved baseline's 52/80 ICD and 57/80 SNOMED. Recovery succeeded on one input and did not itself
retrieve its gold code. New reported usage was $0.03490968, with zero baseline API calls.
See [SELECTIVE_RESULTS.md](../FINAL_REPORT.md) for activation, paired changes and limitations.

## Offline baseline selection audit

```bash
python3 -m independent_agent.analyze_selection
```

This reads the saved 80-input baseline and private review annotations, reconstructs the
28 system-level errors where the gold was available before the final answer, and exports
case-level evidence and aggregate counts under `artifacts/selection_audit_v1/`. It makes no
network or inference calls. The original baseline and all prompts remain unchanged.
See [SELECTION_ERROR_ANALYSIS.md](../FINAL_REPORT.md) for the review, annotation-policy
limitations and the proposed final-selection experiment.

## Prompt-only example-equivalence experiment

`selection_prompt_eval.py` appends one generic final-selection check to the saved baseline's
system prompt: verify concept, modifier and status equivalence before treating a similar
training example's code as deciding evidence. The existing finish evidence field identifies
the decisive target evidence. The check adds no sample-specific mappings, tools, mandatory
investigation or required model responses.

```bash
python3 -m independent_agent.selection_prompt_eval prepare
python3 -m unittest independent_agent.test_selection_prompt -v
python3 -m independent_agent.selection_prompt_eval run --workers 4 --max-cost 0.20
python3 -m independent_agent.selection_prompt_eval score
```

The same 80 development inputs, initial evidence, filtered example store and saved baseline
are copied byte-for-byte. Only the new prompt condition calls DeepSeek V4 Flash on DeepInfra.
The baseline tool schemas, five-response maximum, temperature, seed and token cap are unchanged.
The model can still change its search queries in response to the prompt; this is an end-to-end
prompt experiment, not fixed-candidate replay. Outputs and the new usage ledger are under
`artifacts/selection_prompt_v1/`. Existing evaluations and the selection audit are protected
by hashes. This development experiment follows analysis of the same 80 cases.

The completed prompt run scored 50/80 ICD (62.5%) and 55/80 SNOMED (68.75%), versus the
saved baseline's 52/80 and 57/80. It fixed one reviewed selection error and regressed seven
previously correct system-level predictions. New reported usage was $0.03266994, with no
baseline API calls. See [SELECTION_PROMPT_RESULTS.md](../FINAL_REPORT.md).

## ICD-only backend and selective SNOMED paging ablations

The completed `backend_ablation_v1` experiment reuses the same 80 development inputs and
both cached controls from `dense_agent_v1`. The ICD-only arm retains SapBERT ICD search/paging
and restores original lexical SNOMED search. The other arm retains both new backends and
tests one optional controller-driven SNOMED review page when a proposed finish has joint
confidence below 0.85 during responses 1–3. Both use the same per-input provider as the
cached combined-backend run and the same five-response maximum.

ICD-only scored 56/80 on each system (70% headline). Selective SNOMED review scored 52/80
ICD and 53/80 SNOMED (65% headline). Review exposed one new gold but did not select it;
16 inputs were reviewed, with zero wrong-to-correct and two correct-to-wrong proposed/final
SNOMED changes. No default agent or backend was replaced. Total new recorded usage was
$0.18696374 for 160 new runs; all previous predictions were retained.

```bash
python3 -m unittest independent_agent.test_backend_ablation independent_agent.test_dense_agent -v
python3 -m independent_agent.backend_ablation_analysis
python3 independent_agent/artifacts/backend_ablation_v1/analysis/verify.py
```

These scoring/verification commands make no inference calls and require the private saved
artifacts. Runtime preparation and cached/resumable execution are available through
`python3 -m independent_agent.backend_ablation prepare` and `run --arm icd_only` or
`run --arm selective_snomed`. New inference needs the authorized retrieval service on
loopback port 18473; it has been stopped after evaluation. Preserve the existing run directory
and frozen provenance when designing another experiment.

See [BACKEND_ABLATION_RESULTS.md](../FINAL_REPORT.md) for corrections/regressions,
visibility errors, confidence intervals, CPU Slurm execution, provider limitations and the
recommendation for a larger paired development evaluation.

## Diverse 500-input comparison with long-tail reporting

`diverse500_v1` compares **text search for both systems** with **SapBERT for ICD and original
text search for SNOMED** on the same 500 diagnosis inputs. Both configurations use DeepSeek
V4 Flash 0731 on CoreWeave, identical examples and task prompt, and five responses maximum.
All 1,000 runs completed for $1.64027022. Original scored 67.4% ICD / 70.4% SNOMED; the
ICD variant scored 69.2% / 69.2%. Paired uncertainty includes zero. No default was replaced.

The set has 336 unique ICD codes, 431 SNOMED codes and 382 normalized context groups.
The common filtered example store has 3,423 rows. Primary unseen/rare/common counts are
114/261/125 ICD and 241/229/30 SNOMED; unseen means no permitted example, and rare means
one to five. Golds and bucket labels stay outside runtime tools. The sample is development
evidence, with no prior-agent input-ID overlap but some prior chart/retrieval-benchmark overlap.

```bash
python3 -m unittest independent_agent.test_diverse500 -v
python3 -m independent_agent.diverse500_analysis
python3 independent_agent/artifacts/diverse500_v1/analysis/verify.py
python3 independent_agent/artifacts/diverse500_v1/analysis/near_duplicate_sensitivity.py
python3 independent_agent/artifacts/diverse500_v1/analysis/split_frequency_sensitivity.py
python3 independent_agent/artifacts/diverse500_v1/analysis/workflow_diagnostics.py
```

These analysis commands read saved private artifacts and make no inference calls. Preparation
entry points are `python3 -m independent_agent.diverse500_prepare select` and `evidence`.
`python3 -m independent_agent.diverse500_eval freeze` freezes all source/data/evidence hashes;
`run` executes only pending inputs, retaining successful results. Transport-failed attempts can
resume by replaying saved paid responses locally after verifying the exact conversation, without
charging for them again. The shared API cap was $2 and four concurrent input/configuration runs.
The loopback retrieval service has been stopped after evaluation; new requests require it to be
started again. Preserve this frozen experiment rather than editing it for a new method.

See [DIVERSE500_RESULTS.md](../FINAL_REPORT.md) for long-tail tables, macro code accuracy,
context-cluster intervals, failure analysis and the independent verification. Primary results and
all traces are in `artifacts/diverse500_v1/`; each arm also has a consolidated `predictions.csv`.

## Fixed-evidence selector comparison

The three-condition selector experiment is complete on all 500 inputs: compact selection,
coding/task guidance, and guidance plus lexical candidate distinctions. All 1,500 requests
completed for $1.17187671 with no invalid outputs or transport failures. Headline scores are
65.4%, 67.4% and 68.8%, versus 69.2% for the saved ICD-backend agent. Guidance helps the
compact selector, but none of these conditions replaces the agent. Terminal selection for
budget failures is the next targeted experiment; a post-hoc fallback-only composite is
documented separately from the primary comparisons.

See [SELECTOR500_RESULTS.md](../FINAL_REPORT.md) for paired changes, long-tail scores,
selection accuracy, workflow subgroups, cost, limitations and independent verification.
Artifacts are under `artifacts/selector500_v1/`. Local package-cache cleanup before inference
reclaimed approximately 4.1 GiB without deleting project data or saved experiment traces.

## Reasoning off versus high pilot

The paired condition-B reasoning pilot is complete on 100 diverse inputs, with a common
8,192-token completion cap. Off scores 60% ICD / 73% SNOMED; high scores 58% / 63%.
High produced 13 truncated outputs versus none for off, and valid high-mode SNOMED
selections also regressed. All 200 API requests completed for $0.18416906; actual reasoning
activation, model/provider, frozen evidence, outcomes and billing passed independent checks.
The frozen expansion criterion was not met, so no 500-input high-mode run was started.

See [REASONING100_RESULTS.md](../FINAL_REPORT.md) for paired and long-tail results,
truncation diagnostics, cost/latency and token-counter limitations. Artifacts are under
`artifacts/reasoning100_v1/`. The default agent and prior experiment artifacts remain unchanged.

## Fallback-only guided terminal finalization

Controlled replay of all 500 ICD-backend agent prefixes preserved all 465 accepted decisions
and made 35 fresh condition-B terminal requests only for budget failures, with reasoning off.
All terminal requests finished validly for $0.02710027. ICD rose from 69.2% to 70.6%; SNOMED
from 69.2% to 71.4%; both-correct from 53.0% to 53.6%. This adds at most one response after
the five-response source budget, rather than demonstrating equal-budget superiority.

See [TERMINAL_RESCUE_RESULTS.md](../FINAL_REPORT.md) for paired and long-tail results,
prefix verification, measured continuation cost/latency and limits. The reusable
`terminal_rescue.finalize_after_budget` wrapper is an experimental option; original defaults
and the team's separate pipeline remain unchanged. This is a controlled continuation on
known development traces, not new-input end-to-end confirmation.

## Contextual training-memory index and pilot

The training-only memory index is built from 3,423 permitted rows with exact context/cue
provenance. Offline checks cover all 500 candidate sets and reproduce 1,000 mention-control
rankings. A matched 100-input pilot compares mention/context/context-plus-contrast memory,
with the same guide, candidate IDs and example budgets. Scores are 61%/71%, 62%/72%, and
61%/74% ICD/SNOMED; all 300 requests finished validly for $0.15156114.

Neither variant met the frozen expansion criterion. Repeat variation on identical requests
accounts for important apparent gains, and the contrast rule mostly reorders existing examples.
No 500-input inference run or default change was made. See
[MEMORY_CONTEXT_RESULTS.md](../FINAL_REPORT.md) for audit findings, paired/long-tail
results, repeated-request diagnostics and limitations. Artifacts are under
`artifacts/memory_context_v1/`.

## Research motivation

- [Generative relevance feedback](https://academic.oup.com/bioinformatics/article/42/2/btag011/8426181) motivates context-driven query reformulation.
- [BeLink](https://arxiv.org/abs/2605.22501) motivates selecting from candidate sets using a generative model.
- [RAG-Coding](https://arxiv.org/abs/2605.27377) motivates consulting structured terminology evidence.

This implementation is an independent prototype, not a reproduction of those papers. The supplied data and terminology remain subject to their original restrictions, and the model/provider terms apply to API inference. This run uses the user's explicit authorization for the external-API prototype; a later local-model deployment keeps inference inside the compute environment.
