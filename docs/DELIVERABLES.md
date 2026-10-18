# Submission checklist against TermMap.pdf

The PDF's “Submission Requirements” requires **all three**: documentation PDF, private GitHub repository with judge access, and a 2–3 minute demo video. Its deliverables also require working file-to-file mapping and evaluation predictions. This checklist is not a claim that pending items are finished.

| Required item | Prepared now | Remaining before submission |
|---|---|---|
| Clean private GitHub source | Component packages, final preprocessing, training/inference/evaluation scripts, tests, local component commits | User pushes and grants judges access |
| README, environment, single command | Setup, pinned dependencies, path-only env example, frozen decision models/release manifest and `bash run.sh input.csv output.csv` | Rebuild disposable indexes and perform GPU deployment acceptance; do not publish as fully accepted before this |
| Both code columns for every input | Final predictions and all twelve passing DEV/probe/TEST validators imported into the private bundle | Provide judges restricted bundle access; deployment acceptance remains separate |
| Documentation PDF | Architecture/design/edge-case/attribution docs, final results/error analysis and report draft | Finish the report, create architecture diagram, render final PDF |
| Demo video | 2–3 minute recording script with hard cases | Record live engine, upload privately, supply link in README/PDF |
| Final metrics | All four DEV and internal-TEST flows, three-stage tables, grouped CIs/deltas, buckets/probes, error analysis, missing-code drill and backend/latency evidence in `results/` | Include these exact values and limitations in the final PDF/demo; organizer hidden-set evaluation remains separate |
| Models / training data | Refreshed private ZIP, selected checkpoints, frozen decision layers, deterministic training datasets and four verified public, ungated HF model repositories under `yobro4619` | Private Drive upload/link and judge access to the restricted bundle; models need no token |

The PDF proposes >95% accuracy on both systems; this is the challenge target, not an achieved result. Do not replace missing final numbers with that target or a retrieval-recall number.

The stated judging weights are mapping accuracy 50%, long-tail performance 20%, engineering/reproducibility 15% and error analysis 15%. The problem statement supplies no formula converting these measurements to a single total judging grade. Final F11's primary minimum is 51.30% on DEV and 50.71% on internal TEST; these results do not meet the accuracy target.

The proprietary dataset must not be published or sent to retaining third-party APIs. GitHub must be private. The data ZIP must have restricted access. Model-upload tooling never uploads chart data.

Source preparation does not require rerunning/restarting/cancelling the main evaluation, retraining either model, or opening TEST again. The current ZIP deliberately excludes sealed TEST and raw labelled CSV. The exported DEV/probe reference records are for monitoring only and remain separate from gradient-training examples.
