# Independent agentic solution — hackathon handoff

Start with [SUMMARY.md](SUMMARY.md) for what we tried, results and main inferences. Use [FINAL_REPORT.md](FINAL_REPORT.md) for the complete record and [CODE_INDEX.md](CODE_INDEX.md) for code navigation.

This subdirectory archives the independent agent work inside the team repository. It has its own environment and entry points; the demo uses the team’s released F11 pipeline. Directory integration does not merge the two inference methods. Original work and frozen evidence are retained separately.

## Layout

```text
FINAL_REPORT.md                  Combined report, all experiments and analyses
CODE_INDEX.md                    Source and experiment navigation
independent_agent/               Original Python package and tests
  artifacts/                     Frozen local data, traces, metrics and audits
data/                            Supplied training CSV and dictionary workbooks
remote_workspace/                Supplementary earlier Slurm code/results/logs, if available
tools/verify_package.py          Portable, read-only checksum verification
provenance/                     Content manifest and handoff validation
.env.example                     Placeholder only; no API key included
```

## Safe review and verification

From this folder:

```bash
python3 -m independent_agent --help
```

Frozen artifacts, supplied data, remote logs, original planning documents, duplicate experiment reports and provenance are ignored by Git. They are available in the private full handoff on the research machine. With that complete payload present, `python3 tools/verify_package.py` verifies its original checksums; this historical manifest predates the repository integration README. A Git-only checkout cannot run that full-bundle verifier. The package's Python modules retain their imports and relative layout. Python 3.10+ and SQLite FTS5 are required for runtime use. Baseline requirements are in `independent_agent/requirements.txt`; semantic ONNX embedding requirements are separate in `independent_agent/semantic_embedding_requirements.txt`. Archived dense-SapBERT/Slurm code has its own environment paths and compute requirements.

For a new experiment, create an isolated environment and new outputs, provide your own API credentials via a private `.env`, and review model/provider/cost settings first. Existing historical `run` commands can issue paid requests. Do not use them to inspect results. `prepare` commands may rebuild data and should not be run over frozen artifacts. Stored configurations contain historical absolute paths; use the portable package verifier for relocation integrity rather than editing frozen provenance. Hosted inference is not guaranteed bit-identical across reruns.

The strongest reported full-sample experimental workflow is ICD SapBERT + original SNOMED agent with fallback-only terminal rescue: 70.6% ICD / 71.4% SNOMED on reused development traces. This is not untouched confirmation, a 95% solution, or the team's main model result. No new model training or team integration is claimed.
