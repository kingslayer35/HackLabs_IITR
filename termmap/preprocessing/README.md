# Final mention preprocessing

`normalize.py` is the final reviewed implementation; earlier versions and one-off repair scripts are omitted. `noise.py` shares deterministic noise with probe generation and pretraining.

V0 preserves the input mention. V1 performs conservative Unicode/HTML/punctuation cleanup. V2 adds evidence-gated abbreviation readings. V3 applies guarded spelling correction; V4 handles guarded token segmentation/merging. Existing dictionary names, long text spans and protected clinical tokens are preserved. Check the exact view implementation in `Normalizer.views` and public regression tests.

Clinical inequalities, signed quantities, receptor signs, laterality, stage/type/grade values and Roman numerals are protected. Abbreviation evidence distinguishes public dictionary support from train-group-only evidence. Leave-group-out masking removes self-only expansions for a training row. Ambiguous abbreviations use contextual evidence; a type label is never a candidate filter.

Context receives formatting cleanup only. The mention is anchored on whole-token boundaries, flexible whitespace is supported, absent mentions are flagged, and repeated mentions retain an ambiguity signal. Cue flags describe local negation/history/family/section evidence. No chunk spell correction or clinical interpretation is inserted into the text.

Entry points: `scripts.preprocessing.prepare_queries`, `scripts.preprocessing.audit_rows`. `scripts.preprocessing.measure --tighten` performs the current M1–M4 checks and records manual judgments as pending until reviewed; it does not load an older implementation.
