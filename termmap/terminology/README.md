# Terminology and hierarchical reasoning

`hierarchy.py` loads all supplied concepts with string IDs. ICD parents are the longest existing proper prefix after ignoring the dot; chapters and blocks come from CDC FY2026 Tabular XML. `structure.py` enriches the tree, seventh-character meanings, traversal intervals, conservative SNOMED links and indexed SQLite access. `tables.py` builds aliases, abbreviation evidence, memory and priors from train only.

SNOMED FSN detection uses the official semantic-tag list, not arbitrary parentheses. Each tag maps to its official top-level hierarchy; the product tag follows the documented primary-domain rule. Missing tags get unknown. Tags and top-level/subconcept priors are soft features only.

SNOMED structure is a lexical approximation, not the official RF2 IS-A graph. Trusted specifier links, qualifier siblings and look-alikes are distinguished. No official SNOMED-to-ICD map is claimed. The bridge interface remains available, but the retained retrieval policy disables that channel. This package is the hierarchy / HRKL component of the submission; it does not imply a separately trained hierarchy model.

Expected dictionaries: 98,182 ICD codes, 74,718 leaves, 382,124 SNOMED concepts; 382,120 tagged. The real ICD Z99.3 row labelled ProcedureCode is preserved. Missing dictionary gold L6.0 stays flagged and in evaluation denominators.

Fresh-build commands: `python -m scripts.terminology.build` then `python -m scripts.terminology.enrich`. Existing private bundle tables preserve the actual saved split; do not rebuild that split for comparisons.
