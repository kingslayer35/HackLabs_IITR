# Fitted decision layer

`layer.py` implements candidate feature fusion, specificity, coupling, confidence and missing-code detection. Features include reranker normalization, retrieval/memory/frequency evidence, ICD structure, qualifier/context support and soft SNOMED tag/top-level priors. Types never remove candidates.

ICD descent requires supported specificity and respects legal header exceptions, laterality, unspecified targets and seventh-character evidence. SNOMED links remain conservative. Coupling uses train-derived co-occurrence and original-vector similarity as soft agreement, never a crosswalk gold label. `coupling.py` constructs and infers the opposite-system input without inventing gold labels.

Each flow is fitted independently with nested grouped DEV cross-validation. An evaluated group never fits its own fusion, component choices or calibration. Prescribed unseen/probe guards determine whether coupling is retained; probe evaluation remains separate. Confidence estimates correctness; missing-code detection estimates whether gold is absent from the pool. Every input still receives one legal code in each system.

The fitted full-DEV decision files used on TEST are imported only after the main evaluation finishes. They are pending in the first artifact bundle. `scripts.decision.fit` contains fitting and frozen-application code; never refit on TEST. `common/release.py` rejects missing or changed deployment artifacts before GPU allocation.
