# External resources and attribution

| Resource | Use | Source / licence record |
|---|---|---|
| Supplied ICD-10-CM and SNOMED CT US workbooks | Legal output dictionaries and descriptions | Challenge-provided; private challenge redistribution only |
| CDC FY2026 Tabular and Index XML | ICD chapters, blocks, aliases and official specificity notes | CDC public-domain government resource; exact URL/date/hash in RESOURCE_SOURCES.json |
| Clinical Abbreviation Meta-Inventory | Abbreviation candidates/evidence | lisavirginia/clinical-abbreviations; CC BY 4.0; retained attribution/licence in private resources |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | Original alias/type encoder and adapted query encoders | Named public model; original model card and pinned revision supplied in artifact resources |
| Qwen/Qwen3-Reranker-0.6B | Original reranker plus trainable LoRA base | Named public model; original model card and pinned revision supplied in artifact resources |
| SNOMED CT Editorial Guide semantic tags | Official FSN tags and top-level mapping | Jan 2026 Semantic Tag guide; mapping/source comment in terminology/hierarchy.py |
| simple-icd-10-cm | ICD cross-check | MIT package; pinned dependency |
| torch / transformers / PEFT / vLLM | Model training/inference | Pinned core and separate serving dependency lists |

[RESOURCE_SOURCES.json](RESOURCE_SOURCES.json) preserves the downloaded URLs, dates, checksums and recorded licences without machine-specific cache paths. Original model/resource notices accompany the private bundle. Their licences and challenge access terms continue to apply; this source snapshot does not grant a new licence over terminology or chart data.

UMLS, official SNOMED RF2 and the official SNOMED-to-ICD map were not used. The optional bridge is not an official map and remains disabled in the retained policy. No external clinical inference API is used.

The abandoned v1 implementation used the SapBERT metric-learning reference. The final included P2-v2 method uses frozen-index InfoNCE; do not describe the final model as trained with v1 MultiSimilarityLoss. Original reference URL/commit remains in the provenance record as historical attribution, not executable downloaded code.
