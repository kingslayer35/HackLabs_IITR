# Algorithm time complexities

`N` = concepts; `A` = alias rows; `M` = memory rows; `d` = embedding width; `k` = candidates in a step; `s` = nodes/hits inspected; `Z` = nonzero character n-gram entries; `P` = matching concept-pair comparisons; `E` = links; `B` = one SapBERT encoding.

Fixed code/text lengths and query-view count; average-case hashing. Comparisons use plain scans/rebuilds.

| Algorithm / operation | TermMap | Simple comparison |
| --- | --- | --- |
| Build ICD prefix tree and subtree intervals | `O(N log N)`, including sorting | All-pairs parent search: `O(N²)` |
| Code / parent / depth / tag lookup | `O(1)` in memory | Scan: `O(N)` |
| ICD ancestor check | `O(1)` using intervals | Parent walk: `O(tree height)` |
| Read ICD subtree | `O(s)`; locate slice: `O(1)` | Scan: `O(N)` |
| Group SNOMED descriptions / select FSNs | `O(A log A)` upper bound | Repeated scans: `O(AN)` |
| Build SNOMED qualifier / lexical links | `O(N + P + E log E)` | All-pairs: `O(N² + E log E)` |
| Read SNOMED neighbours | `O(s)` | Scan links: `O(E)` |
| Exact alias lookup + sort matches | `O(1 + s log s)` | `O(A + s log s)` |
| Character TF-IDF / BM25 scoring | `O(Z + A + N)` / `O(A)` | Still index-size dependent |
| Chunked exact dense scoring | `O(Ad)` per query vector | Same full-scan order |
| Memory scoring + ranking | `O(Md + M log M)` | Same full-scan order |
| RRF + hierarchy / family expansion | `O(k log k + s log s)` | Full-dictionary sorting: `O(N log N)` |
| Qwen reranking | `O(k × one model score + k log k)` | Score all concepts: `O(N × one model score)` |

Channel top-k adds `O(N + k log k)` normally; large ties can make it `O(N log N)`. The ≤64 output cap does not bound nodes inspected. SNOMED links are lexical approximations; `P` can be quadratic.

| Dictionary change | Current operation / cost | Comparison |
| --- | --- | --- |
| Add/edit aliases, same codes | Rebuild lexical/vector snapshots: `O(A log A + AB + Ad)` | Same as full rebuilding |
| Add/edit ICD codes | Tree: `O(N log N)`; refresh affected retrieval indexes | All-pairs tree: `O(N²)`, plus indexes |
| Add/edit SNOMED concepts/tags | Grouping: `O(A log A)`; links: `O(N + P + E log E)`; refresh families/indexes | All-pairs links: `O(N² + E log E)`, plus other rebuilds |
| Encode only `m` changed aliases — possible future updater | Vector work only: `O(mB + md)` | Full re-encoding: `O(AB + Ad)` |

Current updates rebuild affected snapshots; a live incremental updater is not implemented. Lexical statistics, ICD intervals, families and files may still need global rebuilding. Dictionary edits do not inherently require model retraining.
