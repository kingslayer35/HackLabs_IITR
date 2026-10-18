# Contextual reranking and local serving

`model.py` builds candidate cards containing preferred terms/aliases, coding-system structure, qualifiers and permitted training examples. The query contains the mention, its selected views and chunk context. `prompts.py` preserves section-aware 512-token packing. Scoring uses the original Qwen yes-minus-no logits; no random baseline or fabricated label is used.

`vllm.py`, `client.py`, `hosted.py` and `cards.py` support a local-only 127.0.0.1 server, LoRA selection, bounded concurrent requests, input/card caches and deterministic request order checks. These caches are local runtime files, not submission artifacts. Model/template/tokenizer/adapter/version/GPU-architecture fingerprints isolate score reuse. `evidence.py` and `provenance.py` record backend evidence.

Original Qwen without LoRA is F00/F10; revised epoch-2 LoRA is F01/F11. Transformers and vLLM bf16 scores can differ numerically; retained results must disclose backend/architecture rather than silently mix them. The smoke compares exact packed tokens, adapter effect, finite margins and repeated/reversed requests. The disposable reference can match bf16 adapter arithmetic without modifying saved weights.

Optional vLLM uses its separate lockfile and configured Python executable. The default portable prediction configuration uses vLLM to retain the final hosted model path; Transformers remains available as a separately configured backend. A final result report must preserve the backend actually used in the main evaluation; changing the deployment backend is not a claim that reported scores were recomputed.
