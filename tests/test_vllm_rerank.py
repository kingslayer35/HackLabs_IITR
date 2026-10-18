from concurrent.futures import ThreadPoolExecutor
import sqlite3
from types import SimpleNamespace

import pytest

from termmap.reranking.vllm import VllmRanker, raw_margin
from scripts.reranking.serve import dispatch_pooling_adapter


def test_named_pooling_adapter_is_used_and_unknown_models_still_fail():
    adapter = object()
    serving = SimpleNamespace(models=SimpleNamespace(lora_requests={"trained": adapter}))
    def original(serving, ctx, supports_default_mm_loras):
        if ctx.request.model != "base": raise ValueError("unknown model")
    ctx = SimpleNamespace(request=SimpleNamespace(model="trained"))
    dispatch_pooling_adapter(original, serving, ctx)
    assert ctx.lora_request is adapter
    dispatch_pooling_adapter(original, serving, SimpleNamespace(request=SimpleNamespace(model="base")))
    with pytest.raises(ValueError):
        dispatch_pooling_adapter(original, serving, SimpleNamespace(request=SimpleNamespace(model="unknown")))


def test_raw_head_scores_keep_reference_logit_scale():
    # A probability instead of this unbounded margin changes fusion/calibration.
    assert raw_margin([-4.0, 6.0]) == 10.0
    assert raw_margin([3.0, -4.0]) == -7.0
    with pytest.raises(ValueError): raw_margin([0.9])
    with pytest.raises(ValueError): raw_margin([float("nan"), 2.0])


def test_concurrent_requests_deduplicate_and_restore_original_input_order():
    ranker = object.__new__(VllmRanker)
    ranker.settings = dict(request_batch=2)
    ranker.connection = sqlite3.connect(":memory:")
    ranker.connection.execute("CREATE TABLE scores (key TEXT PRIMARY KEY,score REAL NOT NULL)")
    ranker.executor = ThreadPoolExecutor(max_workers=2)
    ranker.token_ids = lambda values: [[value] for value in values]
    submitted = []
    def request(rows):
        submitted.extend(row[0] for row in rows)
        return [float(row[0]) for row in rows]
    ranker.request = request
    assert ranker.score([3, 1, 3, 2]) == [3.0, 1.0, 3.0, 2.0]
    assert sorted(submitted) == [1, 2, 3]
    submitted.clear()
    assert ranker.score([2, 3, 1]) == [2.0, 3.0, 1.0]
    assert submitted == []
    # Turning off the cache exercises the service rather than recycling scores.
    assert ranker.score([2, 3], use_cache=False) == [2.0, 3.0]
    assert sorted(submitted) == [2, 3]
    ranker.close()
