"""Public synthetic checks for the asymmetric encoder and fallback contract."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

import termmap.retrieval.asymmetric as v2
from termmap.retrieval.confirmation import RULES, rule_checks, choose_models


def test_adapted_queries_and_memory_share_encoder_but_aliases_and_types_stay_original(monkeypatch, tmp_path):
    cache = tmp_path / "cache/eval"
    base = cache / "C1"
    base.mkdir(parents=True)
    out = tmp_path / "tables"
    out.mkdir()
    model_root = tmp_path / "models"
    selected = {s: model_root / s for s in v2.SYSTEMS}
    for s, model in selected.items():
        model.mkdir(parents=True)
        (model / "model.safetensors").write_bytes(s.encode())
    (model_root / "sapbert_v2").mkdir()
    (model_root / "sapbert_v2/selection.json").write_text("{}")
    queries = ["query-one", "query-two"]
    (cache / "queries.json").write_text(json.dumps(queries))
    (base / "encoding_metadata.json").write_text("{}")
    type_vectors = np.full((2, 2), .5, dtype=np.float16)
    np.save(base / "query_embeddings.npy", type_vectors)
    all_aliases, training, original_bytes = [], [], {}
    codes = {"ICD": ["I10", "I50.9"], "SNOMED": ["100000000000000001", "100000000000000003"]}
    for system in v2.SYSTEMS:
        aliases = pd.DataFrame([dict(system=system, code=code, text=f"alias-{system}-{i}",
                                    norm_text=f"alias {system} {i}", source="dict_pref")
                                for i, code in enumerate(codes[system])])
        aliases.to_parquet(base / f"{system}_alias_rows.parquet")
        all_aliases.append(aliases)
        np.save(base / f"{system}_aliases.npy", np.eye(2, dtype=np.float16))
        original_bytes[system] = (base / f"{system}_aliases.npy").read_bytes()
        for i, code in enumerate(codes[system]):
            training.append(dict(valid_gold=True, row_id=str(len(training)), system=system,
                                 code=code, entity=f"memory-{i}", group_id=f"g{i}"))
    pd.concat(all_aliases).to_parquet(out / "aliases.parquet", index=False)
    pd.DataFrame({"heldout_norm": ["held alias"]}).to_parquet(out / "probe_set.parquet")
    training.append(dict(valid_gold=True,row_id="held",system="ICD",code="I10",
                         entity="held alias (disorder)",group_id="held"))
    monkeypatch.setattr(v2, "require_allocation", lambda: None)
    monkeypatch.setattr(v2, "seed_everything", lambda *args: None)
    monkeypatch.setattr(v2, "term_dir", lambda cfg: out)
    monkeypatch.setattr(v2, "directory", lambda cfg, profile="eval": cache)
    monkeypatch.setattr(v2, "tables", lambda *args: (out, pd.DataFrame(training)))
    monkeypatch.setattr(v2, "base_model", lambda cfg: tmp_path / "original")
    monkeypatch.setattr(v2, "selected_paths", lambda cfg: (selected, {}))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda data, **kwargs: tensor(data, dtype=kwargs.get("dtype")))
    calls = []

    class Encoder:
        def __init__(self, model, *args): self.system = Path(model).name
        def encode(self, texts, target):
            calls.append((self.system, list(texts)))
            is_memory = texts[0].startswith("memory-")
            vector = ([2, 0] if is_memory else [1, 0]) if self.system == "ICD" else ([0, 3] if is_memory else [0, 1])
            result = np.tile(vector, (len(texts), 1)).astype(np.float16)
            np.save(target, result)
            return result

    class Index:
        def __init__(self, aliases, vectors, *args):
            self.codes = aliases.code.to_numpy(dtype=str)
            self.vectors = vectors
        def score_all(self, query, **kwargs): return np.asarray(query) @ self.vectors.T

    monkeypatch.setattr(v2, "Encoder", Encoder)
    monkeypatch.setattr(v2, "DenseIndex", Index)
    cfg = dict(root=tmp_path, paths=dict(model_dir=str(model_root)), training=dict(max_length=40),
               retrieval=dict(encode_batch=8, dense_chunk=16))
    v2.encode_frozen_v2(cfg)
    target = cache / "C2_v2"
    np.testing.assert_array_equal(np.load(target / "query_embeddings.npy"), type_vectors)
    np.testing.assert_array_equal(np.load(target / "ICD_memory_scores.npy"), np.full((2, 2), 2))
    np.testing.assert_array_equal(np.load(target / "SNOMED_memory_scores.npy"), np.full((2, 2), 3))
    for system in v2.SYSTEMS:
        assert (base / f"{system}_aliases.npy").read_bytes() == original_bytes[system]
        assert (target / f"{system}_aliases.npy").is_symlink()
        assert np.load(target / f"{system}_codes.npy").tolist() == codes[system]
        assert (system, ["memory-0", "memory-1"]) in calls
    assert all(not text.startswith("alias-") for _, texts in calls for text in texts)


def test_a_failed_memory_rule_falls_back_only_that_system():
    baseline = {s: {key: 70. for key, _, _ in RULES} for s in v2.SYSTEMS}
    candidate = {s: {key: 72. for key, _, _ in RULES} for s in v2.SYSTEMS}
    candidate["SNOMED"]["memory_R@10"] = 69.
    checks = rule_checks(candidate, baseline)
    result = choose_models(checks, {"ICD": Path("selected_icd"), "SNOMED": Path("selected_snomed")}, Path("original"))
    assert result == {"ICD": "selected_icd", "SNOMED": "original"}


def test_online_new_queries_route_dense_and_memory_per_system(monkeypatch):
    from termmap.retrieval.online import OnlineRetriever, ExtraRows

    class Encoder:
        def __init__(self, value): self.value = value
        def encode(self, rows): return np.full((len(rows), 2), self.value, dtype=np.float16)

    class Index:
        def score_all(self, rows, **kwargs): return rows[:, :1]

    ranker = object.__new__(OnlineRetriever)
    ranker.encoder = Encoder(1)
    ranker.query_encoders = {"ICD": Encoder(2), "SNOMED": Encoder(3)}
    ranker.qid, ranker.queries = {}, []
    ranker.qvectors = ExtraRows(np.empty((0, 2), dtype=np.float16))
    ranker.system_qvectors = {s: ExtraRows(np.empty((0, 2), dtype=np.float16)) for s in v2.SYSTEMS}
    ranker.char, ranker.word, ranker.dense = {}, {}, {}
    ranker.alias_rows = {s: pd.DataFrame(dict(source=["dict_pref"])) for s in v2.SYSTEMS}
    ranker.indexes = {s: Index() for s in v2.SYSTEMS}
    ranker.memory_vectors = {s: np.array([[2, 2]], dtype=np.float16) for s in v2.SYSTEMS}
    ranker.memory_scores = {s: ExtraRows(np.empty((0, 1), dtype=np.float16)) for s in v2.SYSTEMS}
    for s in v2.SYSTEMS:
        ranker.dense[(s, True)] = ExtraRows(np.empty((0, 1), dtype=np.float16))
        ranker.char[(s, True)] = ExtraRows(np.empty((0, 1), dtype=np.float16))
        ranker.word[(s, True)] = []

    class Char:
        def score_all(self, rows): return np.ones((len(rows), 1))

    class Word:
        def search(self, rows, count): return [[("I10", 1.)] for _ in rows]

    ranker.lex_objects_for = lambda key: (Char(), Word())
    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda data, **kwargs: tensor(data, dtype=kwargs.get("dtype")))
    ranker.add_queries(["zeta", "alpha", "alpha"])
    assert ranker.queries == ["alpha", "zeta"]
    np.testing.assert_array_equal(ranker.qvectors[[0, 1]], np.ones((2, 2)))
    for system, expected in [("ICD", 2), ("SNOMED", 3)]:
        np.testing.assert_array_equal(ranker.system_qvectors[system][[0, 1]], np.full((2, 2), expected))
        np.testing.assert_array_equal(ranker.dense[(system, True)][[0, 1]], np.full((2, 1), expected))
        np.testing.assert_array_equal(ranker.memory_scores[system][[0, 1]], np.full((2, 1), expected * 4))
    ranker.add_queries(["alpha"])
    assert len(ranker.qvectors) == len(ranker.system_qvectors["ICD"]) == 2


@pytest.mark.parametrize("system", ["ICD", "SNOMED"])
def test_freeze_rejects_drift_in_either_v2_checkpoint(monkeypatch, tmp_path, system):
    import termmap.common.io as io
    tables = tmp_path / "tables"
    tables.mkdir()
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    original = tmp_path / "original"
    original.mkdir()
    (original / "model.safetensors").write_bytes(b"original")
    selected = {s: models / "sapbert_v2" / s for s in v2.SYSTEMS}
    for s, path in selected.items():
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(s.encode())
    io.write_json(models / "sapbert_v2/selection.json",
                  {"selection": {s: {"checkpoint": str(p)} for s, p in selected.items()}})
    io.write_json(runs / "C2_v2/confirmation.json",
                  dict(ready=True, effective_models={s: str(p) for s, p in selected.items()}))
    for name in ["preprocessing/normalize.py", "preprocessing/noise.py", "terminology/hierarchy.py", "retrieval/dense.py", "retrieval/cache.py", "retrieval/candidates.py", "decision/coupling.py",
                 "retrieval/lexical.py", "reranking/model.py", "reranking/prompts.py", "decision/layer.py", "retrieval/asymmetric.py"]:
        path = tmp_path / "termmap" / name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("source")
    for name in ["aliases", "icd_concepts", "snomed_concepts", "snomed_links", "priors_train",
                 "pair_evidence_train", "seventh_priors_train", "tag_priors_train", "top_level_priors_train",
                 "cooccurrence_train", "header_exceptions_train", "abbreviations",
                 "abbreviation_alias_support_train", "splits"]:
        (tables / (name + ".parquet")).write_bytes(b"frozen")
    adapter = models / "reranker_lora/best/adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    (adapter.parent.parent/"selection.json").write_text(json.dumps(dict(best_epoch=1)))
    decision = models / "decision/F00.joblib"
    decision.parent.mkdir()
    decision.write_bytes(b"decision")
    (models / "type_eval_C1.joblib").write_bytes(b"type")
    (models / "type_eval_C1.json").write_text("{}")
    cfg = dict(root=tmp_path, paths=dict(run_dir="runs", model_dir="models", query_cache_dir="cache"),
               retrieval={}, reranker={}, decision={})
    monkeypatch.setattr(v2, "term_dir", lambda cfg: tables)
    monkeypatch.setattr(io, "term_dir", lambda cfg: tables)
    monkeypatch.setattr(io, "dictionary_hash", lambda cfg: "dictionary")
    monkeypatch.setattr(v2, "base_model", lambda cfg: original)
    index=tmp_path/"fixed_index";index.mkdir()
    for name in ["ICD_alias_rows.parquet","ICD_aliases.npy","snomed_preferred.npy","snomed_preferred_codes.npy"]:
        (index/name).write_bytes(b"original index")
    monkeypatch.setattr(v2,"frozen_alias_source",lambda cfg:index)
    frozen = dict(dictionary_hash="dictionary", splits_sha256=io.sha256(tables / "splits.parquet"),
                  retrieval={}, reranker={}, decision_settings={}, adapter_weights_sha256=io.sha256(adapter),
                  decisions={"F00": io.sha256(decision)}, **v2.frozen_model_metadata(cfg))
    io.write_json(runs / "frozen/CONFIG_FROZEN.json", frozen)
    assert io.verify_frozen(cfg) == frozen
    (selected[system] / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="v2_weights_sha256"):
        io.verify_frozen(cfg)
