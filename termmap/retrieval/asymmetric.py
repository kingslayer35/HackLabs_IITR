"""Asymmetric SapBERT inference; never imports or invokes model training.

Alias vectors use original SapBERT. A system's queries and training-memory
vectors use the same selected encoder. Type inference always uses original
SapBERT query vectors from this preprocessing generation.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from termmap.common.io import path_for, term_dir, write_json, sha256, seed_everything
from termmap.retrieval.cache import directory, tables
from termmap.retrieval.dense import Encoder, DenseIndex, require_allocation
from termmap.preprocessing.normalize import clean
from termmap.terminology.hierarchy import strip_tag

SYSTEMS = ("ICD", "SNOMED")


def base_model(cfg):
    path = Path(json.loads((path_for(cfg, "external_dir") / "model_snapshot.json").read_text())["path"])
    return path if path.is_absolute() else cfg["root"] / path


def weight_hash(model):
    model = Path(model)
    return sha256(next(p for p in [model / "model.safetensors", model / "pytorch_model.bin"] if p.exists()))


def selected_paths(cfg, candidate=False):
    manifest = path_for(cfg, "model_dir") / "sapbert_v2/selection.json"
    value = json.loads(manifest.read_text())
    result = {s: Path(value["selection"][s]["checkpoint"]) for s in SYSTEMS}
    result = {s: p if p.is_absolute() else cfg["root"] / p for s, p in result.items()}
    confirmation = path_for(cfg, "run_dir") / "C2_v2/confirmation.json"
    if not candidate and confirmation.exists():
        checked = json.loads(confirmation.read_text())
        result = {s: Path(checked["effective_models"][s]) for s in SYSTEMS}
        result = {s: p if p.is_absolute() else cfg["root"] / p for s, p in result.items()}
    for model in result.values():
        if not model.is_dir():
            raise FileNotFoundError("A finalized v2 query checkpoint is unavailable")
    return result, value


def frozen_model_metadata(cfg):
    from termmap.decision.coupling import type_cache_path
    confirmation = path_for(cfg, "run_dir") / "C2_v2/confirmation.json"
    result = json.loads(confirmation.read_text())
    if not result["ready"]:
        raise RuntimeError("New-generation v2 confirmation is not complete")
    candidates, _ = selected_paths(cfg, candidate=True)
    effective, _ = selected_paths(cfg)
    lora_selection=path_for(cfg,"model_dir")/"reranker_lora/selection.json"
    return dict(retriever_generation="sapbert_v2_asymmetric",
                lora_selected_epoch=int(json.loads(lora_selection.read_text())["best_epoch"]),
                lora_selection_sha256=sha256(lora_selection),
                v2_weights_sha256={s:weight_hash(candidates[s]) for s in SYSTEMS},
                effective_models={s:str(effective[s]) for s in SYSTEMS},
                effective_weights_sha256={s:weight_hash(effective[s]) for s in SYSTEMS},
                v2_selection_sha256=sha256(path_for(cfg,"model_dir")/"sapbert_v2/selection.json"),
                v2_confirmation_sha256=sha256(confirmation),
                original_sapbert_sha256=weight_hash(base_model(cfg)),
                type_classifier_sha256=sha256(type_cache_path(cfg,"C1")),
                type_classifier_metadata_sha256=sha256(type_cache_path(cfg,"C1").with_suffix(".json")),
                coupling_index_sha256={name:sha256(frozen_alias_source(cfg)/name) for name in
                    ["ICD_alias_rows.parquet","ICD_aliases.npy","snomed_preferred.npy","snomed_preferred_codes.npy"]},
                query_cache_dir=str(path_for(cfg,"query_cache_dir")),
                inference_source_sha256={name:sha256(cfg["root"]/"termmap"/name) for name in
                                        ["preprocessing/normalize.py","preprocessing/noise.py","terminology/hierarchy.py","retrieval/dense.py","retrieval/cache.py","retrieval/candidates.py",
                                         "decision/coupling.py","retrieval/lexical.py","reranking/model.py","reranking/prompts.py","decision/layer.py",
                                         "retrieval/asymmetric.py"]},
                terminology_sha256={name:sha256(term_dir(cfg)/name) for name in
                    ["aliases.parquet","icd_concepts.parquet","snomed_concepts.parquet","snomed_links.parquet",
                     "priors_train.parquet","pair_evidence_train.parquet","seventh_priors_train.parquet",
                     "tag_priors_train.parquet","top_level_priors_train.parquet","cooccurrence_train.parquet",
                     "header_exceptions_train.parquet","abbreviations.parquet","abbreviation_alias_support_train.parquet"]})


def link_readonly(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError("A frozen inference input is unavailable: " + str(source))
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise RuntimeError("Refusing to replace a non-link frozen-index input")
    target.symlink_to(source.resolve())


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def base_run(profile):
    return {"eval": "C1", "eval_test": "test_base", "final": "final_base"}[profile]


def frozen_alias_source(cfg):
    return (path_for(cfg, "frozen_alias_index_dir") if "frozen_alias_index_dir" in cfg["paths"]
            else path_for(cfg, "index_dir") / "eval/C1")


def _matrix_file(path, shape, batches):
    temporary = path.with_name(path.name + ".partial")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    for start, rows in batches:
        values[start:start + len(rows)] = rows
    values.flush()
    del values
    temporary.replace(path)


def encode_generation(cfg, run="C1", profile="eval", adapted=False, smoke=False):
    require_allocation()
    seed_everything(13, True)
    original = base_model(cfg)
    models = selected_paths(cfg)[0] if adapted else {s: original for s in SYSTEMS}
    root = directory(cfg, profile)
    target = root / run
    target.mkdir(parents=True, exist_ok=True)
    base = root / base_run(profile)
    alias_source = base if adapted else frozen_alias_source(cfg)
    if smoke:
        # Full-cache inputs are not changed by smoke. Selected encoders and
        # both dense/memory operations are exercised on public example text.
        anchor = alias_source if (alias_source / "ICD_aliases.npy").exists() else frozen_alias_source(cfg)
        for system in SYSTEMS:
            encoder = Encoder(models[system], 40, 8)
            query = encoder.encode(["hypertension", "chronic kidney disease"])
            memory = encoder.encode(["high blood pressure", "kidney disease"])
            aliases = pd.read_parquet(anchor / f"{system}_alias_rows.parquet").head(256)
            vectors = np.load(anchor / f"{system}_aliases.npy", mmap_mode="r")[:len(aliases)]
            index = DenseIndex(aliases, vectors, cfg["retrieval"]["dense_chunk"])
            scores = index.score_all(query)
            q = torch.tensor(query, device="cuda", dtype=torch.float16)
            m = torch.tensor(memory, device="cuda", dtype=torch.float16)
            assert np.isfinite(scores).all() and torch.isfinite(q @ m.T).all()
            del encoder, index, q, m
            gc.collect()
            torch.cuda.empty_cache()
        print("Asymmetric encoding smoke passed: original aliases; matched query/memory encoders", flush=True)
        return
    queries_file = root / "queries.json"
    queries = json.loads(queries_file.read_text())
    out = term_dir(cfg)
    aliases = pd.read_parquet(out / ("aliases_final.parquet" if profile == "final" else "aliases.parquet"))
    _, training = tables(cfg, "final" if profile == "final" else "eval")
    memory = training.loc[training.valid_gold, ["row_id", "system", "code", "entity", "group_id"]].rename(columns={"entity": "text"})
    memory["norm_text"] = memory.text.map(clean)
    held = set(pd.read_parquet(out / "probe_set.parquet").heldout_norm) if profile != "final" else set()
    held_keys=memory.text.map(lambda value:clean(strip_tag(value)))
    memory = memory.loc[~held_keys.isin(held)].reset_index(drop=True)
    if not aliases.code.map(lambda value: isinstance(value, str)).all() or not memory.code.map(lambda value: isinstance(value, str)).all():
        raise TypeError("Terminology codes must remain strings")
    expected = dict(profile=profile, adapted=adapted, queries_sha256=sha256(queries_file),
                    model_weights={s: weight_hash(models[s]) for s in SYSTEMS},
                    models={s: str(models[s]) for s in SYSTEMS}, alias_rows=frame_hash(aliases),
                    memory_rows=frame_hash(memory), seed=13,
                    settings={"max_length":cfg["training"]["max_length"],"encode_batch":cfg["retrieval"]["encode_batch"],"dense_chunk":cfg["retrieval"]["dense_chunk"]},
                    gpu_architecture="sm" + "".join(map(str, torch.cuda.get_device_capability())),
                    implementation_sha256=sha256(Path(__file__)),
                    geometry="original aliases; same per-system encoder for queries and memory; original type queries")
    if adapted:
        expected["selection_sha256"] = sha256(path_for(cfg, "model_dir") / "sapbert_v2/selection.json")
        expected["base_encoding_sha256"] = sha256(base / "encoding_metadata.json")
    if profile != "final" or adapted:
        expected["alias_vector_sha256"] = {s: sha256(alias_source / f"{s}_aliases.npy") for s in SYSTEMS}
    marker = target / "encoding_metadata.json"
    if marker.exists() and json.loads(marker.read_text()) == expected:
        return target
    marker.unlink(missing_ok=True)
    for partial in [*target.glob("*.npy"), *target.glob("*.partial")]:
        partial.unlink()
    # Original query vectors are also the sole representation for type fitting.
    original_encoder = None
    if adapted:
        link_readonly(base / "query_embeddings.npy", target / "query_embeddings.npy")
    else:
        original_encoder = Encoder(original, cfg["training"]["max_length"], cfg["retrieval"]["encode_batch"])
        original_encoder.encode(queries, target / "query_embeddings.npy")
    for system in SYSTEMS:
        a = aliases.loc[aliases.system.eq(system)].reset_index(drop=True)
        if adapted:
            # Fresh base rows carry revised norm_text while vector row order and
            # code/text/source remain identical to the original frozen anchor.
            link_readonly(base / f"{system}_alias_rows.parquet", target / f"{system}_alias_rows.parquet")
        else:
            a.to_parquet(target / f"{system}_alias_rows.parquet", index=False)
        if not adapted and profile == "final":
            # Probe aliases are restored only after test. Full final aliases
            # still use the original encoder, never either adapted encoder.
            original_encoder.encode(a.text.tolist(), target / f"{system}_aliases.npy")
        else:
            previous = pd.read_parquet(alias_source / f"{system}_alias_rows.parquet")
            columns = ["system", "code", "text", "source"]
            if not a[columns].equals(previous[columns]):
                raise RuntimeError("Alias code/text/source or row order changed; frozen vectors cannot be reused")
            link_readonly(alias_source / f"{system}_aliases.npy", target / f"{system}_aliases.npy")
        encoder = (Encoder(models[system], cfg["training"]["max_length"], cfg["retrieval"]["encode_batch"])
                   if adapted else original_encoder)
        qpath = target / f"{system}_query_embeddings.npy"
        if adapted:
            qvectors = encoder.encode(queries, qpath)
        else:
            link_readonly(target / "query_embeddings.npy", qpath)
            qvectors = np.load(qpath, mmap_mode="r")
        index = DenseIndex(a, np.load(target / f"{system}_aliases.npy", mmap_mode="r"), cfg["retrieval"]["dense_chunk"])
        np.save(target / f"{system}_codes.npy", index.codes.astype(str))
        for cdc in ([True, False] if system == "ICD" else [True]):
            allowed = None if cdc else ~a.source.isin(["icd_index", "icd_inclusion"]).to_numpy()

            def dense_batches():
                for start in range(0, len(queries), 64):
                    if start % 2048 == 0:
                        print(run, profile, system, "dense", start, "/", len(queries), flush=True)
                    yield start, index.score_all(qvectors[start:start + 64], allowed=allowed).astype(np.float16)

            _matrix_file(target / f"{system}_cdc{int(cdc)}_dense.npy", (len(queries), len(index.codes)), dense_batches())
        del index
        gc.collect()
        torch.cuda.empty_cache()
        mem = memory.loc[memory.system.eq(system)].reset_index(drop=True)
        mem.to_parquet(target / f"{system}_memory_rows.parquet", index=False)
        mv = encoder.encode(mem.text.tolist(), target / f"{system}_memory.npy")
        m = torch.tensor(np.asarray(mv), device="cuda", dtype=torch.float16)

        def memory_batches():
            for start in range(0, len(queries), 64):
                q = torch.tensor(np.asarray(qvectors[start:start + 64]), device="cuda", dtype=torch.float16)
                yield start, (q @ m.T).cpu().numpy()

        _matrix_file(target / f"{system}_memory_scores.npy", (len(queries), len(mem)), memory_batches())
        del m, mv, qvectors
        if adapted:
            del encoder
        gc.collect()
        torch.cuda.empty_cache()
    del original_encoder
    gc.collect()
    torch.cuda.empty_cache()
    write_json(marker, expected)
    return target


def encode_frozen_v2(cfg, run="C2_v2", profile="eval", smoke=False):
    return encode_generation(cfg, run, profile, adapted=True, smoke=smoke)


def retrieve_v2(cfg, run="C2_v2", profile="eval", datasets=("dev", "probe")):
    """One input generation and fixed retrieval policy for both encoder arms."""
    from termmap.retrieval.candidates import CachedRetriever
    from termmap.decision.coupling import infer_types, other_row
    from scripts.reranking.score import metadata
    from termmap.evaluation.metrics import evaluate
    from scripts.evaluation.checkpoints import probe_summary
    if cfg["retrieval"].get("bridge_enabled", True):
        raise RuntimeError("The approved v2 inference policy requires bridge disabled")
    root = directory(cfg, profile)
    target = path_for(cfg, "run_dir") / run
    target.mkdir(parents=True, exist_ok=True)
    provenance = dict(encoding_sha256=sha256(root / run / "encoding_metadata.json"),
                      records_sha256=sha256(root / "records.joblib"), retrieval=cfg["retrieval"],
                      implementation={name: sha256(cfg["root"] / "termmap" / name)
                                      for name in ["retrieval/candidates.py", "decision/coupling.py", "retrieval/lexical.py", "preprocessing/normalize.py"]},
                      alias_rows_sha256=sha256(term_dir(cfg) / ("aliases_final.parquet" if profile == "final" else "aliases.parquet")),
                      profile=profile, datasets=list(datasets))
    marker = target / "pool_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) == provenance and all((target / (d + "_pools.joblib")).exists() for d in datasets):
        return
    retriever = CachedRetriever(cfg, run, profile)
    records = joblib.load(root / "records.joblib")
    infer_types(cfg, retriever, records, run)
    structures = metadata(cfg) if "dev" in datasets else None
    for dataset in datasets:
        result = []
        for i, row in enumerate(records[dataset]):
            tick = time.perf_counter()
            item = retriever.retrieve(row)
            if dataset in {"dev", "probe", "test"}:
                item["other_system"] = retriever.retrieve(other_row(row, cfg))
            item["seconds"] = time.perf_counter() - tick
            result.append(item)
            if i % 250 == 0:
                print(run, "pools", dataset, i, "/", len(records[dataset]), flush=True)
        joblib.dump(result, target / (dataset + "_pools.joblib"))
        if dataset == "dev":
            icd, sn, links, freq = structures
            metrics, details = evaluate(pd.DataFrame([x["row"] for x in result]), [x["pool"] for x in result],
                                        freq, icd, sn, cfg["split"]["bootstrap_samples"], 13,
                                        [x["seconds"] for x in result], links=links)
            write_json(target / "metrics.json", metrics)
            details.drop(columns=[c for c in ["entity", "chunk_text", "display", "views", "context"] if c in details]).to_parquet(target / "dev_details.parquet", index=False)
        elif dataset == "probe":
            write_json(target / "probe_metrics.json", probe_summary(pd.DataFrame([x["row"] for x in result]), [x["pool"] for x in result]))
    write_json(marker, provenance)
