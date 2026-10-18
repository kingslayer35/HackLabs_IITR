"""Reconstruct actual seeded training examples, without training or using a GPU.

The post-training exporter imports the immutable training-runtime snapshot. The
pretraining exporter uses the saved pinned query generation and original alias
row order. All content is written only beneath the supplied private destination.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def pretraining(source, destination):
    sys.path.insert(0, str(source))
    import joblib
    import numpy as np
    import pandas as pd
    from termmap.adapt_v2 import (AliasSampler, CDC_SOURCES, FrozenIndex,
        mention_examples, pinned, train_frequency, weight_rare)
    from termmap.cache import directory
    from termmap.io import load_config, path_for, term_dir

    cfg = pinned(load_config(source / "configs/default.yaml"))
    tables = term_dir(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    reference_records = joblib.load(directory(cfg) / "records.joblib")
    if set(reference_records) - {"train", "dev", "probe", "alias"}:
        raise RuntimeError("Unexpected partition in pinned training inputs")
    for dataset, records in reference_records.items():
        (destination / f"reference_{dataset}_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    (destination / "reference_query_texts.json").write_bytes((directory(cfg) / "queries.json").read_bytes())
    (destination / "retrieval_policy_C1.json").write_bytes(
        (path_for(cfg,"run_dir")/"pretrain_v2/retrieval_policy_C1.json").read_bytes())
    indexes = {}
    for system in ["ICD", "SNOMED"]:
        rows = pd.read_parquet(directory(cfg) / "C1" / f"{system}_alias_rows.parquet")
        if system == "ICD" and not cfg["retrieval"].get("cdc_enabled", True):
            rows = rows.loc[~rows.source.isin(CDC_SOURCES)]
        rows = rows.reset_index(drop=True)
        rows.to_parquet(destination / f"{system}_frozen_alias_rows.parquet", index=False)
        # Mask construction and sampling need row identity, not the vectors.
        indexes[system] = FrozenIndex(system, rows, np.empty((len(rows), 0), dtype=np.float16))
    probe = pd.read_parquet(tables / "probe_set.parquet")
    held = set(probe.heldout_norm)
    frequency = train_frequency(cfg)
    examples, skipped = mention_examples(cfg, indexes, "train", held, frequency)
    examples = weight_rare(examples, frequency)
    audit = json.loads((path_for(cfg, "run_dir") / "pretrain_v2/data_audit.json").read_text())
    if len(examples) != audit["train"]["examples"]:
        raise RuntimeError("Pretraining reconstruction does not match the recorded training count")

    def export(rows, path):
        output = []
        for item in rows:
            index = indexes[item["system"]]
            positives = np.setdiff1d(index.by_code[item["code"]], item["masked"])
            output.append(dict(item, masked=item["masked"].tolist(),
                positive_alias_rows=positives.tolist(),
                positive_texts=index.rows.iloc[positives].text.tolist()))
        pd.DataFrame(output).to_parquet(path, index=False)

    export(examples, destination / "mention_examples.parquet")
    settings = cfg["training_v2"]
    sampler = AliasSampler(cfg, indexes, set(zip(probe.system, probe.code)), settings)
    n_alias = int(round(settings["alias_query_ratio"] * len(examples)))
    epochs = []
    for epoch in range(1, settings["max_epochs"] + 1):
        aliases = sampler.sample(epoch, n_alias)
        export(aliases, destination / f"alias_examples_epoch_{epoch:02d}.parquet")
        order = np.random.default_rng([13, epoch]).permutation(len(examples) + len(aliases))
        pd.DataFrame(dict(position=np.arange(len(order)), example_index=order)).to_parquet(
            destination / f"batch_order_epoch_{epoch:02d}.parquet", index=False)
        epochs.append(dict(epoch=epoch, mention_examples=len(examples), alias_examples=len(aliases)))
    manifest = dict(format="frozen-index InfoNCE examples, not v1 metric-learning pairs",
        provenance="Deterministically reconstructed; training did not persist materialized epoch datasets",
        query_generation="Pinned generation used by the original P2-v2 run; not regenerated with later preprocessing",
        reference_dev_and_probe="Evaluation/monitoring only; never training examples",
        seed=13, counts_verified_against_recorded_audit=True,
        source_records_sha256=file_hash(directory(cfg) / "records.joblib"),
        train_skipped=dict(skipped), epochs=epochs,
        negative_universe="Every other concept in the system's frozen alias table, excluding masked alias rows",
        positive_definition="Unmasked aliases of the same code; frozen original SapBERT vectors",
        no_dev_or_test_training=True, no_probe_alias_training=True,
        source_files={name:file_hash(source / "termmap" / name)
                      for name in ["adapt_v2.py", "normalize.py", "noise.py", "hierarchy.py"]})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported pretraining examples:", len(examples), "mentions and", n_alias, "aliases per epoch", flush=True)


def posttraining(source, destination):
    runtime = source / "artifacts/runs/revised_posttrain_20261018/runtime"
    sys.path.insert(0, str(runtime))
    import joblib
    import pandas as pd
    from termmap.io import load_config, path_for, term_dir
    from termmap.rerank import Cards
    from termmap.rerank_training import quiz, sampled_groups
    from termmap.normalize import clean
    from termmap.hierarchy import strip_tag

    cfg = load_config(runtime / "configs/revised_posttrain.yaml")
    ready = json.loads((path_for(cfg, "run_dir") / "READY.json").read_text())
    for name in ["termmap/rerank_training.py", "termmap/rerank.py", "termmap/normalize.py", "termmap/noise.py"]:
        if file_hash(runtime / name) != ready["runtime_sources"][name]:
            raise RuntimeError("Immutable post-training source changed")
    for filename, expected in ready["inputs"].items():
        if file_hash(filename) != expected:
            raise RuntimeError("Immutable post-training inputs changed")
    destination.mkdir(parents=True, exist_ok=True)
    cards = Cards(cfg)
    pools = path_for(cfg, "run_dir") / "C1"
    train = [x for x in joblib.load(pools / "train_pools.joblib")
             if clean(strip_tag(x["row"]["entity"])) not in cards.held]
    aliases = joblib.load(pools / "alias_pools.joblib")
    links = defaultdict(set)
    for row in pd.read_parquet(term_dir(cfg) / "snomed_links.parquet").itertuples():
        links[row.src].add(row.dst)
        links[row.dst].add(row.src)
    counts = []
    for epoch in [1, 2]:
        selected = sampled_groups(cfg, train, aliases, epoch)
        expected = json.loads((path_for(cfg, "model_dir") / "reranker_lora" /
                               f"epoch_{epoch}/training.json").read_text())["n_groups"]
        if len(selected) != expected:
            raise RuntimeError("Reconstructed post-training quiz count differs")
        records = []
        for position, item in enumerate(selected):
            row, codes, gold, use_type, use_seen = quiz(item, cards, links, epoch)
            records.append(dict(epoch=epoch, position=position, row_id=str(row["row_id"]),
                group_id=str(row["group_id"]), system=row["system"], gold_code=str(row["code"]),
                candidate_codes=codes, gold_position=gold, include_type=use_type, include_seen=use_seen,
                payloads=[cards.payload(row, code, use_type, use_seen) for code in codes],
                excluded_group=row.get("excluded_group"), excluded_alias=row.get("excluded_alias")))
        pd.DataFrame(records).to_parquet(destination / f"quizzes_epoch_{epoch}.parquet", index=False)
        counts.append(dict(epoch=epoch, groups=len(records), candidate_judgments=sum(len(r["candidate_codes"]) for r in records)))
    manifest = dict(seed=13, source="Immutable revised-input training runtime and frozen C1 pools",
        provenance="Deterministic reconstruction in exact per-epoch sample/quiz order",
        actual_counts_verified=True, epochs=counts, original_sapbert_pools=True,
        no_dev_or_test_training=True, source_config_sha256=file_hash(cfg["config_path"]),
        pool_sha256={name:file_hash(pools / name) for name in ["train_pools.joblib", "alias_pools.joblib"]})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported post-training quizzes:", counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["pretraining", "posttraining"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    globals()[args.kind](args.source.resolve(), args.destination.resolve())
