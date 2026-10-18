"""Configuration, string-preserving loaders and local artifact I/O."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("TERMMAP_CONFIG","configs/default.yaml")).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["root"] = path.parent.parent
    cfg["config_path"] = path
    policy_path=Path(cfg["paths"]["run_dir"])
    if not policy_path.is_absolute():policy_path=cfg["root"]/policy_path
    policy_path=policy_path/"retrieval_policy.json"
    if policy_path.exists():cfg["retrieval"].update(json.loads(policy_path.read_text()))
    reranker_policy=policy_path.with_name("reranker_policy.json")
    if reranker_policy.exists():cfg.setdefault("reranker",{}).update(json.loads(reranker_policy.read_text()))
    return cfg


def path_for(cfg: dict, key: str) -> Path:
    p = Path(os.path.expandvars(cfg["paths"][key])).expanduser()
    return p if p.is_absolute() else cfg["root"] / p


def data_path(cfg: dict, key: str) -> Path:
    return path_for(cfg, "data_dir") / cfg["paths"][key]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def dictionary_hash(cfg: dict) -> str:
    h = hashlib.sha256()
    for key in ("icd_file", "snomed_file"):
        h.update(sha256(data_path(cfg, key)).encode())
    return h.hexdigest()[:20]


def term_dir(cfg: dict) -> Path:
    p = path_for(cfg, "terminology_dir") / dictionary_hash(cfg)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_strings(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False)
    for col in df:
        df[col] = df[col].str.strip()
    df.insert(0, "source_row", [str(i) for i in range(len(df))])
    return df


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seed_everything(seed: int = 13, torch_enabled: bool = False) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    if torch_enabled:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(6)


def read_partition(cfg: dict, split: str) -> pd.DataFrame:
    """Never deserialize sealed test content during P0–P2."""
    if split=="test" and os.environ.get("TERMMAP_PHASE")=="P5" and (path_for(cfg,"run_dir")/"frozen"/"CONFIG_FROZEN.json").exists():
        return pd.read_parquet(term_dir(cfg)/"sealed_test.parquet")
    if split not in {"train", "dev"}:
        raise ValueError("The test partition is sealed until P5")
    return pd.read_parquet(term_dir(cfg) / f"{split}.parquet")


def verify_frozen(cfg: dict) -> dict:
    """Reject any configuration/artifact drift before sealed-test access."""
    frozen=json.loads((path_for(cfg,"run_dir")/"frozen"/"CONFIG_FROZEN.json").read_text())
    checks={"dictionary_hash":dictionary_hash(cfg),"splits_sha256":sha256(term_dir(cfg)/"splits.parquet"),
            "retrieval":cfg["retrieval"],"reranker":cfg.get("reranker",{}),"decision_settings":cfg.get("decision",{}),
            "adapter_weights_sha256":sha256(path_for(cfg,"model_dir")/"reranker_lora"/"best"/"adapter_model.safetensors"),
            "decisions":{flow:sha256(path_for(cfg,"model_dir")/"decision"/(flow+".joblib")) for flow in frozen["decisions"]}}
    from termmap.retrieval.asymmetric import frozen_model_metadata
    checks.update(frozen_model_metadata(cfg))
    from termmap.reranking.provenance import serving_metadata
    checks.update(serving_metadata(cfg))
    for key,value in checks.items():
        if value!=frozen[key]:raise RuntimeError(f"Frozen artifact/configuration changed: {key}")
    return frozen
