from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "independent_agent" / "artifacts"
SYSTEMS = ("ICD10CM", "SNOMEDCT_US")
SEED = 13


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.findall(r"[\w]+", text))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_id(entity: str, context: str) -> str:
    return digest(entity + "\0" + context)[:20]


def shingles(text: str) -> frozenset[str]:
    words = norm(text).split()
    if len(words) < 3:
        return frozenset(words)
    return frozenset(" ".join(words[i:i + 3]) for i in range(len(words) - 2))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())
