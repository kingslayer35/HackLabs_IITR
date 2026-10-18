"""Verify the portable artifact handoff, independently of research-run paths."""
import json
from termmap.common.io import dictionary_hash, path_for, sha256


def verify_release(cfg):
    path = path_for(cfg, "model_dir") / "RELEASE.json"
    if not path.is_file():
        raise RuntimeError("Final decision-layer handoff is pending. Import the completed main-run artifacts first; see docs/DELIVERABLES.md")
    manifest = json.loads(path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError("The submission artifact handoff is incomplete")
    if manifest["dictionary_hash"] != dictionary_hash(cfg):
        raise RuntimeError("The supplied dictionaries differ from the released models")
    for name in ["retrieval", "preprocessing", "reranker", "decision", "deployment"]:
        if manifest["settings"][name] != cfg[name]:
            raise RuntimeError("Released inference settings changed: " + name)
    for relative, expected in manifest["files"].items():
        file = cfg["root"] / relative
        if sha256(file) != expected:
            raise RuntimeError("Released code or artifact changed: " + relative)
    return manifest
