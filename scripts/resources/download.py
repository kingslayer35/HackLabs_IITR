"""Download only approved public resources; never upload challenge content."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from termmap.common.io import load_config, path_for, sha256, write_json


def fetch(url: str, target: Path, records: list, licence: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(url, timeout=180) as response, target.open("wb") as f:
            while block := response.read(2**20):
                f.write(block)
    records.append(dict(url=url, file=str(target), date=dt.date.today().isoformat(),
                        sha256=sha256(target), licence=licence))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--role", choices=["retriever","reranker"],default="retriever")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ext = path_for(cfg, "external_dir")
    manifest = ext / "sources.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    if not args.model:
        directory = cfg["resources"]["cdc_directory"]
        listing = urllib.request.urlopen(directory).read().decode()
        (ext / "cdc").mkdir(parents=True, exist_ok=True)
        (ext / "cdc" / "directory.html").write_text(listing)
        candidates = re.findall(r'href="([^"]+)"', listing, flags=re.I)
        candidates = [urllib.parse.urljoin(directory, x) for x in candidates
                      if "table" in x.lower() and "index" in x.lower() and x.lower().endswith(".zip")]
        if not candidates:
            raise RuntimeError("CDC FY2026 table/index XML package unavailable")
        # CDC listing has one FY2026 package; preserve listing as provenance.
        url = sorted(candidates)[-1]
        target = ext / "cdc" / urllib.parse.unquote(url.rsplit("/", 1)[-1])
        fetch(url, target, records, "US government public domain")
        with zipfile.ZipFile(target) as z:
            for info in z.infolist():
                dest = (target.parent / "xml" / info.filename).resolve()
                if not dest.is_relative_to((target.parent / "xml").resolve()):
                    raise ValueError("Unsafe ZIP path")
                if info.filename.lower().endswith(".xml"):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(info))
        api = cfg["resources"]["abbreviation_repo"]
        revision = json.load(urllib.request.urlopen(api + "/commits/master"))["sha"]
        base = "https://raw.githubusercontent.com/lisavirginia/clinical-abbreviations/" + revision
        for name in ["metainventory/Metainventory_Version1.0.0.csv", "metainventory/README.md", "LICENSE.txt"]:
            fetch(base + "/" + name, ext / "abbreviations" / name.rsplit("/", 1)[-1], records, "CC BY 4.0")
        # Read, never import or execute, the official loss recipe.
        api = "https://api.github.com/repos/cambridgeltl/sapbert/commits/main"
        revision = json.load(urllib.request.urlopen(api))["sha"]
        url = f"https://raw.githubusercontent.com/cambridgeltl/sapbert/{revision}/src/metric_learning.py"
        fetch(url, ext / "sapbert_reference" / "metric_learning.py.txt", records, "MIT (reference code only)")
    else:
        import os
        os.environ["HF_HOME"] = str(path_for(cfg, "hf_home"))
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        from huggingface_hub import HfApi, snapshot_download
        model = cfg["resources"][args.role]
        revision = HfApi(token=False).model_info(model,token=False).sha
        folder = Path(snapshot_download(model, revision=revision, cache_dir=path_for(cfg, "hf_home") / "hub",
                      token=False, allow_patterns=["config.json", "pytorch_model.bin", "*.safetensors", "tokenizer*", "vocab*", "merges.txt", "special_tokens_map.json", "README.md"]))
        for f in sorted(folder.iterdir()):
            if f.is_file():
                records.append(dict(url=f"https://huggingface.co/{model}/resolve/{revision}/{f.name}",
                                    file=str(f), date=dt.date.today().isoformat(), sha256=sha256(f), licence="Apache-2.0"))
        write_json(ext / ("model_snapshot.json" if args.role=="retriever" else "reranker_snapshot.json"), {"revision": revision, "path": str(folder)})
    records = list({r["file"]: r for r in records}.values())
    write_json(manifest, records)
    text = "# External sources\n\nAll files are read as data only. No challenge data is uploaded.\n\n"
    text += "| URL | Local file | Download date | SHA256 | Licence |\n|---|---|---|---|---|\n"
    for r in records:
        text += f"| {r['url']} | {r['file']} | {r['date']} | {r['sha256']} | {r['licence']} |\n"
    (ext / "SOURCES.md").write_text(text)
    print(f"Recorded {len(records)} public source files")


if __name__ == "__main__":
    main()
