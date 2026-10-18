"""Assemble private data/checkpoints locally and create a verified ZIP.

Never reads the raw labelled CSV or sealed TEST partition. No network calls,
Git changes, training, inference or writes to the source project occur.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            value.update(part)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    root = Path(__file__).resolve().parents[1]
    private = root / "private_assets"
    private.mkdir(exist_ok=True)
    import yaml
    original = yaml.safe_load((source / "configs/default.yaml").read_text())
    data = Path(original["paths"]["data_dir"])
    h = hashlib.sha256()
    for key in ["icd_file", "snomed_file"]:
        h.update(sha(data / original["paths"][key]).encode())
    dictionary_hash = h.hexdigest()[:20]
    source_tables = source / "artifacts/terminology" / dictionary_hash
    table_names = [p.name for p in source_tables.glob("*.parquet")
                   if p.name not in {"sealed_test.parquet", "train_clean.parquet"}
                   and not p.stem.endswith("_final")]

    def copy(source_file, relative):
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        if sha(source_file) != sha(target):
            raise RuntimeError("Asset copy verification failed")

    for name in sorted(table_names):
        copy(source_tables / name, Path("terminology") / dictionary_hash / name)
    for name in ["split_checks.json", "icd_checks.json", "snomed_checks.json"]:
        copy(source_tables / name, Path("terminology") / dictionary_hash / name)
    for key in ["icd_file", "snomed_file"]:
        name = original["paths"][key]
        copy(data / name, Path("data") / name)

    model_files = {"model.safetensors", "pytorch_model.bin", "adapter_model.safetensors",
        "adapter_config.json", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
        "generation_config.json", "chat_template.jinja", "training.json"}
    def model(folder, destination):
        for file in sorted(folder.iterdir()):
            if file.is_file() and (file.name in model_files or
                    file.name.endswith(".safetensors") or file.name.endswith(".safetensors.index.json")):
                copy(file, Path("models") / destination / file.name)

    pretrained = source / "artifacts/models/sapbert_v2"
    selection = json.loads((pretrained / "selection.json").read_text())
    settings=json.loads((pretrained/"settings.json").read_text())
    settings["base"]="private_assets/models/base_sapbert"
    for system in ["ICD", "SNOMED"]:
        model(pretrained / ("best_" + system), Path("sapbert_v2") / ("best_" + system))
        selection["selection"][system]["checkpoint"] = "private_assets/models/sapbert_v2/best_" + system
    (private / "models/sapbert_v2/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (private / "models/sapbert_v2/settings.json").write_text(json.dumps(settings,indent=2)+"\n")
    lora = source / "artifacts/models/revised_posttrain_20261018/reranker_lora"
    (private/"models/reranker_lora").mkdir(parents=True,exist_ok=True)
    copy(lora/"settings.json",Path("models/reranker_lora/settings.json"))
    for name in ["epoch_1", "epoch_2"]:
        model(lora / name, Path("reranker_lora") / name)
    model(lora / "epoch_2", Path("reranker_lora/best"))
    for name in ["epoch_1", "epoch_2", "best"]:
        configuration=private/"models/reranker_lora"/name/"adapter_config.json"
        value=json.loads(configuration.read_text())
        value["base_model_name_or_path"]="Qwen/Qwen3-Reranker-0.6B"
        configuration.write_text(json.dumps(value,indent=2)+"\n")
        (configuration.parent/"README.md").write_text(
            "---\nbase_model: Qwen/Qwen3-Reranker-0.6B\nlibrary_name: peft\nlicense: apache-2.0\n---\n"
            "# TermMap revised-input Qwen LoRA\n\nPrivate challenge-derived adapter. "
            "Load on the original Qwen3-Reranker-0.6B base. "
            +("Selected epoch 2." if name=="best" else "Saved "+name.replace("_"," ")+".")
            +" Tensor bytes are unchanged; the base-model configuration path was made portable.\n")
    (private / "models/reranker_lora/selection.json").write_text(json.dumps(dict(
        best_epoch=2, selected_epoch=2, selection_rule="Explicit user selection before TEST",
        source_automatic_selection_preserved=True,
        original_automatic_selection=json.loads((lora / "selection.json").read_text()),
        source_adapter_sha256=sha(lora / "epoch_2/adapter_model.safetensors")), indent=2) + "\n")

    resources = private / "resources"
    resources.mkdir(exist_ok=True)
    for snapshot_name, folder in [("model_snapshot.json", "base_sapbert"), ("reranker_snapshot.json", "base_qwen")]:
        info = json.loads((source / "external" / snapshot_name).read_text())
        model(Path(info["path"]), Path(folder))
        original_card=Path(info["path"])/"README.md"
        if original_card.exists():copy(original_card,Path("models")/folder/"README.md")
        info["path"] = "private_assets/models/" + folder
        (resources / snapshot_name).write_text(json.dumps(info, indent=2) + "\n")
    for system in ["ICD","SNOMED"]:
        (private/"models/sapbert_v2"/("best_"+system)/"README.md").write_text(
            "---\nbase_model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext\nlibrary_name: transformers\nlicense: apache-2.0\n---\n"
            "# TermMap SapBERT "+system+" query/memory encoder\n\n"
            "Full selected P2-v2 weights. Dictionary/alias vectors and type inference must remain off-the-shelf. "
            "Private challenge-derived checkpoint; no training data is part of model uploads.\n")
    for folder in ["cdc", "abbreviations"]:
        for file in sorted((source / "external" / folder).rglob("*")):
            if file.is_file() and file.suffix.lower() in {".xml", ".csv", ".txt", ".md"}:
                copy(file, Path("resources") / file.relative_to(source / "external"))
    for kind in ["pretraining", "posttraining"]:
        subprocess.run([sys.executable, str(root / "tools/export_training_data.py"), kind,
                        str(source), str(private / "datasets" / kind)], check=True)
    files = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(private.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"]
    manifest = dict(format_version=1, private_challenge_material=True, dictionary_hash=dictionary_hash,
        weights_original_bytes_verified=True, selected_lora_epoch=2,
        pretraining_checkpoints="Selected ICD step 1431 and SNOMED step 950 only; epoch 1/2 not saved",
        includes_raw_labelled_csv=False, includes_sealed_test=False, includes_score_caches=False,
        deployment_decision_layers=("Imported frozen DEV fits" if (private/"models/RELEASE.json").exists()
                                    else "Pending main evaluation handoff"), files=files)
    (private / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for file in sorted(private.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(root))
    temporary.replace(args.output)
    digest = sha(args.output)
    args.output.with_suffix(".zip.sha256").write_text(digest + "  " + args.output.name + "\n")
    print("Private artifact ZIP ready:", args.output, "SHA256", digest, flush=True)


if __name__ == "__main__":
    main()
