"""Add supplied private artifact/model/demo links without using credentials."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--drive")
    parser.add_argument("--sha256")
    parser.add_argument("--sapbert-icd")
    parser.add_argument("--sapbert-snomed")
    parser.add_argument("--qwen-epoch1")
    parser.add_argument("--qwen-epoch2")
    parser.add_argument("--demo")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    path=root/"docs/ARTIFACT_LINKS.json"
    links=json.loads(path.read_text())
    fields={"drive":"private_drive_bundle_url","sha256":"private_drive_bundle_sha256",
        "sapbert_icd":"huggingface_sapbert_icd","sapbert_snomed":"huggingface_sapbert_snomed",
        "qwen_epoch1":"huggingface_qwen_lora_epoch1","qwen_epoch2":"huggingface_qwen_lora_epoch2","demo":"demo_video_url"}
    for option,key in fields.items():
        value=getattr(args,option)
        if value is not None:
            if option!="sha256" and not value.startswith("https://"):
                parser.error("Provide an HTTPS sharing link")
            if option=="sha256" and (len(value)!=64 or any(c not in "0123456789abcdef" for c in value)):
                parser.error("SHA256 must be 64 lowercase hexadecimal characters")
            links[key]=value
    path.write_text(json.dumps(links,indent=2)+"\n")
    labels={"private_drive_bundle_url":"Private artifact ZIP","huggingface_sapbert_icd":"SapBERT ICD query encoder",
        "huggingface_sapbert_snomed":"SapBERT SNOMED query encoder","huggingface_qwen_lora_epoch1":"Qwen LoRA epoch 1",
        "huggingface_qwen_lora_epoch2":"Qwen LoRA epoch 2 (selected)","demo_video_url":"Demo video"}
    rows=["| Artifact | Link |","|---|---|"]
    for key,label in labels.items():
        value=links[key]
        rows.append(f"| {label} | {'['+label+']('+value+')' if value else 'Pending upload / access link'} |")
    if links["private_drive_bundle_sha256"]:
        rows.extend(["", "ZIP SHA256: `"+links["private_drive_bundle_sha256"]+"`."])
    readme=root/"README.md";text=readme.read_text()
    start="<!-- ARTIFACT_LINKS_START -->";end="<!-- ARTIFACT_LINKS_END -->"
    before,rest=text.split(start,1);_,after=rest.split(end,1)
    readme.write_text(before+start+"\n"+"\n".join(rows)+"\n"+end+after)


if __name__=="__main__":main()
