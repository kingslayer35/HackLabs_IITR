"""CPU-only packaging audit; no clinical rows are printed."""
import ast
import json
from pathlib import Path
import re


def main():
    root=Path(__file__).resolve().parents[1]
    missing=[];files=0
    for folder in [root/"termmap",root/"scripts"]:
        for file in folder.rglob("*.py"):
            files+=1
            for node in ast.walk(ast.parse(file.read_text())):
                names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                       [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
                for name in names:
                    if name.startswith(("termmap.","scripts.")):
                        path=root/name.replace(".","/")
                        if not path.with_suffix(".py").exists() and not (path/"__init__.py").exists():
                            missing.append((str(file.relative_to(root)),name))
    if missing:raise RuntimeError("Missing internal imports: "+str(missing))
    bad_links=[]
    for folder in [root,root/"docs",root/"results",root/"termmap",root/"scripts"]:
        paths=folder.glob("*.md") if folder in [root,root/"docs",root/"results"] else folder.rglob("*.md")
        for file in paths:
            for link in re.findall(r"\]\(([^)]+)\)",file.read_text()):
                if "://" in link or link.startswith("#"):continue
                path=(file.parent/link.split("#")[0]).resolve()
                if not path.exists():bad_links.append((str(file.relative_to(root)),link))
    if bad_links:raise RuntimeError("Broken local documentation links: "+str(bad_links))
    manifest=json.loads((root/"private_assets/MANIFEST.json").read_text())
    assert not manifest["includes_raw_labelled_csv"] and not manifest["includes_sealed_test"]
    assert not manifest["includes_score_caches"]
    print("Packaging audit passed:",files,"Python source files; internal imports and documentation links resolve")


if __name__=="__main__":main()
