"""Portable read-only package checksum validation; no dependencies or network."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'provenance/package_manifest.json').read_text())
errors=[];total=0
for relative,item in manifest['files'].items():
    path=root/relative
    if not path.is_file():errors.append(relative+': missing');continue
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    if path.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:errors.append(relative+': mismatch')
    total+=item['bytes']
if errors:
    print('FAILED:',*errors,sep='\n');sys.exit(1)
print(f"VERIFIED {len(manifest['files'])} files, {total:,} bytes; portable SHA-256 integrity passed.")
