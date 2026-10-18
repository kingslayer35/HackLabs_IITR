#!/usr/bin/env python3
"""Read-only audit of tracked and commit-eligible files; never prints secrets."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CSV = {
    'examples/demo_input.csv', 'demo/sample.csv',
    'results/Final_all_flows_metrics.csv',
    'results/Latency_and_Footprint_components.csv',
}
PRIVATE_PREFIXES = (
    'private_assets/', 'runtime/', 'submission_bundle/', 'dist/', 'build/',
    'agentic_solution/data/', 'agentic_solution/provenance/',
    'agentic_solution/remote_workspace/',
    'agentic_solution/independent_agent/artifacts/', 'demo/evidence/',
)
PRIVATE_PARTS = {'node_modules', '__pycache__', '.pytest_cache', '.cache', 'logs'}
PAYLOAD_SUFFIXES = {
    '.csv', '.parquet', '.joblib', '.npy', '.npz', '.pt', '.pth', '.ckpt',
    '.safetensors', '.sqlite', '.sqlite3', '.db', '.zip', '.tar', '.gz',
    '.tgz', '.pkl', '.bin', '.arrow', '.feather', '.jsonl', '.pem', '.key',
}
SECRET_PATTERNS = [
    re.compile(rb'sk-or-v1-[A-Za-z0-9_-]{20,}'),
    re.compile(rb'\bhf_[A-Za-z0-9]{30,}'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
    re.compile(rb'\bAKIA[A-Z0-9]{16}\b'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
]


def problems_for(name, path):
    issues = []
    parts = Path(name).parts
    if name.startswith(PRIVATE_PREFIXES) or PRIVATE_PARTS.intersection(parts):
        issues.append('private/generated path')
    if any(p.startswith('.venv') for p in parts):
        issues.append('virtual environment')
    if any(p == '.env' or (p.startswith('.env.') and p != '.env.example') for p in parts):
        issues.append('environment credentials file')
    if path.is_symlink():
        issues.append('symlink requires explicit review')
        return issues
    if path.stat().st_size > 10 * 1024 * 1024:
        issues.append('file exceeds 10 MB')
        return issues
    if path.suffix.lower() in PAYLOAD_SUFFIXES and name not in PUBLIC_CSV:
        issues.append('data/model/archive payload')
    if any(pattern.search(path.read_bytes()) for pattern in SECRET_PATTERNS):
        issues.append('possible credential (content withheld)')
    return issues


def main():
    try:
        output = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
        )
    except subprocess.CalledProcessError:
        print('Run this checker in the actual Git checkout.', file=sys.stderr)
        return 2
    names = sorted(set(output.decode('utf-8', 'surrogateescape').split('\0')) - {''})
    failures = []
    checked = 0
    for name in names:
        path = ROOT / name
        if not path.exists() and not path.is_symlink():
            continue
        checked += 1
        for problem in problems_for(name, path):
            failures.append(f'{name}: {problem}')
    if failures:
        print('\n'.join(failures))
        print(f'FAIL: {len(failures)} issue(s); nothing staged or changed.')
        return 1
    print(f'PASS: {checked} tracked/eligible files checked; nothing staged or changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
