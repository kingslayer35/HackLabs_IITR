"""Private CSV validation and atomic job state shared by API and GPU worker."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 100
ID = re.compile(r"^[0-9a-f]{32}$")
FIELDS = ["chunk_id", "clinical_entity", "chunk_text", "subconcept"]
TERMINAL = {"completed", "failed", "cancelled"}


def parse_csv(raw: bytes) -> list[dict]:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Choose a non-empty CSV file up to 2 MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the file as UTF-8 CSV and try again.") from exc
    if "\x00" in text:
        raise ValueError("The file is not a valid text CSV.")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("CSV column names must be unique.")
        mention = "clinical_entity" if "clinical_entity" in columns else "entity"
        if mention not in columns:
            raise ValueError("Add a clinical_entity column (or entity). Context goes in chunk_text.")
        rows = []
        for i, row in enumerate(reader, 1):
            if i > MAX_ROWS:
                raise ValueError("This demo accepts up to 100 rows per file. Upload a smaller CSV.")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Row {i} has a different number of fields. Quote commas inside text.")
            entity = row[mention]
            if not entity.strip():
                raise ValueError(f"Row {i} needs a clinical_entity value.")
            rows.append(dict(chunk_id=row.get("chunk_id") or str(i), clinical_entity=entity,
                             chunk_text=row.get("chunk_text", ""), subconcept=row.get("subconcept") or "Unknown"))
    except csv.Error as exc:
        raise ValueError("The CSV could not be read. Check its quoting and column structure.") from exc
    if not rows:
        raise ValueError("Add at least one clinical term below the CSV header.")
    return rows


def write_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs = self.root / "jobs"
        self.jobs.mkdir(exist_ok=True, mode=0o700)
        self.database = self.root / "jobs.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, updated REAL, filename TEXT, rows INTEGER, status TEXT, stage TEXT, completed INTEGER, error TEXT, elapsed REAL)")

    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def folder(self, job_id):
        if not isinstance(job_id, str) or not ID.fullmatch(job_id):
            raise KeyError("Run not found.")
        return self.jobs / job_id

    def get(self, job_id):
        self.folder(job_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Run not found.")
        return dict(row)

    def create(self, raw, filename):
        rows = parse_csv(raw)
        filename = str(filename).replace("\\", "/").split("/")[-1][:120]
        filename = "".join(c for c in filename if c.isprintable()) or "clinical_terms.csv"
        job_id = uuid.uuid4().hex
        folder = self.folder(job_id)
        folder.mkdir(mode=0o700)
        with (folder / "input.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (job_id, now, now, filename, len(rows), "uploaded", "uploaded", 0, None, None))
        return self.get(job_id)

    def input_rows(self, job_id):
        self.get(job_id)
        with (self.folder(job_id) / "input.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    def enqueue(self, job_id):
        self.get(job_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            if existing in {"queued", "running", "cancel_requested"}:
                return self.get(job_id)
            if existing != "uploaded":
                raise ValueError("This run has finished. Choose a file to start a new run.")
            if db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0] >= 4:
                raise ValueError("The inference queue is full. Wait for a run to finish and try again.")
            db.execute("UPDATE jobs SET status='queued', stage='queued', updated=? WHERE id=?", (time.time(), job_id))
        return self.get(job_id)

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            job_id = row[0]
            db.execute("UPDATE jobs SET status='running', stage='preparing', updated=? WHERE id=?", (time.time(), job_id))
        return self.get(job_id)

    def progress(self, job_id, stage, completed):
        with self.connect() as db:
            if db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0] == "cancel_requested":
                raise InterruptedError("Run cancelled.")
            db.execute("UPDATE jobs SET stage=?,completed=?,updated=? WHERE id=?", (stage, completed, time.time(), job_id))

    def finish(self, job_id, status, error=None, elapsed=None):
        assert status in TERMINAL
        with self.connect() as db:
            current = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            if current == "cancel_requested":
                status = "cancelled"
            db.execute("UPDATE jobs SET status=?,stage=?,error=?,elapsed=?,updated=? WHERE id=?",
                       (status, status, error, elapsed, time.time(), job_id))

    def cancel(self, job_id):
        self.get(job_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            new = "cancel_requested" if status in {"running", "cancel_requested"} else "cancelled"
            if status not in TERMINAL:
                db.execute("UPDATE jobs SET status=?,stage=?,updated=? WHERE id=?", (new, new, time.time(), job_id))
        return self.get(job_id)

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='failed',stage='failed',error='The inference worker restarted. Upload the file again to retry.',updated=? WHERE status IN ('running','cancel_requested')", (time.time(),))

    def worker_state(self):
        try:
            value = json.loads((self.root / "worker.json").read_text())
            if time.time() - value["heartbeat"] > 20:
                return dict(status="offline", message="Inference worker is offline.")
            return value
        except (FileNotFoundError, ValueError, KeyError):
            return dict(status="offline", message="Inference worker has not started.")
