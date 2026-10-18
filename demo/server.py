"""Loopback-only demo API and offline static UI. Python standard library only."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .store import JobStore, MAX_BYTES, MAX_ROWS

ROOT = Path(__file__).resolve().parent


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log upload names, medical text or capability URLs.

        def respond(self, status, body, mime="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def dispatch(self):
            path = urlsplit(self.path).path
            if self.command == "POST":
                origin = self.headers.get("Origin")
                if origin and origin not in {"http://" + self.headers.get("Host", ""), "https://" + self.headers.get("Host", "")}:
                    return self.respond(403, {"error": "Open the demo in the same browser origin before submitting a file."})
            if self.command == "GET" and path == "/api/health":
                return self.respond(200, dict(worker=store.worker_state(), max_rows=MAX_ROWS, max_bytes=MAX_BYTES, flow="F11"))
            if self.command == "GET" and path == "/api/sample.csv":
                return self.respond(200, (ROOT / "sample.csv").read_bytes(), "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="termmap_sample.csv"'})
            if self.command == "POST" and path == "/api/uploads":
                if self.headers.get("Content-Type", "").split(";")[0] not in {"text/csv", "application/octet-stream"}:
                    return self.respond(415, {"error": "Upload a CSV file using Content-Type: text/csv."})
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > MAX_BYTES:
                    return self.respond(413, {"error": "Choose a non-empty CSV file up to 2 MB."})
                self.connection.settimeout(20)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("The upload was interrupted. Choose the file again.")
                job = store.create(raw, unquote(self.headers.get("X-Filename", "clinical_terms.csv")))
                job["preview"] = store.input_rows(job["id"])[:3]
                return self.respond(201, job)
            parts = path.strip("/").split("/")
            if len(parts) in {3, 4} and parts[:2] == ["api", "jobs"]:
                job_id = parts[2]
                job = store.get(job_id)
                if self.command == "GET" and len(parts) == 3:
                    if job["status"] == "uploaded":
                        job["preview"] = store.input_rows(job_id)[:3]
                    return self.respond(200, job)
                action = parts[3] if len(parts) == 4 else None
                if self.command == "POST" and action == "run":
                    if store.worker_state()["status"] not in {"ready", "busy"}:
                        return self.respond(409, {"error": "The inference worker is not ready. Your file is saved; try again when the status is Ready."})
                    return self.respond(202, store.enqueue(job_id))
                if self.command == "POST" and action == "cancel":
                    return self.respond(200, store.cancel(job_id))
                if self.command == "GET" and action in {"results", "download"}:
                    if job["status"] != "completed":
                        return self.respond(409, {"error": "Results are available after the run completes."})
                    folder = store.folder(job_id)
                    if action == "results":
                        return self.respond(200, json.loads((folder / "results.json").read_text()))
                    return self.respond(200, (folder / "predictions.csv").read_bytes(), "text/csv; charset=utf-8", {"Content-Disposition": 'attachment; filename="termmap_predictions.csv"'})
            assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/styles.css": ("styles.css", "text/css; charset=utf-8"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}
            if self.command == "GET" and path in assets:
                name, mime = assets[path]
                return self.respond(200, (ROOT / "static" / name).read_bytes(), mime)
            return self.respond(404, {"error": "Page or run not found."})

        def handle_request(self):
            try:
                self.dispatch()
            except KeyError:
                self.respond(404, {"error": "Run not found. Choose a file to start a new run."})
            except ValueError as exc:
                self.respond(400, {"error": str(exc)})
            except TimeoutError:
                self.respond(408, {"error": "The upload timed out. Choose the file again."})
            except Exception:
                self.respond(500, {"error": "The server could not complete this request. Your existing run is retained; try again."})

        do_GET = handle_request
        do_POST = handle_request

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default=str(ROOT.parent / "runtime" / "demo"))
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    import os
    os.umask(0o077)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(JobStore(args.runtime)))
    print(f"TermMap demo: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
