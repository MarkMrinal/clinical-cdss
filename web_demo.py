"""Serve the local free-text CDSS demo and replay saved cases on localhost."""
from __future__ import annotations

import json
import os
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "results" / "execution_logs"
CASES = {
    "medqa_347": {"label": "Single-pass retrieval", "subtitle": "Primary hyperparathyroidism"},
    "medqa_107": {"label": "Adaptive re-query", "subtitle": "Klinefelter syndrome"},
}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
ANALYSIS_QUEUE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cdss-analysis")
LLM_MODULE_PATH: str | None = None


def _run_analysis(job_id: str, case_id: str, question: str) -> None:
    try:
        from src.pipeline import run_case

        result = run_case(question, options=None, mode="full", case_id=case_id, save_log=True)
        with JOBS_LOCK:
            JOBS[job_id].update(status="complete", result=result)
    except Exception as exc:  # noqa: BLE001 - surfaced in the local demo UI
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", error=str(exc))


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ClinicalCDSSDemo/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/version":
            self._send_json({
                "version": "header-auth-v1",
                "llm_module": LLM_MODULE_PATH,
            })
            return
        if path == "/" or path == "/index.html":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/api/cases/"):
            case_id = path.removeprefix("/api/cases/")
            if case_id not in CASES:
                self._send_json({"error": "Unknown demo case"}, 404)
                return
            log_path = LOG_DIR / f"{case_id}.json"
            if not log_path.is_file():
                self._send_json({"error": f"Saved log not found: {case_id}"}, 404)
                return
            try:
                case = json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._send_json({"error": f"Could not read saved log: {exc}"}, 500)
                return
            case["demo"] = CASES[case_id]
            self._send_json(case)
            return
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                snapshot = dict(job) if job else None
            if snapshot is None:
                self._send_json({"error": "Unknown analysis job"}, 404)
                return
            self._send_json(snapshot)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/analyze":
            self._send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384:
                self._send_json({"error": "Enter a case description under 16 KB."}, 400)
                return
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                self._send_json({"error": "Request must be a JSON object."}, 400)
                return
            question = str(body.get("question", "")).strip()
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Request must contain valid JSON."}, 400)
            return
        if len(question) < 20:
            self._send_json({"error": "Add a little more case detail before analysis."}, 400)
            return

        job_id = uuid.uuid4().hex
        case_id = f"live_{job_id[:10]}"
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "running", "case_id": case_id}
        ANALYSIS_QUEUE.submit(_run_analysis, job_id, case_id, question)
        self._send_json({"job_id": job_id, "status": "running"}, 202)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self._send_json({"error": "Demo page is missing"}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


class DemoHTTPServer(ThreadingHTTPServer):
    # On Windows, exclusive binding prevents an old server instance from
    # silently sharing the demo port with a newly started version.
    allow_reuse_address = False

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()


def main() -> None:
    global LLM_MODULE_PATH
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    print(f"Clinical CDSS case viewer: http://{host}:{port}")
    print("Free-text analysis uses the configured pipeline; saved demonstrations replay local logs.")
    import src.llm

    LLM_MODULE_PATH = str(Path(src.llm.__file__).resolve())
    print(f"LLM module loaded from: {LLM_MODULE_PATH}")
    try:
        DemoHTTPServer((host, port), DemoHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nDemo server stopped.")


if __name__ == "__main__":
    main()
