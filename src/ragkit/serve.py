import json
import os
import secrets
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ragkit.agent import run_agent
from ragkit.cli import build_pipeline
from ragkit.observe import log_trace

SERVER_UPLOAD_TOKEN = secrets.token_hex(16)
MAX_UPLOAD_SIZE = 5 * 1024 * 1024
RATE_LIMIT_DICT = {}

class Handler(BaseHTTPRequestHandler):
    pipeline = None
    config_path = "configs/corpus.yaml"
    
    def log_message(self, *args): pass
    
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        elif self.path == "/runs":
            runs = [f.name for f in Path("results").glob("*.json")] if Path("results").exists() else []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"runs": runs}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            
    def do_POST(self):
        client_ip = self.client_address[0]
        now = time.time()
        
        if client_ip not in RATE_LIMIT_DICT:
            RATE_LIMIT_DICT[client_ip] = []
        RATE_LIMIT_DICT[client_ip] = [t for t in RATE_LIMIT_DICT[client_ip] if now - t < 60]
        if len(RATE_LIMIT_DICT[client_ip]) >= 20:
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b"Too Many Requests")
            return
        RATE_LIMIT_DICT[client_ip].append(now)

        length = int(self.headers.get("Content-Length", 0))
        
        if self.path == "/upload":
            if self.headers.get("X-Upload-Token") != SERVER_UPLOAD_TOKEN:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized")
                return
            if length > MAX_UPLOAD_SIZE:
                self.send_response(413)
                self.end_headers()
                self.wfile.write(b"Payload Too Large")
                return

            filename = self.headers.get("X-Filename", "uploaded.txt")
            body = self.rfile.read(length)
            
            os.makedirs("data", exist_ok=True)
            safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
            filepath = os.path.join("data", safe_name)
            
            with open(filepath, "wb") as f:
                f.write(body)
                
            Handler.pipeline = build_pipeline(Handler.config_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": f"Uploaded {safe_name}"}).encode("utf-8"))
            return

        if self.path == "/ask":
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                start_t = time.time()
                ans, tool = run_agent(data["question"], self.pipeline)
                latency = (time.time() - start_t) * 1000
                
                # H2: Observability trace
                log_trace(ans.trace_id, data["question"], latency, 0.0, [c.chunk_id for c in ans.citations])
                
                payload = json.dumps({
                    "answer": ans.answer,
                    "tool": tool,
                    "abstained": ans.abstained,
                    "citations": [
                        {"chunk_id": c.chunk_id, "file": c.file, "page": c.page, "quote": c.quote}
                        for c in ans.citations
                    ],
                    "confidence": ans.confidence,
                    "trace_id": ans.trace_id,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Internal Server Error")
            return
            
        if self.path == "/feedback":
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                Path("results").mkdir(exist_ok=True)
                with open("results/feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Feedback recorded")
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Error saving feedback")
            return
            
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

def serve(pipeline, port=8000):
    Handler.pipeline = pipeline
    host = "0.0.0.0" if os.environ.get("DOCKER") == "1" else "127.0.0.1"
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    print(f"UPLOAD TOKEN: {SERVER_UPLOAD_TOKEN}")
    server.serve_forever()