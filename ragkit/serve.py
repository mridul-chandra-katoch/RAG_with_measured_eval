import json
import os
import secrets
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ragkit.agent import run_agent
from ragkit.cli import build_pipeline
from ragkit.config import Config
from ragkit.chunk import chunk_documents

# Generate a random 8-character token every time the server starts
SERVER_UPLOAD_TOKEN = secrets.token_hex(4)

class Handler(BaseHTTPRequestHandler):
    pipeline = None
    config_path = "configs/demo.json"
    
    def log_message(self, *args): pass
    
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        
        if self.path == "/upload":
            # SECURITY FIX: Require the upload token
            if self.headers.get("X-Upload-Token") != SERVER_UPLOAD_TOKEN:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized: Missing or invalid upload token.")
                return

            filename = self.headers.get("X-Filename", "uploaded.txt")
            body = self.rfile.read(length)
            
            os.makedirs("data", exist_ok=True)
            safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
            filepath = os.path.join("data", safe_name)
            
            with open(filepath, "wb") as f:
                f.write(body)
                
            Handler.pipeline = build_pipeline(Handler.config_path)
            
            cfg = Config.load(Handler.config_path)
            all_chunks = chunk_documents(Path("data"), cfg.max_tokens, cfg.overlap_tokens)
            file_chunks = [c for c in all_chunks if c.source["file"] == safe_name]
            
            if len(file_chunks) == 0:
                msg = f"Uploaded {safe_name}, but extracted 0 chunks. (Is it a scanned image PDF?)"
            else:
                msg = f"Successfully indexed {safe_name} ({len(file_chunks)} chunks)."
                
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": msg}).encode("utf-8"))
            return

        if self.path == "/ask":
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                ans, tool = run_agent(data["question"], self.pipeline)
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
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return
            
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

def serve(pipeline, port=8000):
    Handler.pipeline = pipeline
    # SECURITY FIX: Bind to 127.0.0.1 to prevent external network access
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving on http://127.0.0.1:{port}")
    print(f"UPLOAD TOKEN: {SERVER_UPLOAD_TOKEN} (You must enter this in the UI to upload files)")
    server.serve_forever()