#!/usr/bin/env python3
"""Ollama API proxy — translates llama.cpp responses to exact Ollama format.

Sits between clients and llama-server:
  Client -> :11434 (this proxy) -> :8081 (llama-server)

Translates /api/chat, /api/generate, /api/tags, /api/show responses.
Passes /v1/* routes through unchanged.
"""

import json
import sys
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

BACKEND_PORT = 8081
LISTEN_PORT = 11434
LISTEN_HOST = "0.0.0.0"


class OllamaProxy(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # quieter logs: just method + path
        sys.stderr.write(f"{self.command} {self.path}\n")

    # --- GET ---

    def do_GET(self):
        if self.path == "/":
            # Ollama returns "Ollama is running"
            self._respond(200, "application/json", json.dumps({"status": "Ollama is running (ollama-rdna1)"}))
            return

        if self.path == "/api/tags":
            resp = self._forward("GET", "/api/tags")
            if resp is None:
                return
            try:
                data = json.loads(resp)
                # llama.cpp /api/tags already returns {"models": [...]} but model entries
                # may differ. Normalize each model entry to Ollama format.
                models = data.get("models", data.get("data", []))
                ollama_models = []
                for m in models:
                    ollama_models.append({
                        "name": m.get("name", m.get("id", "unknown")),
                        "model": m.get("model", m.get("id", "unknown")),
                        "modified_at": m.get("modified_at", ""),
                        "size": m.get("size", 0),
                        "digest": m.get("digest", ""),
                        "details": m.get("details", {}),
                    })
                self._respond(200, "application/json", json.dumps({"models": ollama_models}))
            except (json.JSONDecodeError, KeyError):
                self._respond(200, "application/json", resp)
            return

        if self.path == "/api/version":
            self._respond(200, "application/json", json.dumps({"version": "ollama-rdna1-0.1.0"}))
            return

        # passthrough
        resp = self._forward("GET", self.path)
        if resp is not None:
            self._respond(200, "application/json", resp)

    # --- POST ---

    def do_POST(self):
        body = self._read_body()
        if body is None:
            return

        if self.path == "/api/chat":
            self._handle_chat(body)
        elif self.path == "/api/generate":
            self._handle_generate(body)
        elif self.path == "/api/show":
            self._handle_show(body)
        else:
            # passthrough for /v1/*, /completion, etc.
            resp = self._forward("POST", self.path, body)
            if resp is not None:
                self._respond(200, "application/json", resp)

    # --- /api/chat ---

    def _handle_chat(self, body):
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, "application/json", json.dumps({"error": "invalid JSON"}))
            return

        stream = req.get("stream", True)
        model_name = req.get("model", "")

        # Build llama.cpp /v1/chat/completions request
        llama_req = {
            "messages": req.get("messages", []),
            "model": model_name,
            "stream": False,  # always non-stream to backend, we format ourselves
        }
        # forward optional params
        for key in ("temperature", "top_p", "top_k", "seed", "num_predict",
                     "stop", "repeat_penalty", "presence_penalty", "frequency_penalty"):
            if key in req:
                # Ollama uses num_predict, OpenAI uses max_tokens
                if key == "num_predict":
                    llama_req["max_tokens"] = req[key]
                else:
                    llama_req[key] = req[key]
        if "options" in req:
            opts = req["options"]
            for key in ("temperature", "top_p", "top_k", "seed", "num_predict",
                         "stop", "repeat_penalty"):
                if key in opts:
                    if key == "num_predict":
                        llama_req["max_tokens"] = opts[key]
                    else:
                        llama_req[key] = opts[key]

        t0 = time.time_ns()
        resp = self._forward("POST", "/v1/chat/completions", json.dumps(llama_req))
        t1 = time.time_ns()

        if resp is None:
            return

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            self._respond(502, "application/json", json.dumps({"error": "bad backend response"}))
            return

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        ollama_resp = {
            "model": data.get("model", model_name),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            },
            "done_reason": "stop" if choice.get("finish_reason") == "stop" else choice.get("finish_reason", "stop"),
            "done": True,
            "total_duration": t1 - t0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

        if stream:
            self._stream_ollama_response(ollama_resp)
        else:
            self._respond(200, "application/json", json.dumps(ollama_resp))

    # --- /api/generate ---

    def _handle_generate(self, body):
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, "application/json", json.dumps({"error": "invalid JSON"}))
            return

        stream = req.get("stream", True)
        model_name = req.get("model", "")

        # Build llama.cpp /completion request
        llama_req = {
            "prompt": req.get("prompt", ""),
            "stream": False,
        }
        if "system" in req:
            llama_req["prompt"] = req["system"] + "\n" + llama_req["prompt"]
        for key in ("temperature", "top_p", "top_k", "seed", "stop", "repeat_penalty"):
            if key in req:
                llama_req[key] = req[key]
            elif "options" in req and key in req["options"]:
                llama_req[key] = req["options"][key]
        if req.get("num_predict") or (req.get("options") or {}).get("num_predict"):
            llama_req["n_predict"] = req.get("num_predict") or req["options"]["num_predict"]

        t0 = time.time_ns()
        resp = self._forward("POST", "/completion", json.dumps(llama_req))
        t1 = time.time_ns()

        if resp is None:
            return

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            self._respond(502, "application/json", json.dumps({"error": "bad backend response"}))
            return

        ollama_resp = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "response": data.get("content", ""),
            "done": True,
            "done_reason": "stop",
            "total_duration": t1 - t0,
            "prompt_eval_count": data.get("tokens_evaluated", 0),
            "eval_count": data.get("tokens_predicted", 0),
        }

        if stream:
            self._stream_ollama_response(ollama_resp)
        else:
            self._respond(200, "application/json", json.dumps(ollama_resp))

    # --- /api/show ---

    def _handle_show(self, body):
        resp = self._forward("POST", "/api/show", body)
        if resp is not None:
            self._respond(200, "application/json", resp)

    # --- streaming ---

    def _stream_ollama_response(self, final_resp):
        """Ollama streaming: NDJSON lines, last one has done:true."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # For non-streaming backend, emit content as single chunk then done
        content = final_resp.get("message", {}).get("content") or final_resp.get("response", "")

        if "message" in final_resp:
            # chat format
            chunk = {
                "model": final_resp["model"],
                "created_at": final_resp["created_at"],
                "message": {"role": "assistant", "content": content},
                "done": False,
            }
            self._write_chunk(json.dumps(chunk) + "\n")
            final_resp["message"]["content"] = ""
        else:
            # generate format
            chunk = {
                "model": final_resp["model"],
                "created_at": final_resp["created_at"],
                "response": content,
                "done": False,
            }
            self._write_chunk(json.dumps(chunk) + "\n")
            final_resp["response"] = ""

        # final done message
        self._write_chunk(json.dumps(final_resp) + "\n")
        self._write_chunk("")  # end chunked transfer

    def _write_chunk(self, data):
        encoded = data.encode()
        self.wfile.write(f"{len(encoded):x}\r\n".encode())
        self.wfile.write(encoded)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    # --- helpers ---

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return self.rfile.read(length).decode()
        return ""

    def _forward(self, method, path, body=None):
        url = f"http://127.0.0.1:{BACKEND_PORT}{path}"
        try:
            req = urllib.request.Request(url, method=method)
            req.add_header("Content-Type", "application/json")
            data = body.encode() if body else None
            with urllib.request.urlopen(req, data, timeout=300) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else json.dumps({"error": str(e)})
            self._respond(e.code, "application/json", error_body)
            return None
        except urllib.error.URLError:
            self._respond(502, "application/json", json.dumps({"error": "backend not reachable"}))
            return None

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        encoded = body.encode()
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    global BACKEND_PORT
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN_PORT
    backend = int(sys.argv[2]) if len(sys.argv) > 2 else BACKEND_PORT
    BACKEND_PORT = backend

    server = HTTPServer((LISTEN_HOST, port), OllamaProxy)
    print(f"ollama-proxy listening on {LISTEN_HOST}:{port} -> backend :{backend}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
