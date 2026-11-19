#!/usr/bin/env python3
"""Minimal multi-session LLM serving prototype over the NPUW pipeline.

Architecture (single-pipeline mode, the recommended one):
  - One persistent LLMPipeline owned by exactly one worker thread.
  - Client requests arrive via HTTP (ThreadingHTTPServer), are validated,
    and enqueued; the worker executes them strictly sequentially.
  - Session state is an explicit per-session token history; every request
    re-prefills its full history (stateless per request) — request isolation
    by construction: no pipeline state survives a request.
  - Prompts exceeding MAX_PROMPT_LEN are rejected with HTTP 400 + a JSON
    error body before touching the pipeline.
  - SIGTERM/SIGINT => graceful shutdown: stop accepting, drain queue,
    finish in-flight request, exit.

Two-pipeline mode (--workers 2) exists to benchmark NPU behavior with two
concurrent compiled pipelines; it is NOT recommended by default (see
serving.md).

Endpoints:
  POST /generate        {"session": "id", "prompt": "...", "max_new_tokens": 32}
  POST /reset           {"session": "id"}
  GET  /health          {"ok": true, "requests_done": N, "queue_depth": M}
  POST /shutdown        graceful shutdown

Usage:
  server.py [--port 8931] [--workers 1] [--max-prompt-len 64]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if __name__ != '__main__':  # GenAI forkserver re-import guard
    raise SystemExit(0)

import openvino_genai as genai

MAX_PROMPT_LEN = 64
HISTORY_LIMIT_TOKENS = 1024  # sessions refuse to grow beyond this


class Worker:
    """Owns one LLMPipeline; executes queued requests sequentially."""

    def __init__(self, worker_id: int, max_prompt_len: int):
        self.id = worker_id
        self.max_prompt_len = max_prompt_len
        self.q: 'queue.Queue[dict | None]' = queue.Queue()
        self.requests_done = 0
        self.errors = 0
        t0 = time.perf_counter()
        props = dict(MAX_PROMPT_LEN=max_prompt_len, MIN_RESPONSE_LEN=64,
                     GENERATE_HINT='BEST_PERF')
        cache = os.environ.get('NPUW_CACHE_DIR')
        if cache:
            props['CACHE_DIR'] = cache
        ai_root = os.environ.get('FLOSSWARE_AI_ROOT', os.path.expanduser('~/AI'))
        model_dir = os.environ.get('NPU_MODEL_DIR', f'{ai_root}/models/openvino/qwen3-0.6b-int4')
        self.pipe = genai.LLMPipeline(model_dir, device='NPU', **props)
        self.load_s = time.perf_counter() - t0
        self.tokenizer = self.pipe.get_tokenizer()
        self.thread = threading.Thread(target=self._run, name=f'npu-worker-{worker_id}',
                                       daemon=True)
        self.thread.start()

    def count_tokens(self, text: str) -> int:
        import numpy as np
        return int(np.asarray(self.tokenizer.encode(text).input_ids.data).size)

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                t0 = time.perf_counter()
                out = self.pipe.generate(item['prompt'],
                                         max_new_tokens=item['max_new_tokens'],
                                         do_sample=False)
                item['result'] = {
                    'text': out,
                    'gen_s': round(time.perf_counter() - t0, 3),
                    'worker': self.id,
                }
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                item['error'] = f'{type(e).__name__}: {e}'
            finally:
                self.requests_done += 1
                item['done'].set()

    def submit(self, req: dict) -> None:
        self.q.put(req)

    def queue_depth(self) -> int:
        return self.q.qsize()


class SessionStore:
    """Explicit per-session history; requests are self-contained."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, list[str]] = {}

    def build_prompt(self, session: str, prompt: str, counter) -> str:
        with self._lock:
            hist = self._history.setdefault(session, [])
            full = ''.join(hist) + prompt
            n_tok = counter(full)
            if n_tok > HISTORY_LIMIT_TOKENS:
                raise ValueError(f'session history ({n_tok} tokens) exceeds '
                                 f'{HISTORY_LIMIT_TOKENS}-token limit; reset the session')
            return full

    def record(self, session: str, prompt: str, answer: str) -> None:
        with self._lock:
            self._history.setdefault(session, []).extend([prompt, answer])

    def reset(self, session: str) -> None:
        with self._lock:
            self._history.pop(session, None)


class ServerState:
    workers: list[Worker] = []
    sessions: SessionStore = None  # type: ignore[assignment]
    rr = 0
    shutdown = False

    @classmethod
    def next_worker(cls) -> Worker:
        w = cls.workers[cls.rr % len(cls.workers)]
        cls.rr += 1
        return w


def make_handler(state: type[ServerState]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self._json(200, {
                    'ok': True,
                    'workers': len(state.workers),
                    'requests_done': [w.requests_done for w in state.workers],
                    'errors': [w.errors for w in state.workers],
                    'queue_depth': [w.queue_depth() for w in state.workers],
                })
            else:
                self._json(404, {'error': 'not_found'})

        def do_POST(self):
            if state.shutdown:
                self._json(503, {'error': 'shutting_down'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                payload = json.loads(self.rfile.read(length) or b'{}')
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {'error': f'bad_request: {e}'})
                return

            if self.path == '/shutdown':
                self._json(200, {'ok': True, 'detail': 'shutting down'})
                threading.Thread(target=graceful_shutdown, daemon=True).start()
                return

            if self.path == '/reset':
                session = str(payload.get('session', ''))
                if not session:
                    self._json(400, {'error': 'missing session'})
                    return
                state.sessions.reset(session)
                self._json(200, {'ok': True})
                return

            if self.path != '/generate':
                self._json(404, {'error': 'not_found'})
                return

            session = str(payload.get('session', 'default'))
            prompt = payload.get('prompt')
            max_new = int(payload.get('max_new_tokens', 32))
            if not isinstance(prompt, str) or not prompt:
                self._json(400, {'error': 'prompt must be a non-empty string'})
                return
            if not (1 <= max_new <= 256):
                self._json(400, {'error': 'max_new_tokens out of range [1, 256]'})
                return

            try:
                full_prompt = state.sessions.build_prompt(
                    session, prompt, state.workers[0].count_tokens)
            except ValueError as e:
                self._json(400, {'error': str(e)})
                return

            n_tok = state.workers[0].count_tokens(full_prompt)
            if n_tok > state.workers[0].max_prompt_len:
                self._json(400, {
                    'error': f'prompt_too_long: {n_tok} tokens > MAX_PROMPT_LEN '
                             f'{state.workers[0].max_prompt_len}; reset session or shorten',
                })
                return

            req = {'prompt': full_prompt, 'max_new_tokens': max_new,
                   'done': threading.Event()}
            state.next_worker().submit(req)
            if not req['done'].wait(timeout=300):
                self._json(504, {'error': 'timeout'})
                return
            if 'error' in req:
                self._json(500, {'error': req['error']})
                return
            state.sessions.record(session, prompt, req['result']['text'])
            self._json(200, {'session': session, 'prompt_tokens': n_tok,
                             **req['result']})

    return Handler


def graceful_shutdown() -> None:
    state = ServerState
    if state.shutdown:
        return
    state.shutdown = True
    # Drain: enqueue sentinels; workers exit after finishing in-flight work.
    for w in state.workers:
        w.submit(None)
    for w in state.workers:
        w.thread.join(timeout=60)
    os._exit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8931)
    ap.add_argument('--workers', type=int, default=1, choices=(1, 2))
    ap.add_argument('--max-prompt-len', type=int, default=64)
    args = ap.parse_args()

    global MAX_PROMPT_LEN
    MAX_PROMPT_LEN = args.max_prompt_len

    ServerState.sessions = SessionStore()
    for i in range(args.workers):
        t0 = time.perf_counter()
        ServerState.workers.append(Worker(i, args.max_prompt_len))
        print(f'[worker {i}] pipeline ready in {time.perf_counter()-t0:.1f}s',
              flush=True)

    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(ServerState))
    signal.signal(signal.SIGTERM, lambda *_: graceful_shutdown())
    signal.signal(signal.SIGINT, lambda *_: graceful_shutdown())
    print(f'[server] listening on 127.0.0.1:{args.port} '
          f'({args.workers} worker(s), mpl={args.max_prompt_len})', flush=True)
    try:
        httpd.serve_forever()
    finally:
        graceful_shutdown()


if __name__ == '__main__':
    main()
