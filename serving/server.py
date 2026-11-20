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
  - SIGTERM/SIGINT => the orderly shutdown sequence described below.

Two-pipeline mode (--workers 2) exists to benchmark NPU behavior with two
concurrent compiled pipelines; it is NOT recommended by default (see
serving.md).

Ordering semantics (issue #4)
-----------------------------
* Requests to the **same** session are strictly serialized in
  gate-acquisition order (i.e. the order in which their HTTP handlers reach
  the per-session gate; for a single client this is arrival order).  The gate
  is held from before the history snapshot is built until the generated
  answer has been appended to the history, so prompt construction and history
  recording are atomic with respect to other same-session operations.
  Exception: a request that hits ``GENERATE_TIMEOUT_S`` gets HTTP 504 and
  releases the gate immediately; if its generation had already started it
  runs to completion but its result is discarded (never recorded), so
  history order is unaffected.
* Requests to **different** sessions never share a gate and may execute
  concurrently, limited only by the number of pipelines/workers.
* ``POST /reset`` participates in the same per-session order: it **waits**
  for any earlier-ticketed generation for that session (whether queued or
  in-flight) to finish and *then* clears the history.  It is never
  interleaved with a generation and never rejected with 409.
* At most ``MAX_SESSION_PENDING`` (8) operations may hold or wait for one
  session's gate; further ``/generate`` or ``/reset`` calls for that session
  are rejected immediately with HTTP 429 (no ticket taken, nothing mutated),
  so same-session floods cannot bypass the bounded queue.

Session capacity and eviction (issue #12)
-----------------------------------------
* ``SessionStore`` is bounded: at most ``--max-sessions`` (env
  ``SESSIONS_MAX``, default 1024) sessions retain history and a gate.
* Admitting a *new* session while at capacity evicts the idle entry whose
  last operation completed longest ago (LRU by completion time; creation
  time is the origin for a session that never completed an operation).
  Ties are broken by session id so eviction is deterministic.
* An entry is **active** — and therefore pinned, never evictable — from the
  moment admission takes its gate ticket until :meth:`SessionStore.finish`
  releases it.  That covers waiting on the gate, queued work, in-flight
  generation and reset.  A 504-abandoned generation releases its ticket
  immediately (its result is never recorded), so it no longer pins state.
* Admission (lookup-or-create, eviction, ticket) is a single critical
  section, so concurrent first requests for one session share exactly one
  gate and the limit can never be bypassed.
* If **every** stored session is active, a request for a new session is
  rejected with HTTP 429 (``overloaded: session capacity ...``); nothing is
  evicted and existing sessions are unaffected.
* Eviction is invisible to the client: an evicted session id stays valid,
  the next request creates fresh state (empty history) and is served
  normally — never a 4xx just because the session was evicted.

Queueing and validation (issue #7)
----------------------------------
* Every worker has a bounded FIFO queue: ``--max-queue`` (default 32),
  overridable with the ``QUEUE_MAX`` environment variable.
* When all worker queues are full the request is rejected with HTTP 429 and
  ``{"error": "overloaded: ..."}``.  A rejected request never enters a queue
  and never touches session history.
* Requests are validated *before* queueing: the body must be a JSON object;
  ``session`` a non-empty printable string of at most 128 characters;
  ``prompt`` a non-empty string; ``max_new_tokens`` an int (not a bool) in
  [1, 256].  Violations return HTTP 400 and mutate nothing.

KV-cache capacity (issue #13)
-----------------------------
* The NPUW KV window holds ``MAX_PROMPT_LEN + MIN_RESPONSE_LEN`` tokens
  (prompt + generated combined; 64 + 64 = 128 by default).  GenAI stops a
  generation gracefully when it reaches that window, returning a shorter,
  prefix-exact response with no error (docs/npu-early-stop-investigation.md,
  validation/results/boundary-probes.json).
* So that this truncation is never silent, a request whose counted prompt
  tokens (history included) plus ``max_new_tokens`` exceed that capacity is
  rejected with HTTP 400 ``exceeds_kv_capacity: ...`` before queueing.
* The counted prompt is the raw text; ``generate()`` additionally applies the
  model's chat template, which adds tokens.  The check therefore rejects only
  requests that would certainly be truncated; an accepted request close to
  the boundary may still receive fewer than ``max_new_tokens`` tokens.

Shutdown policy (issue #3)
--------------------------
``POST /shutdown``, SIGTERM and SIGINT all run the same idempotent sequence:
  1. ``httpd.shutdown()`` — stop accepting/serving new HTTP requests.  The
     listening socket is closed once workers have been joined.
  2. Requests still waiting in a worker queue are **cancelled**: they are
     answered with HTTP 503 and never run.
  3. The request already executing on each worker is allowed to **finish**.
  4. Workers are joined with a configurable drain timeout
     (``--drain-timeout``, default 30 s, env ``DRAIN_TIMEOUT_S``).  If a
     worker does not stop in time the event is logged to stderr and recorded
     in the shutdown report (``timed_out`` / ``unstopped_workers``); there is
     no forced process exit.  HTTP handlers are then given until the drain
     deadline plus ``HANDLER_GRACE_S`` (2 s) to write their responses
     (finished in-flight results, 503 cancellations); any still running are
     reported as ``unfinished_handlers``.
  5. The HTTP server is closed and the process returns normally.  A second
     shutdown request is acknowledged with HTTP 200 and does not restart the
     sequence; concurrent callers wait for the first call to finish.

Endpoints:
  POST /generate        {"session": "id", "prompt": "...", "max_new_tokens": 32}
  POST /reset           {"session": "id"}
  GET  /health          {"ok": true, "requests_done": [...], "queue_depth": [...],
                         "queue_capacity": [...], "queue_depth_total": N,
                         "sessions": N, "sessions_active": N,
                         "session_capacity": N, "session_evictions": N,
                         "kv_capacity": N, ...}
  POST /shutdown        orderly shutdown

Usage:
  server.py [--port 8931] [--workers 1] [--device NPU|GPU] [--f32] [--max-prompt-len 64]
            [--max-queue 32] [--max-sessions 1024] [--drain-timeout 30]
            [--history-limit 1024]
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

# The OpenVINO GenAI runtime is imported lazily in _load_pipeline().  The
# guard below preserves the original "do not re-run the server in a GenAI
# forkserver child" behaviour; tests import this module with
# SERVING_ALLOW_IMPORT=1 and inject a stub pipeline, so no NPU runtime (and
# no numpy) is required.
if __name__ != '__main__' and os.environ.get('SERVING_ALLOW_IMPORT') != '1':
    raise SystemExit(0)

MAX_PROMPT_LEN = 64
HISTORY_LIMIT_TOKENS = 1024  # sessions refuse to grow beyond this
MAX_SESSION_ID_LEN = 128
MAX_NEW_TOKENS_MIN = 1
MAX_NEW_TOKENS_MAX = 256
# Passed to the pipeline as MIN_RESPONSE_LEN; with MAX_PROMPT_LEN it fixes the
# NPUW KV window (prompt + generated tokens) -- see "KV-cache capacity".
MIN_RESPONSE_LEN = 64
MAX_BODY_BYTES = 1 << 20
GENERATE_TIMEOUT_S = 300.0
# Max operations (holding + waiting) per session gate.  Requests waiting on a
# session gate are not in a worker queue, so without this cap a flood of
# same-session requests would grow blocked handler threads without bound and
# bypass the bounded queue (issue #7).
MAX_SESSION_PENDING = 8
# Grace period, after the drain deadline, for HTTP handlers to write the
# responses of work that finished (or was cancelled) during the drain.
HANDLER_GRACE_S = 2.0
SENTINEL = object()  # worker stop marker


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_MAX_QUEUE = max(1, _env_int('QUEUE_MAX', 32))
DEFAULT_DRAIN_TIMEOUT_S = max(0.0, _env_float('DRAIN_TIMEOUT_S', 30.0))
# Upper bound on retained session state (history + gate).  See the module
# docstring for the eviction policy (issue #12).
DEFAULT_MAX_SESSIONS = max(1, _env_int('SESSIONS_MAX', 1024))


class WorkerStopping(Exception):
    """Raised when a request is submitted to a worker that is shutting down."""


def _is_multimodal(model_dir: str) -> bool:
    """True when the IR is a vision-language export (needs VLMPipeline).

    LLMPipeline fails on multimodal exports with a tensor-port error
    ("Port for tensor name input_ids was not found"), because those graphs
    route inputs through the embedding/vision interface.  Detection is by
    config: an ``image_token_id`` or a ``ForConditionalGeneration``
    architecture.  Unreadable configs default to the plain LLM path so the
    error surfaces exactly as it did before.
    """
    try:
        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)
    except (OSError, ValueError):
        return False
    if 'image_token_id' in config:
        return True
    architectures = config.get('architectures') or []
    return any(isinstance(a, str) and a.endswith('ForConditionalGeneration')
               for a in architectures)


def _load_pipeline(props: dict, device: str = 'NPU', f32: bool = False):
    """Create the real OpenVINO GenAI pipeline (lazy import of the runtime).

    NPU: NPUW properties (MAX_PROMPT_LEN etc.) apply and the KV window is
    bounded (mpl + mrl). GPU: the plain stateful path — NPUW properties are
    rejected by the plugin — with optional f32 exactness hints.  Multimodal
    exports (Qwen3.6-35B-A3B, Qwen3-VL, ...) are served text-only through
    VLMPipeline: LLMPipeline cannot load their graphs at all.
    """
    import openvino_genai as genai

    ai_root = os.environ.get('FLOSSWARE_AI_ROOT', os.path.expanduser('~/AI'))
    model_dir = os.environ.get('NPU_MODEL_DIR',
                               f'{ai_root}/models/openvino/qwen3-0.6b-int4')
    if device == 'GPU':
        if _is_multimodal(model_dir):
            # f32 exactness hints are untested on the VLM path; leave them off.
            return genai.VLMPipeline(model_dir, device='GPU')
        gpu_props = {}
        if f32:
            # Exactness mode: both hints are required on this build —
            # INFERENCE_PRECISION_HINT alone hits a Paged-Attention
            # block-size error (see docs/correctness.md / roadmap M2).
            gpu_props = {'INFERENCE_PRECISION_HINT': 'f32',
                         'KV_CACHE_PRECISION': 'f32'}
        return genai.LLMPipeline(model_dir, device='GPU', **gpu_props)
    return genai.LLMPipeline(model_dir, device='NPU', **props)


class Worker:
    """Owns one LLMPipeline; executes queued requests sequentially.

    ``pipeline_factory`` and ``token_counter`` are injection points for the
    stub-based test suite; production callers omit them.
    """

    def __init__(self, worker_id: int, max_prompt_len: int,
                 queue_max: int = DEFAULT_MAX_QUEUE,
                 pipeline_factory=None, token_counter=None,
                 device: str = 'NPU', f32: bool = False):
        if queue_max < 1:
            raise ValueError('queue_max must be >= 1')
        self.id = worker_id
        self.device = device
        self.max_prompt_len = max_prompt_len
        # GenAI reads the configured (unaligned) values back from the compiled
        # model, so this is the exact prompt+generated stop boundary (#13).
        # GPU: the plain stateful path has NO KV window — capacity is None and
        # the mpl/capacity request checks are skipped (M3 phase 1).
        self.kv_capacity = (max_prompt_len + MIN_RESPONSE_LEN
                            if device == 'NPU' else None)
        self.queue_max = queue_max
        self.q: 'queue.Queue[dict | object]' = queue.Queue(maxsize=queue_max)
        self.requests_done = 0
        self.cancelled = 0
        self.errors = 0
        self._lock = threading.Lock()
        self._stopping = False
        self._in_flight: dict | None = None

        t0 = time.perf_counter()
        props = {}
        if device == 'NPU':
            props = dict(MAX_PROMPT_LEN=max_prompt_len,
                         MIN_RESPONSE_LEN=MIN_RESPONSE_LEN,
                         GENERATE_HINT='BEST_PERF')
            cache = os.environ.get('NPUW_CACHE_DIR')
            if cache:
                props['CACHE_DIR'] = cache
        if pipeline_factory is not None:
            self.pipe = pipeline_factory(props)
        else:
            self.pipe = _load_pipeline(props, device=device, f32=f32)
        self.load_s = time.perf_counter() - t0
        if token_counter is None:
            self.tokenizer = self.pipe.get_tokenizer()
            self._token_counter = self._count_with_tokenizer
        else:
            self.tokenizer = None
            self._token_counter = token_counter

        self.thread = threading.Thread(target=self._run, name=f'npu-worker-{worker_id}',
                                       daemon=True)
        self.thread.start()

    def _count_with_tokenizer(self, text: str) -> int:
        # ov.Tensor.data is a numpy ndarray (.size = element count); plain
        # sequences (test stubs) fall back to len().  No numpy import, so the
        # stub-based tests and CI need only pytest (issue #14).
        data = self.tokenizer.encode(text).input_ids.data
        size = getattr(data, 'size', None)
        return int(size) if size is not None else len(data)

    def count_tokens(self, text: str) -> int:
        return self._token_counter(text)

    def _run(self) -> None:
        while True:
            try:
                item = self.q.get()
            except Exception:  # noqa: BLE001 - queue.get does not normally raise
                continue
            if item is SENTINEL:
                return
            assert isinstance(item, dict)
            with self._lock:
                self._in_flight = item
            try:
                if item.get('cancelled'):
                    # Client already gave up before the worker picked it up.
                    item['error'] = 'request cancelled after client timeout'
                    item['status'] = 504
                else:
                    t0 = time.perf_counter()
                    out = self.pipe.generate(item['prompt'],
                                             max_new_tokens=item['max_new_tokens'],
                                             do_sample=False)
                    # VLMPipeline (multimodal exports served text-only)
                    # returns VLMDecodedResults, not a plain string.
                    text = out.texts[0] if getattr(out, 'texts', None) else out
                    item['result'] = {
                        'text': text,
                        'gen_s': round(time.perf_counter() - t0, 3),
                        'worker': self.id,
                    }
            except Exception as e:  # noqa: BLE001 - a request must not kill the worker
                self.errors += 1
                item['error'] = f'{type(e).__name__}: {e}'
            finally:
                with self._lock:
                    self._in_flight = None
                self.requests_done += 1
                item['done'].set()

    def submit(self, req: dict) -> None:
        """Enqueue a validated request; raises queue.Full or WorkerStopping."""
        with self._lock:
            if self._stopping:
                raise WorkerStopping()
            self.q.put_nowait(req)

    def stop(self) -> int:
        """Cancel queued-not-started work, then ask the thread to exit.

        Returns the number of requests that were cancelled.  Idempotent and
        safe to call while the worker is executing an in-flight request (that
        request is allowed to finish).  The submit/stop lock guarantees that
        nothing can be enqueued after cancellation has begun.
        """
        with self._lock:
            if self._stopping:
                return 0
            self._stopping = True
            cancelled = 0
            while True:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    break
                if item is SENTINEL:
                    continue
                assert isinstance(item, dict)
                item['error'] = 'server shutting down'
                item['status'] = 503
                item['done'].set()
                cancelled += 1
            self.cancelled += cancelled
            # The queue was just drained, so there is room for the sentinel.
            self.q.put_nowait(SENTINEL)
            return cancelled

    def queue_depth(self) -> int:
        return self.q.qsize()

    def busy(self) -> bool:
        with self._lock:
            return self._in_flight is not None


class SessionGate:
    """FIFO gate used to serialize all operations on one session.

    Callers acquire a ticket in the order they call :meth:`acquire` and are
    released in that same order.  This makes same-session ordering
    deterministic and makes reset ordering deterministic relative to
    generations (see the module docstring).
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._now_serving = 0

    def try_acquire(self, max_pending: int | None = None) -> int | None:
        """Take a ticket without waiting for it to be served.

        Returns ``None`` (without taking a ticket) if ``max_pending``
        operations already hold or wait for this gate.  The caller waits with
        :meth:`wait` and must eventually :meth:`release`.  Splitting ticket
        acquisition from waiting lets ``SessionStore`` take the ticket inside
        its admission critical section, which pins the entry against
        eviction before any other thread can observe it.
        """
        with self._cond:
            if (max_pending is not None
                    and self._next_ticket - self._now_serving >= max_pending):
                return None
            ticket = self._next_ticket
            self._next_ticket += 1
            return ticket

    def wait(self, ticket: int) -> None:
        """Block until ``ticket`` is the one being served."""
        with self._cond:
            while ticket != self._now_serving:
                self._cond.wait()

    def acquire(self, max_pending: int | None = None) -> int | None:
        """Take a ticket and wait for it to be served (convenience form).

        ``SessionStore.acquire`` splits these two steps so the ticket can be
        taken inside its admission critical section.
        """
        ticket = self.try_acquire(max_pending)
        if ticket is None:
            return None
        self.wait(ticket)
        return ticket

    def release(self) -> None:
        with self._cond:
            self._now_serving += 1
            self._cond.notify_all()

    def pending(self) -> int:
        """Number of operations currently holding or waiting for the gate."""
        with self._cond:
            return self._next_ticket - self._now_serving


class HistoryLimitError(ValueError):
    """A session's history plus the new prompt exceeds HISTORY_LIMIT_TOKENS."""


class SessionCapacityError(RuntimeError):
    """A new session cannot be admitted: every stored session is active."""


class SessionEntry:
    """One retained session: its gate, history and idle timestamp.

    ``last_done`` is the monotonic time of the last *completed* operation and
    is the origin for idle-based LRU eviction (creation time until the first
    operation completes).  ``history`` and ``gate`` are removed together when
    the entry is evicted.
    """

    __slots__ = ('session', 'gate', 'history', 'last_done')

    def __init__(self, session: str, now: float) -> None:
        self.session = session
        self.gate = SessionGate()
        self.history: list[str] = []
        self.last_done = now


class SessionStore:
    """Bounded per-session history and gates; requests are self-contained.

    Admission and eviction happen atomically under ``_lock``: :meth:`acquire`
    looks up or creates the entry, evicts the idle LRU entry if a new session
    would exceed ``max_sessions``, and takes the session's gate ticket before
    the lock is released.  Taking the ticket inside the critical section
    means a newly admitted entry is already *active* (pinned) when another
    thread can first see it, so it can never be evicted between creation and
    :meth:`wait`.

    ``finish`` must be called exactly once per successful :meth:`acquire`; it
    records the completion time and releases the ticket.  The internal lock
    is only ever taken before a gate's condition (never the reverse), so
    there is no lock-order inversion.

    ``clock`` is injectable so tests can make idle ordering deterministic.
    """

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS, clock=None) -> None:
        if max_sessions < 1:
            raise ValueError('max_sessions must be >= 1')
        self._lock = threading.Lock()
        self._entries: dict[str, SessionEntry] = {}
        self._max_sessions = max_sessions
        self._evictions = 0
        self._clock = clock or time.monotonic

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def _idle_lru(self) -> SessionEntry | None:
        """Least-recently-completed entry with no ticket held or waiting.

        Caller holds ``_lock``.  ``(last_done, session)`` makes the choice
        deterministic even if the clock has coarse resolution.
        """
        victim = None
        for entry in self._entries.values():
            if entry.gate.pending() != 0:
                continue
            if victim is None or ((entry.last_done, entry.session)
                                  < (victim.last_done, victim.session)):
                victim = entry
        return victim

    def _get_or_create_entry(self, session: str) -> SessionEntry:
        """Look up a session, admitting a new one (evicting if needed).

        Caller holds ``_lock``.  Raises :class:`SessionCapacityError` when a
        new session is requested while every stored session is active.
        """
        entry = self._entries.get(session)
        if entry is not None:
            return entry
        if len(self._entries) >= self._max_sessions:
            victim = self._idle_lru()
            if victim is None:
                raise SessionCapacityError(
                    f'session capacity {self._max_sessions} reached; all '
                    f'{len(self._entries)} stored session(s) are active')
            del self._entries[victim.session]
            self._evictions += 1
        entry = SessionEntry(session, self._clock())
        self._entries[session] = entry
        return entry

    def acquire(self, session: str,
                max_pending: int | None = None) -> tuple[SessionEntry | None, int | None]:
        """Admit ``session`` and take its gate ticket atomically.

        Returns ``(entry, ticket)``; ``(None, None)`` if ``max_pending``
        operations already hold or wait for this session.  Raises
        :class:`SessionCapacityError` if admitting a new session would exceed
        ``max_sessions`` and nothing is evictable.
        """
        with self._lock:
            entry = self._get_or_create_entry(session)
            ticket = entry.gate.try_acquire(max_pending)
            if ticket is None:
                return None, None
            return entry, ticket

    def finish(self, entry: SessionEntry) -> None:
        """Record completion time and release the entry's gate ticket."""
        with self._lock:
            entry.last_done = self._clock()
            entry.gate.release()

    def gate(self, session: str) -> SessionGate:
        """Diagnostic/test hook: the session's gate (admitting it if absent).

        Production request paths must use :meth:`acquire`, which takes a
        ticket in the same critical section as admission.
        """
        with self._lock:
            return self._get_or_create_entry(session).gate

    def build_prompt(self, entry: SessionEntry, prompt: str,
                     counter) -> tuple[str, int]:
        """Return (full prompt, token count) without mutating the history.

        Only the snapshot is taken under ``_lock``; tokenization runs outside
        it so one session's counting does not serialize all sessions.  The
        caller holds the entry's gate ticket, and history is only mutated by
        the ticket holder (``record``/``reset``), so the snapshot cannot go
        stale while it is being counted.
        """
        with self._lock:
            full = ''.join(entry.history) + prompt
        n_tok = counter(full)
        if n_tok > HISTORY_LIMIT_TOKENS:
            raise HistoryLimitError(f'session history ({n_tok} tokens) exceeds '
                                    f'{HISTORY_LIMIT_TOKENS}-token limit; reset the session')
        return full, n_tok

    def record(self, entry: SessionEntry, prompt: str, answer: str) -> None:
        with self._lock:
            entry.history.extend([prompt, answer])

    def reset(self, target: SessionEntry | str) -> None:
        """Clear a session's history (accepts an entry or a session id)."""
        with self._lock:
            entry = (self._entries.get(target) if isinstance(target, str)
                     else target)
            if entry is not None:
                entry.history.clear()

    def history(self, session: str) -> list[str]:
        """Snapshot of a session's history (used by tests and diagnostics)."""
        with self._lock:
            entry = self._entries.get(session)
            return list(entry.history) if entry is not None else []

    def stats(self) -> dict:
        """Capacity diagnostics for /health (issue #12)."""
        with self._lock:
            active = sum(1 for e in self._entries.values()
                         if e.gate.pending() != 0)
            return {
                'sessions': len(self._entries),
                'sessions_active': active,
                'session_capacity': self._max_sessions,
                'session_evictions': self._evictions,
            }


class ServerState:
    """Per-process serving state (one instance per server, easy to test)."""

    def __init__(self, workers: list[Worker], sessions: SessionStore,
                 drain_timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
                 max_session_pending: int = MAX_SESSION_PENDING):
        self.workers = workers
        self.sessions = sessions
        self.drain_timeout_s = drain_timeout_s
        self.max_session_pending = max_session_pending
        # HTTP handlers currently executing (so shutdown can wait for the
        # responses of drained/cancelled work to be written before the
        # process exits; handler threads are daemon threads).
        self._active_handlers = 0
        self._handlers_cond = threading.Condition()
        self.httpd: ThreadingHTTPServer | None = None
        self.shutdown = False
        self.shutdown_report: dict | None = None
        self.shutdown_finished = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._rr = 0
        self._rr_lock = threading.Lock()

    def submit(self, req: dict) -> Worker:
        """Round-robin submit, falling over to any worker with queue room.

        Raises ``queue.Full`` when every worker queue is full and
        ``WorkerStopping`` when the workers have begun shutting down.
        """
        n = len(self.workers)
        with self._rr_lock:
            start = self._rr
            self._rr += 1
        for i in range(n):
            worker = self.workers[(start + i) % n]
            try:
                worker.submit(req)
                return worker
            except WorkerStopping:
                raise
            except queue.Full:
                continue
        raise queue.Full(f'all {n} worker queue(s) are full')

    def begin_shutdown(self) -> bool:
        """Mark shutdown; returns True only for the caller that started it."""
        with self._shutdown_lock:
            if self.shutdown:
                return False
            self.shutdown = True
            return True

    def queue_capacity(self) -> int:
        return sum(w.queue_max for w in self.workers)

    def handler_enter(self) -> None:
        with self._handlers_cond:
            self._active_handlers += 1

    def handler_exit(self) -> None:
        with self._handlers_cond:
            self._active_handlers -= 1
            self._handlers_cond.notify_all()

    def wait_handlers_idle(self, timeout: float) -> int:
        """Wait up to ``timeout`` s for running handlers; return how many remain."""
        deadline = time.monotonic() + timeout
        with self._handlers_cond:
            while self._active_handlers > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._handlers_cond.wait(remaining)
            return self._active_handlers

    def queue_depth(self) -> int:
        return sum(w.queue_depth() for w in self.workers)


def _validate_session(payload: dict) -> tuple[str | None, str | None]:
    session = payload.get('session')
    if not isinstance(session, str) or not session.strip():
        return None, 'session must be a non-empty string'
    if len(session) > MAX_SESSION_ID_LEN:
        return None, f'session must be at most {MAX_SESSION_ID_LEN} characters'
    if not session.isprintable():
        return None, 'session must contain printable characters only'
    return session, None


def _validate_prompt(payload: dict) -> tuple[str | None, str | None]:
    prompt = payload.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        return None, 'prompt must be a non-empty string'
    return prompt, None


def _validate_max_new_tokens(payload: dict) -> tuple[int | None, str | None]:
    value = payload.get('max_new_tokens', 32)
    if isinstance(value, bool) or not isinstance(value, int):
        return None, 'max_new_tokens must be an integer'
    if not (MAX_NEW_TOKENS_MIN <= value <= MAX_NEW_TOKENS_MAX):
        return None, (f'max_new_tokens out of range '
                      f'[{MAX_NEW_TOKENS_MIN}, {MAX_NEW_TOKENS_MAX}]')
    return value, None


def make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            try:
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                # Client went away; nothing useful left to do.
                self.close_connection = True

        def _read_json(self) -> tuple[dict | None, str | None]:
            raw_len = self.headers.get('Content-Length')
            try:
                length = int(raw_len) if raw_len is not None else 0
            except (TypeError, ValueError):
                self.close_connection = True
                return None, 'bad_request: invalid Content-Length'
            if length < 0 or length > MAX_BODY_BYTES:
                self.close_connection = True
                return None, f'bad_request: Content-Length must be 0..{MAX_BODY_BYTES}'
            try:
                raw = self.rfile.read(length)
            except OSError as e:
                self.close_connection = True
                return None, f'bad_request: {e}'
            try:
                payload = json.loads(raw or b'{}')
            except (ValueError, json.JSONDecodeError) as e:
                return None, f'bad_request: {e}'
            if not isinstance(payload, dict):
                return None, 'bad_request: JSON body must be an object'
            return payload, None

        def do_GET(self):
            state.handler_enter()
            try:
                self._do_get()
            finally:
                state.handler_exit()

        def do_POST(self):
            state.handler_enter()
            try:
                self._do_post()
            finally:
                state.handler_exit()

        def _session_overloaded(self) -> None:
            self._json(429, {
                'error': f'overloaded: session already has '
                         f'{state.max_session_pending} pending operation(s)',
            })

        def _do_get(self):
            if self.path == '/health':
                self._json(200, {
                    'ok': True,
                    'shutting_down': state.shutdown,
                    'workers': len(state.workers),
                    'requests_done': [w.requests_done for w in state.workers],
                    'errors': [w.errors for w in state.workers],
                    'cancelled': [w.cancelled for w in state.workers],
                    'queue_depth': [w.queue_depth() for w in state.workers],
                    'queue_capacity': [w.queue_max for w in state.workers],
                    'queue_depth_total': state.queue_depth(),
                    'queue_capacity_total': state.queue_capacity(),
                    'busy': [w.busy() for w in state.workers],
                    'kv_capacity': state.workers[0].kv_capacity,
                    **state.sessions.stats(),
                })
            else:
                self._json(404, {'error': 'not_found'})

        def _do_post(self):
            path = self.path

            # Shutdown is idempotent and must be answerable while shutting down.
            if path == '/shutdown':
                # The body (if any) is not read: close the connection so
                # unread bytes cannot be parsed as a following request.
                self.close_connection = True
                started = request_shutdown(state)
                self._json(200, {'ok': True,
                                 'detail': 'shutting down' if started
                                           else 'already shutting down'})
                return

            if state.shutdown:
                self.close_connection = True  # body not read; see above
                self._json(503, {'error': 'shutting_down'})
                return

            payload, err = self._read_json()
            if err is not None:
                self._json(400, {'error': err})
                return

            if path == '/reset':
                self._handle_reset(payload)
                return
            if path == '/generate':
                self._handle_generate(payload)
                return
            if path == '/v1/messages':
                self._handle_messages(payload)
                return
            self._json(404, {'error': 'not_found'})

        # ---- Anthropic Messages-API shim (v1: text-only) ----

        def _handle_messages(self, payload: dict) -> None:
            """POST /v1/messages — Anthropic Messages API shape (text-only).

            Lets local models act as ensemble workers/arbiters for simple
            clients (scripts, curl, custom worker harnesses). One server
            instance serves one model; the request's ``model`` field is
            accepted as-is. Tool use is NOT supported: requests carrying
            ``tools`` are rejected with a clear error rather than silently
            ignored. Each call is self-contained (no session history).
            """
            import json as _json

            if payload.get('tools'):
                self._json(400, {
                    'type': 'error',
                    'error': {'type': 'invalid_request_error',
                              'message': 'tool use is not supported by the '
                                         'local serving shim (text-only v1)'}})
                return
            msgs = payload.get('messages')
            if not isinstance(msgs, list) or not msgs:
                self._json(400, {
                    'type': 'error',
                    'error': {'type': 'invalid_request_error',
                              'message': 'messages must be a non-empty list'}})
                return
            try:
                max_tokens = int(payload.get('max_tokens', 1024))
            except (TypeError, ValueError):
                self._json(400, {
                    'type': 'error',
                    'error': {'type': 'invalid_request_error',
                              'message': 'max_tokens must be an integer'}})
                return
            if not (1 <= max_tokens <= 4096):
                self._json(400, {
                    'type': 'error',
                    'error': {'type': 'invalid_request_error',
                              'message': 'max_tokens out of range [1, 4096]'}})
                return

            def _content_text(content):
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [b.get('text', '') for b in content
                             if isinstance(b, dict) and b.get('type') == 'text']
                    return '\n'.join(p for p in parts if p)
                return ''

            system = payload.get('system') or ''
            if not isinstance(system, str):
                system = _content_text(system)
            prompt_parts = [system] if system else []
            for m in msgs:
                role = m.get('role', 'user')
                text = _content_text(m.get('content', ''))
                if not text:
                    continue
                tag = 'Human' if role == 'user' else 'Assistant'
                prompt_parts.append(f'{tag}: {text}')
            prompt_parts.append('Assistant:')
            full_prompt = '\n\n'.join(prompt_parts)

            n_tok = state.workers[0].count_tokens(full_prompt)
            if n_tok > HISTORY_LIMIT_TOKENS:
                self._json(400, {
                    'type': 'error',
                    'error': {'type': 'invalid_request_error',
                              'message': f'prompt_too_long: {n_tok} tokens > '
                                         f'{HISTORY_LIMIT_TOKENS}'}})
                return

            stream = bool(payload.get('stream'))
            req = {'prompt': full_prompt, 'max_new_tokens': max_tokens,
                   'done': threading.Event(), 'stream': stream,
                   'chunks': []}
            try:
                state.submit(req)
            except queue.Full:
                self._json(429, {
                    'type': 'error',
                    'error': {'type': 'overloaded_error',
                              'message': 'all worker queues are full'}})
                return
            if not req['done'].wait(timeout=600):
                self._json(504, {
                    'type': 'error',
                    'error': {'type': 'api_error',
                              'message': 'generation timed out'}})
                return
            if 'error' in req:
                self._json(500, {
                    'type': 'error',
                    'error': {'type': 'api_error',
                              'message': req['error']}})
                return

            text = req['result']['text']
            msg_id = f'msg_local_{req["result"].get("worker", 0)}_{int(time.time() * 1000)}'
            out_tokens = req['result'].get('generated_tokens', len(text) // 4)
            stop = ('max_tokens' if req['result'].get('generated')
                    == max_tokens else 'end_turn')
            if stream:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()

                def sse(event, data):
                    self.wfile.write(
                        f'event: {event}\ndata: {_json.dumps(data)}\n\n'
                        .encode())

                sse('message_start', {
                    'type': 'message_start',
                    'message': {'id': msg_id, 'type': 'message',
                                'role': 'assistant',
                                'model': payload.get('model', 'local'),
                                'content': [], 'stop_reason': None,
                                'usage': {'input_tokens': n_tok,
                                          'output_tokens': 0}}})
                sse('content_block_start', {
                    'type': 'content_block_start', 'index': 0,
                    'content_block': {'type': 'text', 'text': ''}})
                # coarse deltas: whole text in one delta (v1)
                sse('content_block_delta', {
                    'type': 'content_block_delta', 'index': 0,
                    'delta': {'type': 'text_delta', 'text': text}})
                sse('content_block_stop', {
                    'type': 'content_block_stop', 'index': 0})
                sse('message_delta', {
                    'type': 'message_delta',
                    'delta': {'stop_reason': stop, 'stop_sequence': None},
                    'usage': {'output_tokens': out_tokens}})
                sse('message_stop', {'type': 'message_stop'})
            else:
                self._json(200, {
                    'id': msg_id, 'type': 'message', 'role': 'assistant',
                    'model': payload.get('model', 'local'),
                    'content': [{'type': 'text', 'text': text}],
                    'stop_reason': stop,
                    'stop_sequence': None,
                    'usage': {'input_tokens': n_tok,
                              'output_tokens': out_tokens}})

        def _handle_reset(self, payload: dict) -> None:
            session, err = _validate_session(payload)
            if err is not None:
                self._json(400, {'error': err})
                return
            try:
                entry, ticket = state.sessions.acquire(
                    session, state.max_session_pending)
            except SessionCapacityError as e:
                self._json(429, {'error': f'overloaded: {e}'})
                return
            # Waits for earlier same-session operations (bounded, see
            # MAX_SESSION_PENDING).  The ticket was taken atomically with
            # admission, so the entry cannot be evicted while we wait.
            if entry is None:
                self._session_overloaded()
                return
            entry.gate.wait(ticket)
            try:
                if state.shutdown:
                    self._json(503, {'error': 'shutting_down'})
                    return
                state.sessions.reset(entry)
            finally:
                state.sessions.finish(entry)
            self._json(200, {'ok': True})

        def _handle_generate(self, payload: dict) -> None:
            session, err = _validate_session(payload)
            if err is not None:
                self._json(400, {'error': err})
                return
            prompt, err = _validate_prompt(payload)
            if err is not None:
                self._json(400, {'error': err})
                return
            max_new, err = _validate_max_new_tokens(payload)
            if err is not None:
                self._json(400, {'error': err})
                return

            try:
                entry, ticket = state.sessions.acquire(
                    session, state.max_session_pending)
            except SessionCapacityError as e:
                self._json(429, {'error': f'overloaded: {e}'})
                return
            # FIFO: same-session operations are ordered (bounded wait list).
            # The ticket was taken atomically with admission, so the entry
            # stays pinned (unevictable) from creation through completion.
            if entry is None:
                self._session_overloaded()
                return
            entry.gate.wait(ticket)
            try:
                if state.shutdown:
                    self._json(503, {'error': 'shutting_down'})
                    return

                counter = state.workers[0].count_tokens
                try:
                    full_prompt, n_tok = state.sessions.build_prompt(
                        entry, prompt, counter)
                except HistoryLimitError as e:
                    self._json(400, {'error': str(e)})
                    return
                except Exception as e:  # noqa: BLE001 - tokenizer failure
                    # Server-side failure (e.g. the tokenizer raised): answer
                    # with a JSON 500 instead of dropping the connection.
                    self._json(500, {'error': f'tokenizer_error: '
                                              f'{type(e).__name__}: {e}'})
                    return

                if (state.workers[0].device == 'NPU'
                        and n_tok > state.workers[0].max_prompt_len):
                    self._json(400, {
                        'error': f'prompt_too_long: {n_tok} tokens > MAX_PROMPT_LEN '
                                 f'{state.workers[0].max_prompt_len}; '
                                 f'reset session or shorten',
                    })
                    return

                capacity = state.workers[0].kv_capacity
                if capacity is not None and n_tok + max_new > capacity:
                    # NPUW would stop at capacity - n_tok tokens without any
                    # error (issue #13); refuse instead of truncating silently.
                    self._json(400, {
                        'error': f'exceeds_kv_capacity: {n_tok} prompt tokens + '
                                 f'max_new_tokens {max_new} > MAX_PROMPT_LEN + '
                                 f'MIN_RESPONSE_LEN = {capacity}; the response '
                                 f'would be truncated to at most '
                                 f'{capacity - n_tok} tokens. Lower '
                                 f'max_new_tokens, shorten/reset the session, or '
                                 f'raise MIN_RESPONSE_LEN',
                    })
                    return

                req = {'prompt': full_prompt, 'max_new_tokens': max_new,
                       'done': threading.Event()}
                try:
                    state.submit(req)
                except WorkerStopping:
                    self._json(503, {'error': 'shutting_down'})
                    return
                except queue.Full:
                    self._json(429, {
                        'error': f'overloaded: all {len(state.workers)} worker '
                                 f'queue(s) full (capacity '
                                 f'{state.queue_capacity()})',
                    })
                    return

                if not req['done'].wait(timeout=GENERATE_TIMEOUT_S):
                    # Ask the worker to skip it if it has not started.  If it
                    # already started, its result is abandoned: it is never
                    # recorded in the session history, so releasing the
                    # session gate now cannot reorder history.  The 504 is
                    # delivered after GENERATE_TIMEOUT_S, not after the
                    # (possibly wedged) generation returns.
                    req['cancelled'] = True
                    self._json(504, {'error': 'timeout'})
                    return
                if 'error' in req:
                    self._json(req.get('status', 500), {'error': req['error']})
                    return
                state.sessions.record(entry, prompt, req['result']['text'])
                self._json(200, {'session': session, 'prompt_tokens': n_tok,
                                 **req['result']})
            finally:
                # Records completion time and releases the ticket; after this
                # the entry may be evicted by a later admission.
                state.sessions.finish(entry)

    return Handler


def request_shutdown(state: ServerState) -> bool:
    """Start the orderly shutdown in a helper thread; True if newly started.

    Safe to call from HTTP handlers and from signal handlers (both must not
    call ``httpd.shutdown()`` on the serving thread).
    """
    if not state.begin_shutdown():
        return False
    threading.Thread(target=graceful_shutdown, args=(state, True),
                     name='serving-shutdown', daemon=True).start()
    return True


def graceful_shutdown(state: ServerState, claimed: bool = False) -> dict | None:
    """Run the documented, idempotent, orderly shutdown sequence.

    Queued-not-started requests are cancelled with HTTP 503; in-flight
    requests are allowed to finish; workers are joined within
    ``state.drain_timeout_s`` and any that fail to stop are logged/reported.
    The listening socket is closed and the call returns normally — there is
    no ``os._exit()``.

    ``claimed=True`` means the caller already won :meth:`ServerState.
    begin_shutdown` (used by :func:`request_shutdown`, which must claim before
    spawning the thread); every other caller competes for the claim itself.
    """
    if not claimed and not state.begin_shutdown():
        # Another caller owns the sequence: wait for it, then report.  Every
        # step of the owner's sequence is individually bounded, so this wait
        # is bounded too; returning early would let main() exit while the
        # owner is still draining.
        state.shutdown_finished.wait()
        return state.shutdown_report

    report: dict = {
        'cancelled_requests': 0,
        'workers_stopped': [],
        'unstopped_workers': [],
        'timed_out': False,
        'drain_timeout_s': state.drain_timeout_s,
        'httpd_shutdown_timed_out': False,
        'unfinished_handlers': 0,
    }
    try:
        httpd = state.httpd
        if httpd is not None:
            # shutdown() blocks until serve_forever() returns.  Run it on a
            # helper thread so an unexpected state can never wedge shutdown
            # silently; the wait is bounded and reported.
            stopper = threading.Thread(target=httpd.shutdown,
                                       name='httpd-shutdown', daemon=True)
            stopper.start()
            stopper.join(timeout=max(5.0, state.drain_timeout_s))
            if stopper.is_alive():
                report['httpd_shutdown_timed_out'] = True

        # Stop dispatching: cancel queued-not-started work (HTTP 503).
        for worker in state.workers:
            report['cancelled_requests'] += worker.stop()

        # Let in-flight work finish, within the configured drain timeout.
        deadline = time.monotonic() + state.drain_timeout_s
        for worker in state.workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.thread.join(timeout=remaining)
            if worker.thread.is_alive():
                report['unstopped_workers'].append(worker.id)
            else:
                report['workers_stopped'].append(worker.id)
        report['timed_out'] = bool(report['unstopped_workers'])

        # Let handler threads deliver the responses of finished/cancelled
        # work (they are daemon threads and would otherwise be killed when
        # main() returns).  A POST /shutdown handler is counted too; it
        # answers immediately after starting this sequence, so it does not
        # delay the wait meaningfully.
        grace = max(0.0, deadline - time.monotonic()) + HANDLER_GRACE_S
        report['unfinished_handlers'] = state.wait_handlers_idle(grace)

        if httpd is not None:
            httpd.server_close()
    except Exception as e:  # noqa: BLE001 - shutdown must always complete
        report['error'] = f'{type(e).__name__}: {e}'
    finally:
        state.shutdown_report = report
        state.shutdown_finished.set()

    if report['timed_out']:
        print(f'[shutdown] WARNING: worker(s) {report["unstopped_workers"]} did '
              f'not stop within {state.drain_timeout_s}s; their in-flight requests '
              f'are abandoned (daemon threads stop at interpreter exit; no '
              f'forced os._exit is used)',
              file=sys.stderr, flush=True)
    if report['unfinished_handlers']:
        print(f'[shutdown] WARNING: {report["unfinished_handlers"]} HTTP '
              f'handler(s) still running; their responses may be lost',
              file=sys.stderr, flush=True)
    if report['httpd_shutdown_timed_out']:
        print('[shutdown] WARNING: httpd.shutdown() did not return in time',
              file=sys.stderr, flush=True)
    print(f'[shutdown] stopped workers {report["workers_stopped"]}; '
          f'cancelled {report["cancelled_requests"]} queued request(s)',
          flush=True)
    return report


def main() -> None:
    global HISTORY_LIMIT_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8931)
    ap.add_argument('--workers', type=int, default=1, choices=(1, 2))
    ap.add_argument('--max-prompt-len', type=int, default=MAX_PROMPT_LEN)
    ap.add_argument('--device', choices=('NPU', 'GPU'), default='NPU',
                    help='accelerator for the pipeline. GPU uses the plain '
                         'stateful path (no NPUW properties, no KV window; '
                         'see docs/serving.md). Default NPU.')
    ap.add_argument('--f32', action='store_true',
                    help='GPU only: force f32 inference + KV-cache precision '
                         '(CPU-identical output; slower than the default f16)')
    ap.add_argument('--history-limit', type=int,
                    default=int(os.environ.get('HISTORY_LIMIT_TOKENS',
                                               HISTORY_LIMIT_TOKENS)),
                    help='maximum session history + prompt tokens before the '
                         'session is rejected (must exceed --max-prompt-len '
                         'plus the pipeline KV capacity margin; default '
                         f'{HISTORY_LIMIT_TOKENS}; env HISTORY_LIMIT_TOKENS). '
                         'Raise to the bucket + MIN_RESPONSE_LEN when serving '
                         'MAX_PROMPT_LEN >= 1024 (roadmap M1).')
    ap.add_argument('--max-queue', type=int, default=DEFAULT_MAX_QUEUE,
                    help='bounded per-worker request queue capacity '
                         f'(default {DEFAULT_MAX_QUEUE}; env QUEUE_MAX)')
    ap.add_argument('--max-sessions', type=int, default=DEFAULT_MAX_SESSIONS,
                    help='maximum retained sessions (history + gate); idle LRU '
                         'entries are evicted on admission '
                         f'(default {DEFAULT_MAX_SESSIONS}; env SESSIONS_MAX)')
    ap.add_argument('--drain-timeout', type=float, default=DEFAULT_DRAIN_TIMEOUT_S,
                    help='seconds to wait for in-flight work during shutdown '
                         f'(default {DEFAULT_DRAIN_TIMEOUT_S}; env DRAIN_TIMEOUT_S)')
    args = ap.parse_args()
    if args.max_queue < 1:
        ap.error('--max-queue must be >= 1')
    if args.max_sessions < 1:
        ap.error('--max-sessions must be >= 1')
    if args.drain_timeout < 0:
        ap.error('--drain-timeout must be >= 0')
    if args.history_limit <= args.max_prompt_len:
        ap.error('--history-limit must exceed --max-prompt-len '
                 '(history must be able to fill the prompt bucket)')

    HISTORY_LIMIT_TOKENS = args.history_limit

    sessions = SessionStore(max_sessions=args.max_sessions)
    workers: list[Worker] = []
    for i in range(args.workers):
        t0 = time.perf_counter()
        workers.append(Worker(i, args.max_prompt_len, queue_max=args.max_queue,
                              device=args.device, f32=args.f32))
        print(f'[worker {i}] pipeline ready in {time.perf_counter()-t0:.1f}s',
              flush=True)

    state = ServerState(workers, sessions, drain_timeout_s=args.drain_timeout)
    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(state))
    state.httpd = httpd

    def _on_signal(signum, frame):  # noqa: ARG001
        request_shutdown(state)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    print(f'[server] listening on 127.0.0.1:{args.port} '
          f'({args.workers} worker(s), mpl={args.max_prompt_len}, '
          f'queue={args.max_queue}, max_sessions={args.max_sessions})', flush=True)
    try:
        httpd.serve_forever()
    finally:
        graceful_shutdown(state)


if __name__ == '__main__':
    main()
