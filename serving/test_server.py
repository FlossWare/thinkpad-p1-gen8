#!/usr/bin/env python3
"""Behavioural tests for serving/server.py using a stub pipeline.

Pure standard library; no NPU, model or OpenVINO runtime is required.  The
real ``genai.LLMPipeline`` is replaced by :class:`StubPipeline`, injected
through :class:`server.Worker`'s ``pipeline_factory``/``token_counter`` hooks.
The HTTP server is started in-process on an ephemeral port.

Covered behaviour (issues #3, #4, #7, #12, #13):
  * same-session requests serialize in arrival order and history is atomic;
  * different sessions execute concurrently (two workers);
  * reset waits for earlier same-session work, then clears history;
  * bounded queue saturation returns 429 without queueing or mutating state;
  * malformed / missing / out-of-range fields return 400 before queueing;
  * prompt + max_new_tokens beyond the NPUW KV window returns 400;
  * session state is bounded by max_sessions with idle LRU eviction, active
    sessions are pinned, concurrent admission never duplicates gates or
    exceeds the limit, and /health reports capacity/eviction counters;
  * orderly shutdown cancels queued work (503), finishes in-flight work,
    reports drain timeouts, is idempotent and never uses os._exit();
  * pipeline exceptions surface as 500 and do not kill the worker.

Run:  python3 serving/test_server.py -v
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('SERVING_ALLOW_IMPORT', '1')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import server  # noqa: E402


def wait_until(predicate, timeout: float = 5.0, message: str = 'condition'):
    """Poll ``predicate`` until true or timeout (no fixed sleeps)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f'timed out waiting for {message}')


class StubPipeline:
    """Deterministic stand-in for ``genai.LLMPipeline``.

    ``block(prompt, call_index)`` is invoked before answering and may wait on
    events/barriers (or raise) to exercise concurrency paths.  Answers and
    call order are recorded for assertions.
    """

    def __init__(self, block=None, delay: float = 0.0):
        self.block = block
        self.delay = delay
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.answers: list[str] = []
        self.active = 0
        self.max_active = 0

    def generate(self, prompt, max_new_tokens=1, do_sample=False):  # noqa: ARG002
        with self._lock:
            self.calls.append(prompt)
            index = len(self.calls)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.block is not None:
                self.block(prompt, index)
            if self.delay:
                time.sleep(self.delay)
            answer = f' reply{index}'
            with self._lock:
                self.answers.append(answer)
            return answer
        finally:
            with self._lock:
                self.active -= 1

    def snapshot_calls(self) -> list[str]:
        with self._lock:
            return list(self.calls)


class Harness:
    """Runs serving/server.py's real HTTP front end + workers in-process."""

    def __init__(self, pipelines, max_queue: int = 8,
                 drain_timeout_s: float = 5.0, max_prompt_len: int = 64,
                 token_counter=None, max_sessions: int = 1024,
                 session_clock=None):
        self.sessions = server.SessionStore(
            max_sessions=max_sessions, clock=session_clock)
        self.workers = [
            server.Worker(
                i, max_prompt_len, queue_max=max_queue,
                pipeline_factory=lambda props, p=p: p,  # noqa: ARG005
                token_counter=token_counter or (lambda text: len(text)),
            )
            for i, p in enumerate(pipelines)
        ]
        self.state = server.ServerState(self.workers, self.sessions,
                                        drain_timeout_s=drain_timeout_s)
        self.httpd = server.ThreadingHTTPServer(
            ('127.0.0.1', 0), server.make_handler(self.state))
        self.state.httpd = self.httpd
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever,
                                        kwargs={'poll_interval': 0.01},
                                        daemon=True)
        self._thread.start()

    # -- HTTP helpers -----------------------------------------------------
    def _request(self, method: str, path: str, data: bytes | None,
                 timeout: float = 10.0):
        req = urllib.request.Request(
            f'http://127.0.0.1:{self.port}{path}', data=data, method=method,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b'{}')
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
                try:
                    payload = json.loads(raw or b'{}')
                except json.JSONDecodeError:
                    payload = {'raw': raw.decode(errors='replace')}
                return e.code, payload
            finally:
                e.close()

    def get(self, path: str, timeout: float = 10.0):
        return self._request('GET', path, None, timeout)

    def post(self, path: str, obj: dict | None = None, timeout: float = 10.0):
        return self._request('POST', path, json.dumps(obj or {}).encode(), timeout)

    def post_raw(self, path: str, raw: bytes, timeout: float = 10.0):
        return self._request('POST', path, raw, timeout)

    def close(self):
        server.graceful_shutdown(self.state)
        self._thread.join(timeout=10)
        for worker in self.workers:
            worker.thread.join(timeout=10)


class SameSessionOrderingTest(unittest.TestCase):
    def test_same_session_serializes_in_arrival_order(self):
        first_started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                first_started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=4)
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(h.post, '/generate',
                               {'session': 's', 'prompt': 'A', 'max_new_tokens': 4})
                self.assertTrue(first_started.wait(5), 'first generation did not start')
                f2 = ex.submit(h.post, '/generate',
                               {'session': 's', 'prompt': 'B', 'max_new_tokens': 4})

                # The second request must be blocked on the session gate, not
                # in the worker queue, and must not have reached the pipeline.
                wait_until(lambda: h.state.sessions.gate('s').pending() >= 2,
                           message='second request waiting on the session gate')
                self.assertFalse(f2.done(), 'second request should not have completed')
                self.assertEqual(h.get('/health')[1]['queue_depth_total'], 0)
                self.assertEqual(pipe.active, 1)

                release.set()
                c1, b1 = f1.result(timeout=10)
                c2, b2 = f2.result(timeout=10)

            self.assertEqual((c1, c2), (200, 200))
            self.assertEqual(pipe.max_active, 1, 'same-session work overlapped')
            # The second prompt must contain the first exchange: history read
            # and recorded atomically with respect to the first request.
            self.assertEqual(pipe.snapshot_calls(),
                             ['A', 'A' + b1['text'] + 'B'])
            self.assertEqual(h.state.sessions.history('s'),
                             ['A', b1['text'], 'B', b2['text']])
        finally:
            release.set()
            h.close()


class IndependentSessionsTest(unittest.TestCase):
    def test_different_sessions_proceed_concurrently(self):
        # Both workers must be inside generate() at the same time or the
        # barrier times out -> BrokenBarrierError -> HTTP 500.
        barrier = threading.Barrier(2)

        def block(prompt, index):  # noqa: ARG001
            barrier.wait(timeout=5)

        pipes = [StubPipeline(block=block), StubPipeline(block=block)]
        h = Harness(pipes, max_queue=4)
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(h.post, '/generate', {'session': 's1', 'prompt': 'one'})
                f2 = ex.submit(h.post, '/generate', {'session': 's2', 'prompt': 'two'})
                c1, b1 = f1.result(timeout=10)
                c2, b2 = f2.result(timeout=10)

            self.assertEqual((c1, c2), (200, 200))
            self.assertEqual(sorted([w.requests_done for w in h.workers]), [1, 1])
            self.assertEqual(h.state.sessions.history('s1'), ['one', b1['text']])
            self.assertEqual(h.state.sessions.history('s2'), ['two', b2['text']])
        finally:
            h.close()


class ResetSemanticsTest(unittest.TestCase):
    def test_reset_waits_for_in_flight_generation(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=4)
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fg = ex.submit(h.post, '/generate', {'session': 's', 'prompt': 'A'})
                self.assertTrue(started.wait(5))
                fr = ex.submit(h.post, '/reset', {'session': 's'})
                wait_until(lambda: h.state.sessions.gate('s').pending() >= 2,
                           message='reset waiting on the session gate')
                self.assertFalse(fr.done(), 'reset must wait for the in-flight generation')
                self.assertEqual(h.state.sessions.history('s'), [])

                release.set()
                self.assertEqual(fg.result(timeout=10)[0], 200)
                self.assertEqual(fr.result(timeout=10)[0], 200)

            self.assertEqual(h.state.sessions.history('s'), [])
            # A later generation must see the cleared history.
            self.assertEqual(h.post('/generate',
                                    {'session': 's', 'prompt': 'C'})[0], 200)
            self.assertEqual(pipe.snapshot_calls()[-1], 'C')
        finally:
            release.set()
            h.close()

    def test_reset_is_ordered_after_earlier_same_session_requests(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=4)
        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                fa = ex.submit(h.post, '/generate', {'session': 's', 'prompt': 'A'})
                self.assertTrue(started.wait(5))
                fb = ex.submit(h.post, '/generate', {'session': 's', 'prompt': 'B'})
                wait_until(lambda: h.state.sessions.gate('s').pending() >= 2,
                           message='second generation waiting')
                fr = ex.submit(h.post, '/reset', {'session': 's'})
                wait_until(lambda: h.state.sessions.gate('s').pending() >= 3,
                           message='reset waiting behind both generations')
                self.assertFalse(fr.done())

                release.set()
                self.assertEqual(fa.result(timeout=10)[0], 200)
                self.assertEqual(fb.result(timeout=10)[0], 200)
                self.assertEqual(fr.result(timeout=10)[0], 200)

            self.assertEqual(len(pipe.snapshot_calls()), 2)
            self.assertEqual(h.state.sessions.history('s'), [])
        finally:
            release.set()
            h.close()


class BoundedQueueTest(unittest.TestCase):
    def test_saturation_returns_429_without_queueing(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=2)
        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(h.post, '/generate', {'session': 's0', 'prompt': 'zero'})
                self.assertTrue(started.wait(5), 'in-flight request did not start')
                f1 = ex.submit(h.post, '/generate', {'session': 's1', 'prompt': 'one'})
                f2 = ex.submit(h.post, '/generate', {'session': 's2', 'prompt': 'two'})
                wait_until(lambda: h.get('/health')[1]['queue_depth_total'] == 2,
                           message='queue to fill to capacity')

                code, body = h.post('/generate', {'session': 's3', 'prompt': 'three'})
                self.assertEqual(code, 429)
                self.assertIn('overloaded', body['error'])

                health = h.get('/health')[1]
                self.assertEqual(health['queue_capacity'], [2])
                self.assertEqual(health['queue_capacity_total'], 2)
                self.assertEqual(health['queue_depth_total'], 2,
                                 'rejected request must not consume queue capacity')
                self.assertEqual(health['requests_done'], [0])
                self.assertEqual(h.state.sessions.history('s3'), [],
                                 'rejected request must not mutate session history')

                release.set()
                codes = [f.result(timeout=10)[0] for f in (f0, f1, f2)]
            self.assertEqual(codes, [200, 200, 200])
            self.assertEqual(h.state.sessions.history('s3'), [])
            self.assertEqual(len(h.state.sessions.history('s1')), 2)
            self.assertEqual(len(h.state.sessions.history('s2')), 2)
            # 'zero' started first; the two queued sessions (distinct, so
            # independent) may reach the queue in either order.
            calls = pipe.snapshot_calls()
            self.assertEqual(calls[0], 'zero')
            self.assertEqual(sorted(calls[1:]), ['one', 'two'])
        finally:
            release.set()
            h.close()


class SessionPendingCapTest(unittest.TestCase):
    def test_same_session_flood_is_bounded_with_429(self):
        # Gate waiters are not in a worker queue; the per-session cap keeps
        # them bounded (arbiter fix for issue #7).
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=4)
        h.state.max_session_pending = 3
        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                f1 = ex.submit(h.post, '/generate', {'session': 's', 'prompt': 'A'})
                self.assertTrue(started.wait(5))
                f2 = ex.submit(h.post, '/generate', {'session': 's', 'prompt': 'B'})
                wait_until(lambda: h.state.sessions.gate('s').pending() == 2,
                           message='B waiting on the session gate')
                f3 = ex.submit(h.post, '/reset', {'session': 's'})
                wait_until(lambda: h.state.sessions.gate('s').pending() == 3,
                           message='three operations on the session gate')

                code, body = h.post('/generate', {'session': 's', 'prompt': 'C'})
                self.assertEqual(code, 429)
                self.assertIn('overloaded', body['error'])
                code, body = h.post('/reset', {'session': 's'})
                self.assertEqual(code, 429)
                self.assertEqual(h.state.sessions.gate('s').pending(), 3,
                                 'rejected operations must not take a ticket')
                # Other sessions are unaffected by one session's cap.
                fo = ex.submit(h.post, '/generate', {'session': 'other', 'prompt': 'O'})

                release.set()
                self.assertEqual([f.result(timeout=10)[0] for f in (f1, f2, f3, fo)],
                                 [200, 200, 200, 200])
            self.assertNotIn('C', ''.join(pipe.snapshot_calls()))
            self.assertEqual(h.state.sessions.history('s'), [])  # reset ran last
        finally:
            release.set()
            h.close()


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.pipe = StubPipeline()
        self.h = Harness([self.pipe], max_queue=4)

    def tearDown(self):
        self.h.close()

    def _assert_rejected(self, path: str, payload: dict) -> dict:
        code, body = self.h.post(path, payload)
        self.assertEqual(code, 400, f'{payload!r} -> {code} {body!r}')
        self.assertIn('error', body)
        health = self.h.get('/health')[1]
        self.assertEqual(health['queue_depth_total'], 0)
        self.assertEqual(sum(health['requests_done']), 0)
        return body

    def test_generate_rejects_invalid_fields(self):
        cases = [
            {'prompt': 'x'},                                        # missing session
            {'session': '', 'prompt': 'x'},                         # empty session
            {'session': '   ', 'prompt': 'x'},                      # blank session
            {'session': 123, 'prompt': 'x'},                        # non-string session
            {'session': 's' * (server.MAX_SESSION_ID_LEN + 1), 'prompt': 'x'},
            {'session': 'bad\nsession', 'prompt': 'x'},             # non-printable
            {'session': 's'},                                       # missing prompt
            {'session': 's', 'prompt': ''},                         # empty prompt
            {'session': 's', 'prompt': 5},                          # non-string prompt
            {'session': 's', 'prompt': 'x', 'max_new_tokens': 0},
            {'session': 's', 'prompt': 'x', 'max_new_tokens': 257},
            {'session': 's', 'prompt': 'x', 'max_new_tokens': -1},
            {'session': 's', 'prompt': 'x', 'max_new_tokens': '32'},  # wrong type
            {'session': 's', 'prompt': 'x', 'max_new_tokens': True},  # bool is not int
            {'session': 's', 'prompt': 'x', 'max_new_tokens': 1.5},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self._assert_rejected('/generate', payload)
        self.assertEqual(self.h.state.sessions.history('s'), [])

    def test_reset_rejects_invalid_session(self):
        for payload in ({}, {'session': ''}, {'session': 7},
                        {'session': 'bad\x01'}):
            with self.subTest(payload=payload):
                self._assert_rejected('/reset', payload)

    def test_malformed_json_is_400(self):
        for raw in (b'{not json', b'[]', b'"string"', b'null', b'42'):
            with self.subTest(raw=raw):
                code, body = self.h.post_raw('/generate', raw)
                self.assertEqual(code, 400, f'{raw!r} -> {code} {body!r}')
                self.assertIn('error', body)
        self.assertEqual(self.h.get('/health')[1]['queue_depth_total'], 0)

    def test_overlong_prompt_rejected_without_history(self):
        # Stub token counter is len(text); MAX_PROMPT_LEN is 64.
        body = self._assert_rejected('/generate',
                                     {'session': 'long', 'prompt': 'x' * 100})
        self.assertIn('prompt_too_long', body['error'])
        self.assertEqual(self.h.state.sessions.history('long'), [])

    def test_boundary_values_accepted(self):
        # 256 (the range maximum) is exercised in KvCapacityTest: at the
        # default MAX_PROMPT_LEN=64 it exceeds the 128-token KV window.
        for max_new in (1, 64):
            with self.subTest(max_new_tokens=max_new):
                code, body = self.h.post('/generate', {
                    'session': f'ok-{max_new}', 'prompt': 'x',
                    'max_new_tokens': max_new})
                self.assertEqual(code, 200, body)
                self.assertEqual(body['prompt_tokens'], 1)


class TokenizerCountTest(unittest.TestCase):
    """Production token counter needs no numpy (CI installs only pytest)."""

    def test_counts_ndarray_like_and_plain_sequences(self):
        class _Arr:  # stands in for the ndarray returned by ov.Tensor.data
            size = 7
        for data, expected in ((_Arr(), 7), (list(range(5)), 5), ([], 0)):
            class _Tok:
                def encode(self, text, data=data):  # noqa: ARG002
                    return type('E', (), {'input_ids': type(
                        'I', (), {'data': data})()})()
            w = server.Worker.__new__(server.Worker)
            w.tokenizer = _Tok()
            with self.subTest(expected=expected):
                self.assertEqual(w._count_with_tokenizer('x'), expected)

    def test_server_source_does_not_import_numpy(self):
        tree = ast.parse((HERE / 'server.py').read_text())
        names = {a.name.split('.')[0] for n in ast.walk(tree)
                 if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in (n.names if isinstance(n, ast.Import) else
                           [type('A', (), {'name': n.module or ''})()])}
        self.assertNotIn('numpy', names)


class KvCapacityTest(unittest.TestCase):
    """prompt + max_new_tokens beyond MAX_PROMPT_LEN + MIN_RESPONSE_LEN (#13).

    NPUW would silently truncate such a generation, so the server rejects it
    with 400 before queueing.  Stub token counter: len(text).
    """

    def _harness(self, max_prompt_len):
        pipe = StubPipeline()
        h = Harness([pipe], max_prompt_len=max_prompt_len)
        self.addCleanup(h.close)
        return h, pipe

    def test_exact_capacity_accepted_one_over_rejected(self):
        h, pipe = self._harness(64)
        self.assertEqual(h.get('/health')[1]['kv_capacity'], 128)
        # 64 + 64 = 128: fits exactly (cf. probe [192,128] at mpl=256).
        code, body = h.post('/generate', {'session': 'a', 'prompt': 'x' * 64,
                                          'max_new_tokens': 64})
        self.assertEqual(code, 200, body)
        # 5 + 128 = 133 > 128: the recorded re-mp64-bestperf-128tok case
        # (NPUW generated 123 of 128).
        code, body = h.post('/generate', {'session': 'b', 'prompt': 'x' * 5,
                                          'max_new_tokens': 128})
        self.assertEqual(code, 400, body)
        self.assertIn('exceeds_kv_capacity', body['error'])
        self.assertIn('at most 123 tokens', body['error'])
        self.assertEqual(h.state.sessions.history('b'), [])
        self.assertEqual(len(pipe.snapshot_calls()), 1)
        self.assertEqual(h.get('/health')[1]['queue_depth_total'], 0)

    def test_history_counts_toward_capacity(self):
        h, pipe = self._harness(64)
        code, body = h.post('/generate', {'session': 's', 'prompt': 'x' * 30,
                                          'max_new_tokens': 32})
        self.assertEqual(code, 200, body)
        # History is now 'x'*30 + ' reply1' (37 chars); 37 + 'y'*10 = 47
        # prompt tokens, under MAX_PROMPT_LEN, but 47 + 82 = 129 > 128.
        code, body = h.post('/generate', {'session': 's', 'prompt': 'y' * 10,
                                          'max_new_tokens': 82})
        self.assertEqual(code, 400, body)
        self.assertIn('exceeds_kv_capacity', body['error'])
        code, body = h.post('/generate', {'session': 's', 'prompt': 'y' * 10,
                                          'max_new_tokens': 81})
        self.assertEqual(code, 200, body)
        self.assertEqual(len(pipe.snapshot_calls()), 2)

    def test_larger_bucket_admits_range_maximum(self):
        h, _ = self._harness(256)
        self.assertEqual(h.get('/health')[1]['kv_capacity'], 320)
        code, body = h.post('/generate', {'session': 'm', 'prompt': 'x',
                                          'max_new_tokens': 256})
        self.assertEqual(code, 200, body)
        # The recorded near-limit case: 193 + 128 = 321 > 320.
        code, body = h.post('/generate', {'session': 'n',
                                          'prompt': 'x' * 193,
                                          'max_new_tokens': 128})
        self.assertEqual(code, 400, body)
        self.assertIn('at most 127 tokens', body['error'])


class ShutdownTest(unittest.TestCase):
    def test_queued_cancelled_in_flight_finishes(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=4, drain_timeout_s=5.0)
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fin = ex.submit(h.post, '/generate',
                                {'session': 'inflight', 'prompt': 'A'})
                self.assertTrue(started.wait(5))
                fq = ex.submit(h.post, '/generate',
                               {'session': 'queued', 'prompt': 'B'})
                wait_until(lambda: h.get('/health')[1]['queue_depth_total'] == 1,
                           message='second request to be queued')

                code, body = h.post('/shutdown')
                self.assertEqual(code, 200)
                self.assertEqual(body['detail'], 'shutting down')

                # Queued-not-started is cancelled with 503.
                qcode, qbody = fq.result(timeout=10)
                self.assertEqual(qcode, 503)
                self.assertEqual(qbody['error'], 'server shutting down')

                # In-flight request is allowed to finish (shutdown waits for it).
                release.set()
                self.assertEqual(fin.result(timeout=10)[0], 200)
                self.assertTrue(h.state.shutdown_finished.wait(10),
                                'shutdown did not finish')

            report = h.state.shutdown_report
            self.assertEqual(report['cancelled_requests'], 1)
            self.assertEqual(report['workers_stopped'], [0])
            self.assertEqual(report['unstopped_workers'], [])
            self.assertFalse(report['timed_out'])
            self.assertFalse(report['httpd_shutdown_timed_out'])
            self.assertEqual(report['unfinished_handlers'], 0)
            self.assertEqual(h.state.sessions.history('queued'), [])
            # The in-flight request completed and recorded its history.
            self.assertEqual(len(h.state.sessions.history('inflight')), 2)

            # Repeated shutdown is idempotent and returns the same report.
            self.assertIs(server.graceful_shutdown(h.state), report)
            self.assertEqual(h.httpd.socket.fileno(), -1,
                             'listening socket should be closed')
        finally:
            release.set()
            h.close()

    def test_repeated_shutdown_requests_are_idempotent(self):
        h = Harness([StubPipeline()], max_queue=2)
        try:
            first = h.post('/shutdown')[0]
            self.assertEqual(first, 200)
            self.assertTrue(h.state.shutdown_finished.wait(10))
            report = h.state.shutdown_report
            # Second call at the API level is a no-op returning the report.
            self.assertIs(server.graceful_shutdown(h.state), report)
            # And request_shutdown() reports "not newly started".
            self.assertFalse(server.request_shutdown(h.state))
        finally:
            h.close()

    def test_drain_timeout_is_observable(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            started.set()
            release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=2, drain_timeout_s=0.2)
        try:
            client = threading.Thread(
                target=lambda: h.post('/generate',
                                      {'session': 's', 'prompt': 'A'}),
                daemon=True)
            client.start()
            self.assertTrue(started.wait(5))

            t0 = time.monotonic()
            report = server.graceful_shutdown(h.state)
            elapsed = time.monotonic() - t0

            self.assertGreaterEqual(elapsed, 0.15)
            self.assertTrue(report['timed_out'])
            self.assertEqual(report['unstopped_workers'], [0])
            # The handler waiting on the unstopped worker is reported too.
            self.assertEqual(report['unfinished_handlers'], 1)
            self.assertEqual(report['workers_stopped'], [])
            # Not force-exited: the worker is still alive and completes once
            # the generation is allowed to finish.
            self.assertTrue(h.workers[0].thread.is_alive())

            release.set()
            h.workers[0].thread.join(timeout=10)
            self.assertFalse(h.workers[0].thread.is_alive())
        finally:
            release.set()
            h.close()

    def test_no_forced_process_exit_call_in_source(self):
        tree = ast.parse(pathlib.Path(server.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == '_exit':
                    self.fail(f'os._exit() call found at line {node.lineno}')
                if (isinstance(func, ast.Name) and func.id == '_exit'):
                    self.fail(f'_exit() call found at line {node.lineno}')


_CHILD = r"""
import os, sys, threading, time
os.environ['SERVING_ALLOW_IMPORT'] = '1'
sys.path.insert(0, sys.argv[1])
import server

class _Ids:
    def __init__(self, n): self.data = list(range(n))

class _Enc:
    def __init__(self, n): self.input_ids = _Ids(n)

class _Tok:
    def encode(self, text): return _Enc(len(text))

class _Pipe:
    def get_tokenizer(self): return _Tok()
    def generate(self, prompt, max_new_tokens=1, do_sample=False):
        time.sleep(float(os.environ.get('STUB_GEN_S', '0')))
        return ' ok'

server._load_pipeline = lambda props, device='NPU', f32=False: _Pipe()
sys.argv = ['server.py', '--port', sys.argv[2], '--drain-timeout', '5']
server.main()
print('[child] main returned', flush=True)
"""


class ProcessExitTest(unittest.TestCase):
    """Run server.main() in a real child process and check it exits normally."""

    def _run(self, stop):
        import socket
        import subprocess
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, STUB_GEN_S='0.5')
        proc = subprocess.Popen(
            [sys.executable, '-c', _CHILD, str(HERE), str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        try:
            base = f'http://127.0.0.1:{port}'

            def ready():
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1):
                        return True
                except OSError:
                    return False
            wait_until(ready, timeout=10, message='child server ready')

            result = {}

            def client():
                req = urllib.request.Request(
                    base + '/generate', method='POST',
                    data=json.dumps({'session': 's', 'prompt': 'hi'}).encode())
                with urllib.request.urlopen(req, timeout=10) as r:
                    result['status'] = r.status
                    result['body'] = json.loads(r.read())
            t = threading.Thread(target=client)
            t.start()
            # Wait until the request is in flight on the worker, then stop.
            def busy():
                with urllib.request.urlopen(base + '/health', timeout=1) as r:
                    return json.loads(r.read())['busy'] == [True]
            wait_until(busy, timeout=5, message='request in flight')
            stop(proc, base)
            t.join(timeout=10)
            out, _ = proc.communicate(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        return proc.returncode, out, result

    def _check(self, code, out, result):
        self.assertEqual(code, 0, out)
        self.assertIn('[child] main returned', out)
        self.assertIn('[shutdown] stopped workers [0]', out)
        self.assertNotIn('WARNING', out)
        # The in-flight request finished and its response was delivered
        # before the process exited.
        self.assertEqual(result.get('status'), 200, out)
        self.assertEqual(result['body']['text'], ' ok')

    def test_post_shutdown_exits_normally_after_in_flight(self):
        def stop(proc, base):  # noqa: ARG001
            req = urllib.request.Request(base + '/shutdown', method='POST', data=b'{}')
            with urllib.request.urlopen(req, timeout=5) as r:
                self.assertEqual(r.status, 200)
        self._check(*self._run(stop))

    def test_sigterm_exits_normally_after_in_flight(self):
        import signal as _signal
        self._check(*self._run(lambda proc, base: proc.send_signal(_signal.SIGTERM)))


class WorkerRobustnessTest(unittest.TestCase):
    def test_pipeline_exception_returns_500_and_worker_survives(self):
        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                raise RuntimeError('boom')

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=2)
        try:
            code, body = h.post('/generate', {'session': 's', 'prompt': 'A'})
            self.assertEqual(code, 500)
            self.assertIn('boom', body['error'])
            self.assertEqual(h.state.sessions.history('s'), [],
                             'failed generation must not be recorded')

            code, body = h.post('/generate', {'session': 's', 'prompt': 'B'})
            self.assertEqual(code, 200, body)
            self.assertEqual(h.state.sessions.history('s'), ['B', body['text']])
            self.assertEqual([w.errors for w in h.workers], [1])
            self.assertEqual([w.requests_done for w in h.workers], [2])
        finally:
            h.close()


class TokenizerFailureTest(unittest.TestCase):
    def test_tokenizer_exception_returns_json_500_and_releases_gate(self):
        def counter(text):
            if 'BOOM' in text:
                raise RuntimeError('tokenizer exploded')
            return len(text)

        pipe = StubPipeline()
        h = Harness([pipe], max_queue=2, token_counter=counter)
        try:
            code, body = h.post('/generate', {'session': 's', 'prompt': 'BOOM'})
            self.assertEqual(code, 500, body)
            self.assertIn('tokenizer_error', body['error'])
            self.assertEqual(h.state.sessions.history('s'), [])
            self.assertEqual(pipe.snapshot_calls(), [])
            # The session gate was released: the same session still works.
            code, body = h.post('/generate', {'session': 's', 'prompt': 'ok'})
            self.assertEqual(code, 200, body)
        finally:
            h.close()

    def test_history_limit_is_still_400(self):
        pipe = StubPipeline()
        h = Harness([pipe], max_queue=2, max_prompt_len=10_000,
                    token_counter=lambda text: len(text))
        try:
            code, body = h.post('/generate', {
                'session': 's', 'prompt': 'x' * (server.HISTORY_LIMIT_TOKENS + 1)})
            self.assertEqual(code, 400, body)
            self.assertIn('exceeds', body['error'])
        finally:
            h.close()


class GenerateTimeoutTest(unittest.TestCase):
    def test_timeout_504_is_bounded_and_abandoned_result_not_recorded(self):
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                release.wait(10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=2)
        saved = server.GENERATE_TIMEOUT_S
        server.GENERATE_TIMEOUT_S = 0.2
        try:
            t0 = time.monotonic()
            code, body = h.post('/generate', {'session': 's', 'prompt': 'A'})
            self.assertEqual(code, 504, body)
            # Delivered while the generation is still blocked (not unbounded).
            self.assertFalse(release.is_set())
            self.assertLess(time.monotonic() - t0, 5.0)
            server.GENERATE_TIMEOUT_S = saved
            release.set()
            wait_until(lambda: h.get('/health')[1]['busy'] == [False],
                       message='abandoned generation finished')
            self.assertEqual(h.state.sessions.history('s'), [],
                             'timed-out generation must not be recorded')
            code, body = h.post('/generate', {'session': 's', 'prompt': 'B'})
            self.assertEqual(code, 200, body)
            self.assertEqual(h.state.sessions.history('s'), ['B', body['text']])
        finally:
            server.GENERATE_TIMEOUT_S = saved
            release.set()
            h.close()


class HealthAndProtocolTest(unittest.TestCase):
    def test_health_fields_and_endpoint_shapes(self):
        pipe = StubPipeline()
        h = Harness([pipe], max_queue=3)
        try:
            code, health = h.get('/health')
            self.assertEqual(code, 200)
            for key in ('ok', 'shutting_down', 'workers', 'requests_done',
                        'errors', 'cancelled', 'queue_depth', 'queue_capacity',
                        'queue_depth_total', 'queue_capacity_total', 'busy'):
                self.assertIn(key, health)
            self.assertEqual(health['queue_capacity_total'], 3)
            self.assertFalse(health['shutting_down'])

            code, body = h.post('/generate',
                                {'session': 's', 'prompt': 'hi',
                                 'max_new_tokens': 8})
            self.assertEqual(code, 200)
            self.assertEqual(body['session'], 's')
            self.assertEqual(body['prompt_tokens'], 2)
            self.assertEqual(body['text'], ' reply1')
            self.assertIn('gen_s', body)
            self.assertIn('worker', body)

            self.assertEqual(h.post('/reset', {'session': 's'})[0], 200)
            self.assertEqual(h.state.sessions.history('s'), [])
            self.assertEqual(h.get('/nope')[0], 404)
            self.assertEqual(h.post('/nope', {})[0], 404)
        finally:
            h.close()


class FakeClock:
    """Strictly increasing monotonic clock (deterministic idle ordering)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._now = 0.0

    def __call__(self) -> float:
        with self._lock:
            self._now += 1.0
            return self._now


class SessionCapacityTest(unittest.TestCase):
    """Issue #12: bounded session state with idle LRU eviction."""

    def test_idle_lru_evicts_oldest_completed_and_evicted_id_starts_fresh(self):
        clock = FakeClock()
        pipe = StubPipeline()
        h = Harness([pipe], max_queue=4, max_sessions=3, session_clock=clock)
        try:
            for i in range(1, 4):
                code, body = h.post('/generate',
                                    {'session': f's{i}', 'prompt': f'p{i}'})
                self.assertEqual(code, 200, body)
            # Touch s1 so s2 is the least recently *completed* session.
            self.assertEqual(h.post('/generate',
                                    {'session': 's1', 'prompt': 'again'})[0], 200)

            health = h.get('/health')[1]
            self.assertEqual(health['sessions'], 3)
            self.assertEqual(health['sessions_active'], 0)
            self.assertEqual(health['session_capacity'], 3)
            self.assertEqual(health['session_evictions'], 0)

            code, body = h.post('/generate', {'session': 's4', 'prompt': 'p4'})
            self.assertEqual(code, 200, body)
            health = h.get('/health')[1]
            self.assertEqual(health['sessions'], 3)
            self.assertEqual(health['session_evictions'], 1)
            self.assertEqual(h.state.sessions.history('s2'), [],
                             'least recently completed idle session should be evicted')
            self.assertEqual(len(h.state.sessions.history('s1')), 4)
            self.assertEqual(len(h.state.sessions.history('s3')), 2)
            self.assertEqual(len(h.state.sessions.history('s4')), 2)

            # An evicted-but-valid session id is not an error: it is admitted
            # fresh (empty history) and served normally.
            calls_before = len(pipe.snapshot_calls())
            code, body = h.post('/generate', {'session': 's2', 'prompt': 'fresh'})
            self.assertEqual(code, 200, body)
            self.assertEqual(pipe.snapshot_calls()[calls_before], 'fresh',
                             'evicted session must restart with an empty history')
            self.assertEqual(h.state.sessions.history('s2'), ['fresh', body['text']])
            self.assertEqual(h.get('/health')[1]['session_evictions'], 2)
        finally:
            h.close()

    def test_active_session_is_pinned_and_new_session_rejected_when_all_active(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=4, max_sessions=1)
        try:
            with ThreadPoolExecutor(max_workers=3) as ex:
                fbusy = ex.submit(h.post, '/generate',
                                  {'session': 'busy', 'prompt': 'A'})
                self.assertTrue(started.wait(5), 'generation did not start')
                # A same-session waiter also pins the entry (it holds a ticket).
                fwait = ex.submit(h.post, '/generate',
                                  {'session': 'busy', 'prompt': 'B'})
                wait_until(lambda: h.state.sessions.gate('busy').pending() == 2,
                           message='second operation waiting on the gate')

                code, body = h.post('/generate',
                                    {'session': 'new', 'prompt': 'N'})
                self.assertEqual(code, 429, body)
                self.assertIn('session capacity', body['error'])
                # Attempted eviction of the busy session was a no-op.
                health = h.get('/health')[1]
                self.assertEqual(health['sessions'], 1)
                self.assertEqual(health['sessions_active'], 1)
                self.assertEqual(health['session_evictions'], 0)
                self.assertEqual(health['requests_done'], [0])
                self.assertEqual(h.state.sessions.history('new'), [])

                release.set()
                self.assertEqual(fbusy.result(timeout=10)[0], 200)
                self.assertEqual(fwait.result(timeout=10)[0], 200)

            self.assertEqual(len(h.state.sessions.history('busy')), 4)
            self.assertEqual(h.get('/health')[1]['session_evictions'], 0)

            # Once idle, the single slot is evictable and 'new' is admitted.
            code, body = h.post('/generate', {'session': 'new', 'prompt': 'N'})
            self.assertEqual(code, 200, body)
            health = h.get('/health')[1]
            self.assertEqual(health['sessions'], 1)
            self.assertEqual(health['session_evictions'], 1)
            self.assertEqual(h.state.sessions.history('busy'), [])
        finally:
            release.set()
            h.close()

    def test_concurrent_first_requests_share_one_gate(self):
        n = 8
        pipe = StubPipeline()
        h = Harness([pipe], max_queue=n, max_sessions=4, max_prompt_len=10_000)
        seen = []
        orig_acquire = h.state.sessions.acquire

        def spy(session, max_pending=None):
            entry, ticket = orig_acquire(session, max_pending)
            if entry is not None:
                seen.append(entry)
            return entry, ticket

        h.state.sessions.acquire = spy
        try:
            barrier = threading.Barrier(n)

            def call(i):
                barrier.wait(timeout=10)
                return h.post('/generate',
                              {'session': 'shared', 'prompt': f'p{i}'})

            with ThreadPoolExecutor(max_workers=n) as ex:
                futures = [ex.submit(call, i) for i in range(n)]
                results = [f.result(timeout=20) for f in futures]

            self.assertEqual([code for code, _ in results], [200] * n)
            self.assertEqual(len(seen), n)
            self.assertEqual(len({id(entry) for entry in seen}), 1,
                             'concurrent first requests must share one entry/gate')
            health = h.get('/health')[1]
            self.assertEqual(health['sessions'], 1)
            self.assertEqual(health['sessions_active'], 0)
            self.assertEqual(health['session_evictions'], 0)
            self.assertEqual(len(h.state.sessions.history('shared')), 2 * n)
            # One gate serializes the operations: every prompt extends the
            # previous one (two gates would let snapshots fork).
            calls = pipe.snapshot_calls()
            self.assertEqual(len(calls), n)
            for earlier, later in zip(calls, calls[1:]):
                self.assertTrue(later.startswith(earlier) and len(later) > len(earlier),
                                'same-session generations overlapped')
        finally:
            h.close()

    def test_concurrent_admission_at_limit_never_bypasses_capacity(self):
        started = threading.Event()
        release = threading.Event()

        def block(prompt, index):  # noqa: ARG001
            if index == 1:
                started.set()
                release.wait(timeout=10)

        n = 12
        pipe = StubPipeline(block=block)
        h = Harness([pipe], max_queue=n, max_sessions=4)
        try:
            barrier = threading.Barrier(n)
            results: list[tuple[int, dict]] = []
            results_lock = threading.Lock()

            def call(i):
                barrier.wait(timeout=10)
                outcome = h.post('/generate',
                                 {'session': f'c{i}', 'prompt': f'p{i}'})
                with results_lock:
                    results.append(outcome)
                return outcome

            with ThreadPoolExecutor(max_workers=n) as ex:
                futures = [ex.submit(call, i) for i in range(n)]
                self.assertTrue(started.wait(5), 'first generation did not start')
                # The four admitted sessions are pinned (one in flight, three
                # holding tickets); the other eight are rejected deterministically.
                wait_until(lambda: len(results) == n - 4,
                           timeout=10,
                           message='rejections at session capacity')
                codes = sorted(code for code, _ in results)
                self.assertEqual(codes, [429] * (n - 4))

                health = h.get('/health')[1]
                self.assertEqual(health['sessions'], 4)
                self.assertEqual(health['sessions_active'], 4)
                self.assertEqual(health['session_evictions'], 0)

                release.set()
                for f in futures:
                    code, body = f.result(timeout=20)
                    self.assertIn(code, (200, 429), body)

            wins = [body for code, body in results if code == 200]
            self.assertEqual(len(wins), 4)
            health = h.get('/health')[1]
            self.assertEqual(health['sessions'], 4)
            self.assertLessEqual(health['sessions'], health['session_capacity'])
            self.assertEqual(health['sessions_active'], 0)
            self.assertEqual(health['session_evictions'], 0)
            for body in wins:
                self.assertEqual(len(h.state.sessions.history(body['session'])), 2)

            # With all four slots now idle, a 13th session evicts exactly one.
            code, body = h.post('/generate', {'session': 'extra', 'prompt': 'x'})
            self.assertEqual(code, 200, body)
            health = h.get('/health')[1]
            self.assertEqual(health['sessions'], 4)
            self.assertEqual(health['session_evictions'], 1)
        finally:
            release.set()
            h.close()

    def test_max_sessions_must_be_positive(self):
        with self.assertRaises(ValueError):
            server.SessionStore(max_sessions=0)
        self.assertGreaterEqual(server.DEFAULT_MAX_SESSIONS, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class MessagesShimTest(unittest.TestCase):
    """POST /v1/messages — Anthropic Messages API shape (text-only v1)."""

    def test_non_streaming_roundtrip(self):
        pipe = StubPipeline()
        h = Harness([pipe])
        try:
            code, body = h.post('/v1/messages', {
                'model': 'local-qwen3',
                'max_tokens': 32,
                'system': 'Be terse.',
                'messages': [
                    {'role': 'user', 'content': 'What is 2+2?'},
                    {'role': 'assistant', 'content': '4'},
                    {'role': 'user', 'content': 'And 2+3?'},
                ]})
            self.assertEqual(code, 200)
            self.assertEqual(body['type'], 'message')
            self.assertEqual(body['role'], 'assistant')
            self.assertEqual(body['model'], 'local-qwen3')
            self.assertEqual(len(body['content']), 1)
            self.assertEqual(body['content'][0]['type'], 'text')
            self.assertIn(body['stop_reason'], ('end_turn', 'max_tokens'))
            self.assertIn('input_tokens', body['usage'])
            # conversation flattening: system + both roles present
            self.assertIn('Be terse.', pipe.calls[0])
            self.assertIn('Human: What is 2+2?', pipe.calls[0])
            self.assertIn('Assistant: 4', pipe.calls[0])
            self.assertIn('Human: And 2+3?', pipe.calls[0])
            self.assertTrue(pipe.calls[0].endswith('Assistant:'))
        finally:
            h.close()

    def test_tools_rejected_explicitly(self):
        pipe = StubPipeline()
        h = Harness([pipe])
        try:
            code, body = h.post('/v1/messages', {
                'model': 'local', 'max_tokens': 8,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'tools': [{'name': 'x', 'input_schema': {}}]})
            self.assertEqual(code, 400)
            self.assertEqual(body['type'], 'error')
            self.assertIn('tool use', body['error']['message'])
            self.assertEqual(len(pipe.calls), 0)
        finally:
            h.close()

    def test_structured_content_blocks_flattened(self):
        pipe = StubPipeline()
        h = Harness([pipe])
        try:
            code, body = h.post('/v1/messages', {
                'max_tokens': 8,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'part one '},
                    {'type': 'text', 'text': 'part two'},
                ]}]})
            self.assertEqual(code, 200)
            self.assertIn('part one \npart two', pipe.calls[0])
        finally:
            h.close()

    def test_invalid_requests(self):
        pipe = StubPipeline()
        h = Harness([pipe])
        try:
            code, body = h.post('/v1/messages', {'max_tokens': 8})
            self.assertEqual(code, 400)
            code, body = h.post('/v1/messages', {
                'messages': [{'role': 'user', 'content': 'x'}],
                'max_tokens': 99999})
            self.assertEqual(code, 400)
            self.assertEqual(body['error']['type'], 'invalid_request_error')
        finally:
            h.close()
