#!/usr/bin/env python3
"""Behavioural tests for serving/server.py using a stub pipeline.

Pure standard library; no NPU, model or OpenVINO runtime is required.  The
real ``genai.LLMPipeline`` is replaced by :class:`StubPipeline`, injected
through :class:`server.Worker`'s ``pipeline_factory``/``token_counter`` hooks.
The HTTP server is started in-process on an ephemeral port.

Covered behaviour (issues #3, #4, #7):
  * same-session requests serialize in arrival order and history is atomic;
  * different sessions execute concurrently (two workers);
  * reset waits for earlier same-session work, then clears history;
  * bounded queue saturation returns 429 without queueing or mutating state;
  * malformed / missing / out-of-range fields return 400 before queueing;
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
                 token_counter=None):
        self.sessions = server.SessionStore()
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
        for max_new in (1, 256):
            with self.subTest(max_new_tokens=max_new):
                code, body = self.h.post('/generate', {
                    'session': f'ok-{max_new}', 'prompt': 'x',
                    'max_new_tokens': max_new})
                self.assertEqual(code, 200, body)
                self.assertEqual(body['prompt_tokens'], 1)


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
