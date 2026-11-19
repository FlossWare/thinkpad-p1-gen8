#!/usr/bin/env python3
"""Tests for the bench_serving client's request accounting.

Runs bench_serving.bench() against a stub HTTP server (stdlib only) whose
/generate endpoint returns a deterministic mix of 200 / 400 / 429 / 500
responses, and checks:

  - attempted / successful / failed counts
  - throughput computed over successful requests only
  - failures grouped by error class (http_<status> / exception type)
  - the server's error field surfaced in per-request records
  - a note present in the result line whenever a run had failures

Run: python3 serving/test_bench_client.py
"""
from __future__ import annotations

import argparse
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bench_serving


def make_stub_handler(mixed: bool = True):
    """Deterministic stub /generate. With mixed=True, request i (from the
    session name) maps to i % 5 == 0 -> 400 prompt_too_long, i % 5 == 2 ->
    429 queue_full, i % 5 == 4 -> 500 internal, otherwise 200. With
    mixed=False every request succeeds."""
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self._json(200, {'ok': True})
            else:
                self._json(404, {'error': 'not_found'})

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
            if self.path != '/generate':
                self._json(404, {'error': 'not_found'})
                return
            i = int(str(payload.get('session', 'x-0')).rsplit('-', 1)[1])
            if not mixed:
                self._json(200, {'session': payload['session'],
                                 'text': 'stub answer', 'gen_s': 0.01})
            elif i % 5 == 0:
                self._json(400, {'error': 'prompt_too_long: 70 tokens > '
                                          'MAX_PROMPT_LEN 64'})
            elif i % 5 == 2:
                self._json(429, {'error': 'queue_full: bounded queue is full'})
            elif i % 5 == 4:
                self._json(500, {'error': 'internal: simulated worker crash'})
            else:
                self._json(200, {'session': payload['session'],
                                 'text': 'stub answer', 'gen_s': 0.01})

    return Handler


class StubServer:
    def __init__(self, mixed: bool = True):
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0),
                                          make_stub_handler(mixed))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def ns(**kw):
    base = dict(mode='sequential', n=10, clients=4, port=0, label=None,
                max_new=16)
    base.update(kw)
    return argparse.Namespace(**base)


class BenchClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = StubServer()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()

    def test_mixed_outcomes_sequential(self):
        res = bench_serving.bench(ns(port=self.stub.port, n=10,
                                     label='stub-mixed-seq'))
        # n=10: failures at i in {0,5} -> 400, {2,7} -> 429, {4,9} -> 500.
        self.assertEqual(res['attempted'], 10)
        self.assertEqual(res['successful'], 4)
        self.assertEqual(res['failed'], 6)
        self.assertEqual(res['errors'], 6)  # legacy alias
        self.assertEqual(res['errors_by_class'],
                         {'http_400': 2, 'http_429': 2, 'http_500': 2})
        # Throughput counts successful requests only: tokens_per_s is the
        # successful-only request rate scaled by tokens per request, and the
        # attempted rate stays strictly above it.
        self.assertGreater(res['successful_req_per_s'], 0.0)
        self.assertGreater(res['attempted_req_per_s'],
                           res['successful_req_per_s'])
        self.assertAlmostEqual(res['tokens_per_s'],
                               res['successful_req_per_s'] * 16, delta=2.0)
        # A run with failures carries an explicit note.
        self.assertIn('6 of 10 requests failed', res['note'])
        self.assertIn('http_400=2', res['note'])
        # Per-request records: 10 of them, ordered, with full outcome detail.
        recs = res['requests']
        self.assertEqual(len(recs), 10)
        self.assertEqual([r['i'] for r in recs], list(range(10)))
        for r in recs:
            self.assertIn('session', r)
            self.assertIn('mode', r)
            self.assertIn('latency_s', r)
        bad = {r['i']: r for r in recs if not r['ok']}
        good = {r['i']: r for r in recs if r['ok']}
        self.assertEqual(sorted(bad), [0, 2, 4, 5, 7, 9])
        self.assertEqual(sorted(good), [1, 3, 6, 8])
        self.assertIn('prompt_too_long', bad[0]['error'])
        self.assertEqual(bad[0]['status'], 400)
        self.assertIn('queue_full', bad[2]['error'])
        self.assertEqual(bad[2]['status'], 429)
        self.assertIn('internal', bad[4]['error'])
        self.assertEqual(bad[4]['status'], 500)
        self.assertIsNone(good[1]['error'])
        self.assertEqual(good[1]['status'], 200)
        # Latency stats exist (successful-only sample is non-empty here).
        self.assertIsNotNone(res['latency_p50_s'])

    def test_mixed_outcomes_concurrent(self):
        res = bench_serving.bench(ns(mode='concurrent', port=self.stub.port,
                                     n=10, clients=4, label='stub-mixed-conc'))
        self.assertEqual(res['attempted'], 10)
        self.assertEqual(res['successful'], 4)
        self.assertEqual(res['failed'], 6)
        self.assertEqual(res['errors_by_class'],
                         {'http_400': 2, 'http_429': 2, 'http_500': 2})
        self.assertEqual(len(res['requests']), 10)
        self.assertEqual([r['i'] for r in res['requests']], list(range(10)))
        self.assertTrue(all(r['mode'] == 'concurrent' for r in res['requests']))

    def test_all_success(self):
        clean = StubServer(mixed=False)
        try:
            res = bench_serving.bench(ns(port=clean.port, n=5,
                                         label='stub-clean'))
        finally:
            clean.stop()
        self.assertEqual(res['attempted'], 5)
        self.assertEqual(res['successful'], 5)
        self.assertEqual(res['failed'], 0)
        self.assertEqual(res['errors_by_class'], {})
        self.assertNotIn('note', res)  # no failure note on a clean run
        self.assertGreater(res['successful_req_per_s'], 0.0)
        self.assertAlmostEqual(res['tokens_per_s'],
                               res['successful_req_per_s'] * 16, delta=2.0)
        self.assertTrue(all(r['ok'] for r in res['requests']))

    def test_connection_errors_are_recorded_not_crashes(self):
        # Nothing listens on this port: every request must become a recorded
        # failure (URLError), and throughput must be 0, not n / wall.
        res = bench_serving.bench(ns(port=1, n=3, label='stub-conn-refused'))
        self.assertEqual(res['attempted'], 3)
        self.assertEqual(res['successful'], 0)
        self.assertEqual(res['failed'], 3)
        self.assertEqual(sum(res['errors_by_class'].values()), 3)
        self.assertIn('URLError', res['errors_by_class'])
        self.assertEqual(res['successful_req_per_s'], 0.0)
        self.assertEqual(res['tokens_per_s'], 0.0)
        self.assertIn('note', res)
        self.assertIsNone(res['latency_p50_s'])
        self.assertTrue(all(r['status'] == 'exception' for r in res['requests']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
