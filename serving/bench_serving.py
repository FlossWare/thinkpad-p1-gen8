#!/usr/bin/env python3
"""Benchmark client for the NPU serving prototype.

Modes:
  sequential   - N requests, one at a time (queue never builds up)
  concurrent   - N requests fired simultaneously from C client threads
                 (measures queueing on a single pipeline)
The two-pipeline arrangement is measured by running the server with
--workers 2 and re-running the same modes (round-robin dispatch).

Accounting: every request is recorded with its outcome (ok / HTTP status /
exception). Throughput is computed over SUCCESSFUL completions only
(successful_req_per_s, tokens_per_s); wall time and the attempted count are
reported separately so failed requests can never inflate throughput. Any
4xx/5xx response is recorded as a failed request (never a crash) and the
response body's "error" field, when present, is surfaced verbatim.

tokens_per_s is NOMINAL: successful requests x max_new_tokens / wall. The
server response does not report a generated-token count, so a request that
stops early at EOS is still counted as max_new_tokens (upper bound); the
result line records this in tokens_per_s_basis.

Usage: bench_serving.py [--mode sequential|concurrent] [--n 20] [--clients 4]
                        [--port 8931] [--label NAME] [--out results.jsonl]
                        [--max-new 16]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    'The capital of France is',
    'Water boils at a temperature of',
    'The largest planet in the solar system is',
    'The first president of the United States was',
    'Photosynthesis converts sunlight into',
]


def get_health(port: int, timeout: int = 300) -> dict:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',
                                timeout=timeout) as r:
        return json.loads(r.read())


def post_generate(port: int, obj: dict, timeout: int = 300) -> tuple[int, dict]:
    """POST obj to /generate and return (http_status, parsed_body).

    HTTP error statuses (4xx/5xx) are returned, not raised, so the benchmark
    can record them as failed requests. Transport-level problems (connection
    refused, timeout, ...) still raise.
    """
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/generate',
        data=json.dumps(obj).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        finally:
            e.close()
        try:
            body = json.loads(raw or b'{}')
        except ValueError:
            body = {'error': raw.decode('utf-8', 'replace').strip()[:200]
                    or f'http status {e.code} with empty body'}
        return e.code, body


def nearest_rank(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list (q in (0, 1]).

    For n=20 and q=0.95 this is the 19th smallest value, not the maximum
    (the pre-2026-11-19 client used index int(0.95*n), i.e. the maximum).
    """
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


def error_class(rec: dict) -> str:
    """Group a failed request under an error class: HTTP status or exception type."""
    if rec['status'] == 'exception':
        return rec['error'].split(':', 1)[0]  # exception type name
    return f'http_{rec["status"]}'


def bench(args: argparse.Namespace) -> dict:
    t_start = time.perf_counter()

    run_id = f'{int(time.time()*1000) % 100000}'

    def one(i: int) -> dict:
        session = f'bench-{run_id}-{i}'
        prompt = PROMPTS[i % len(PROMPTS)]
        rec: dict = {'i': i, 'session': session, 'mode': args.mode,
                     'ok': False, 'latency_s': None, 'status': None,
                     'error': None}
        t0 = time.perf_counter()
        try:
            # Unique session per request (and per run): measures request
            # throughput, not conversation growth.
            status, body = post_generate(args.port, {
                'session': session, 'prompt': prompt,
                'max_new_tokens': args.max_new})
            rec['status'] = status
            if status == 200 and 'error' not in body:
                rec['ok'] = True
            else:
                # Surface the server's error field when present.
                rec['error'] = str(body.get('error')
                                   or f'http status {status} without error body')
        except Exception as e:  # noqa: BLE001
            rec['status'] = 'exception'
            rec['error'] = f'{type(e).__name__}: {e}'
        rec['latency_s'] = round(time.perf_counter() - t0, 3)
        if not rec['ok']:
            print(f'[error] request {i}: {rec["error"]}', flush=True)
        return rec

    if args.mode == 'sequential':
        records = [one(i) for i in range(args.n)]
    else:
        with ThreadPoolExecutor(max_workers=args.clients) as ex:
            records = list(ex.map(one, range(args.n)))  # map preserves order
    records.sort(key=lambda r: r['i'])

    wall = time.perf_counter() - t_start
    attempted = len(records)
    successful = sum(1 for r in records if r['ok'])
    failed = attempted - successful
    latencies = sorted(r['latency_s'] for r in records if r['ok'])
    errors_by_class: dict[str, int] = {}
    for r in records:
        if not r['ok']:
            errors_by_class[error_class(r)] = errors_by_class.get(
                error_class(r), 0) + 1

    res = {
        'label': args.label or f'{args.mode}-n{args.n}-c{args.clients}',
        'mode': args.mode,
        'n': args.n,
        'clients': args.clients,
        'max_new_tokens': args.max_new,
        'attempted': attempted,
        'successful': successful,
        'failed': failed,
        'errors': failed,  # legacy alias: count of failed requests
        'wall_s': round(wall, 2),
        # Throughput counts SUCCESSFUL completions only; wall time and the
        # attempted count are reported separately so nothing is hidden.
        'successful_req_per_s': round(successful / wall, 3) if wall else None,
        'attempted_req_per_s': round(attempted / wall, 3) if wall else None,
        'tokens_per_s': round(successful * args.max_new / wall, 1)
                        if wall else None,
        'tokens_per_s_basis': 'nominal: successful * max_new_tokens / wall_s '
                              '(server does not report generated-token counts)',
        'errors_by_class': errors_by_class,
        'requests': records,
        'latency_p50_s': round(latencies[len(latencies)//2], 3) if latencies else None,
        'latency_p95_s': round(nearest_rank(latencies, 0.95), 3) if latencies else None,
        'latency_percentile_method': 'p95: nearest-rank (ceil(0.95*n)-th '
                                     'smallest); p50: element n//2 (upper median)',
        'latency_max_s': round(latencies[-1], 3) if latencies else None,
    }
    if failed:
        classes = ', '.join(f'{c}={n}' for c, n in sorted(errors_by_class.items()))
        res['note'] = (f'{failed} of {attempted} requests failed ({classes}); '
                       'successful_req_per_s and tokens_per_s count successful '
                       'requests only')
    print(json.dumps(res), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='sequential', choices=('sequential', 'concurrent'))
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--clients', type=int, default=4)
    ap.add_argument('--port', type=int, default=8931)
    ap.add_argument('--label', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-new', type=int, default=16,
                    help='max_new_tokens sent per request (used for tokens_per_s)')
    args = ap.parse_args()

    h = get_health(args.port)
    print(f'[health] {h}', flush=True)
    res = bench(args)
    h2 = get_health(args.port)
    print(f'[health-after] {h2}', flush=True)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps({**res, 'health_after': h2}) + '\n')


if __name__ == '__main__':
    main()
