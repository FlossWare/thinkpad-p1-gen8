#!/usr/bin/env python3
"""Benchmark client for the NPU serving prototype.

Modes:
  sequential   - N requests, one at a time (queue never builds up)
  concurrent   - N requests fired simultaneously from C client threads
                 (measures queueing on a single pipeline)
The two-pipeline arrangement is measured by running the server with
--workers 2 and re-running the same modes (round-robin dispatch).

Usage: bench_serving.py [--mode sequential|concurrent] [--n 20] [--clients 4]
                        [--port 8931] [--label NAME] [--out results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    'The capital of France is',
    'Water boils at a temperature of',
    'The largest planet in the solar system is',
    'The first president of the United States was',
    'Photosynthesis converts sunlight into',
]


def post(port: int, path: str, obj: dict, timeout: int = 300) -> dict:
    if path == '/health':
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=timeout) as r:
            return json.loads(r.read())
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}',
        data=json.dumps(obj).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bench(args: argparse.Namespace) -> dict:
    t_start = time.perf_counter()
    latencies: list[float] = []
    errors = 0

    run_id = f'{int(time.time()*1000) % 100000}'

    def one(i: int) -> None:
        prompt = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        try:
            # Unique session per request (and per run): measures request
            # throughput, not conversation growth.
            r = post(args.port, '/generate',
                     {'session': f'bench-{run_id}-{i}',
                      'prompt': prompt, 'max_new_tokens': 16})
            latencies.append(time.perf_counter() - t0)
            if 'error' in r:
                raise RuntimeError(r['error'])
        except Exception as e:  # noqa: BLE001
            nonlocal errors
            errors += 1
            print(f'[error] request {i}: {e}', flush=True)

    if args.mode == 'sequential':
        for i in range(args.n):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=args.clients) as ex:
            list(ex.map(one, range(args.n)))

    wall = time.perf_counter() - t_start
    latencies.sort()
    res = {
        'label': args.label or f'{args.mode}-n{args.n}-c{args.clients}',
        'mode': args.mode,
        'n': args.n,
        'clients': args.clients,
        'errors': errors,
        'wall_s': round(wall, 2),
        'throughput_req_per_s': round(args.n / wall, 3) if wall else None,
        'tokens_per_s': round(args.n * 16 / wall, 1) if wall else None,
        'latency_p50_s': round(latencies[len(latencies)//2], 3) if latencies else None,
        'latency_p95_s': round(latencies[int(len(latencies)*0.95)], 3) if latencies else None,
        'latency_max_s': round(latencies[-1], 3) if latencies else None,
    }
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
    args = ap.parse_args()

    h = post(args.port, '/health', {})
    print(f'[health] {h}', flush=True)
    res = bench(args)
    h2 = post(args.port, '/health', {})
    print(f'[health-after] {h2}', flush=True)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps({**res, 'health_after': h2}) + '\n')


if __name__ == '__main__':
    main()
