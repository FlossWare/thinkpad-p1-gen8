#!/usr/bin/env python3
"""M1 serving-side probe: queue wait, per-turn latency, and session isolation
under growing conversation history, over HTTP against the real NPU server.

Measures (per MAX_PROMPT_LEN bucket):
  - queue wait: submit -> worker start (client wall minus the server-reported
    generation time), per turn, as a conversation grows
  - per-turn latency decomposition (queue wait + server gen time)
  - session isolation: two sessions interleaved on one server — each turn's
    latency must not depend on the other session's history length

Usage:
  m1_server_probe.py [--port 8943] [--bucket 256] [--turns 8] [--out FILE]

Requires the server running with a matching bucket, e.g.:
  python3 serving/server.py --port 8943 --max-prompt-len 256 \
      --history-limit 320 --max-new... (defaults are fine)
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

if __name__ != '__main__':
    raise SystemExit(0)

USER_PROMPTS = [
    'What is the difference between latency and throughput?',
    'Explain how a CPU cache hierarchy works.',
    'Describe the design of a modern solid-state drive.',
    'Compare interpreted and compiled languages.',
    'Walk through what happens when a browser loads a web page.',
    'Explain consensus in distributed systems.',
    'Explain the memory hierarchy of a modern computer in depth.',
    'What is a hash collision and how is it handled?',
]


def post(port, path, obj, timeout=300):
    if path == '/health':
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',
                                    timeout=timeout) as r:
            return json.loads(r.read())
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}', data=json.dumps(obj).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # 4xx rejections (e.g. prompt_too_long at KV capacity) are outcomes,
        # not crashes: surface the server's error body to the caller.
        try:
            body = json.loads(e.read())
        except Exception:  # noqa: BLE001
            body = {'error': f'HTTP {e.code}'}
        return {'error': body.get('error', f'HTTP {e.code}'),
                'status': e.code}


def conversation_turns(port, session, turns, reply_tokens=64):
    """Grow one session; per turn record queue wait and server gen time."""
    out = []
    for t in range(turns):
        prompt = USER_PROMPTS[t % len(USER_PROMPTS)]
        t0 = time.perf_counter()
        r = post(port, '/generate',
                 {'session': session, 'prompt': prompt,
                  'max_new_tokens': reply_tokens})
        wall = time.perf_counter() - t0
        gen = r.get('gen_s')
        out.append({
            'turn': t,
            'prompt_tokens': r.get('prompt_tokens'),
            'client_wall_s': round(wall, 3),
            'server_gen_s': gen,
            'queue_wait_s': round(wall - gen, 3) if gen is not None else None,
        })
        if 'error' in r:
            out[-1]['error'] = r['error']
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8943)
    ap.add_argument('--bucket', type=int, default=256)
    ap.add_argument('--turns', type=int, default=8)
    ap.add_argument('--sessions', type=int, default=2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    health = post(args.port, '/health', {})
    print(f'[health] {health}', flush=True)

    result = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'bucket': args.bucket, 'turns': args.turns,
        'health_at_start': health,
        'sessions': [],
    }

    # 1. Single-session growth (queue wait should stay ~0 with no contention)
    run_id = f'{int(time.time())}'
    result['sessions'].append({
        'session': f'm1-{run_id}-solo', 'mode': 'solo',
        'turns': conversation_turns(args.port, f'm1-{run_id}-solo', args.turns)})
    print(f"[solo] done: {[t['client_wall_s'] for t in result['sessions'][0]['turns']]}",
          flush=True)

    # 2. Interleaved sessions: alternate turns; isolation means each session's
    #    per-turn latency tracks its OWN history, not the other's.
    names = [f'm1-{run_id}-iso-{i}' for i in range(args.sessions)]
    inter = {n: [] for n in names}
    for t in range(args.turns):
        for n in names:
            prompt = USER_PROMPTS[t % len(USER_PROMPTS)]
            t0 = time.perf_counter()
            r = post(args.port, '/generate',
                     {'session': n, 'prompt': prompt, 'max_new_tokens': 64})
            wall = time.perf_counter() - t0
            gen = r.get('gen_s')
            inter[n].append({
                'turn': t, 'client_wall_s': round(wall, 3),
                'server_gen_s': gen,
                'queue_wait_s': round(wall - gen, 3) if gen is not None else None,
                'prompt_tokens': r.get('prompt_tokens'),
            })
            if 'error' in r:
                inter[n][-1]['error'] = r['error']
    for n in names:
        result['sessions'].append({'session': n, 'mode': 'interleaved',
                                   'turns': inter[n]})
        print(f"[iso {n}] {[t['client_wall_s'] for t in inter[n]]}", flush=True)

    result['health_at_end'] = post(args.port, '/health', {})
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=1)
        print(f'[done] wrote {args.out}', flush=True)
    else:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
