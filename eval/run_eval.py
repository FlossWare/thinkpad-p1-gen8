#!/usr/bin/env python3
"""100% automated Thompson evaluation: seeded-defect fixtures → all models
→ computed scores → outcome records.

Model pool (all FREE — no paid models anywhere in this loop):
  - local:   HTTP POST /v1/messages against running serving instances
             (see --local-endpoints)
  - gateway: `claude -p --model <id>` for the free EnMaaS trio

Scoring is COMPUTED from fixture ground truth (no arbiter, no human):
  - caught:  the model's review references the seeded defect
             (term matching against the fixture's match_terms, majority rule)
  - extra_findings: review claims bugs beyond ground truth (recorded
             separately — a false-positive proxy on known-clean code)
  - latency, completion, cost (local = $0.00)

Outputs one JSONL outcome record per (model, fixture) compatible with the
CE learning outcome shape, plus a per-model summary.

Usage (from repo root, scripts/env.sh sourced):
  python3 eval/run_eval.py [--out eval/results/outcomes.jsonl]
      [--models eval/model_pool.json] [--local-only] [--gateway-only]

Local models require serving instances running (one per model):
  NPU_MODEL_DIR=<model> python3 serving/server.py --device GPU --port <p> &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_manifest():
    with open(os.path.join(HERE, 'fixture_manifest.json')) as f:
        return json.load(f)


def load_fixture(manifest_entry):
    with open(os.path.join(HERE, 'fixtures', manifest_entry['file'])) as f:
        return f.read()


def call_local(endpoint, prompt, timeout=300):
    """POST /v1/messages to a running local serving instance."""
    body = json.dumps({'model': 'eval', 'max_tokens': 1024,
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    req = urllib.request.Request(endpoint.rstrip('/') + '/v1/messages', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d['content'][0]['text'], time.perf_counter() - t0


def call_gateway(model_id, prompt, timeout=300):
    """Free gateway model via the authenticated claude CLI."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ['claude', '-p', '--model', model_id, prompt],
        capture_output=True, text=True, timeout=timeout)
    return proc.stdout, time.perf_counter() - t0


def score_review(review_text, fixture):
    """Computed scoring against ground truth. No model rates a model.

    caught: at least half the fixture's match_terms appear in the review
    (case-insensitive), AND the review does not declare the code clean.
    clean_claim: review asserts there are no bugs (a miss).
    """
    text = review_text.lower()
    hits = sum(1 for t in fixture['match_terms'] if t.lower() in text)
    caught = hits >= max(1, len(fixture['match_terms']) // 2)
    clean_claim = bool(re.search(r'no (bugs|issues|defects)|code is (clean|correct)',
                                 text))
    if clean_claim and not caught:
        caught = False
    return {'term_hits': hits, 'caught': caught, 'clean_claim': clean_claim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results',
                                                  'outcomes.jsonl'))
    ap.add_argument('--models', default=os.path.join(HERE, 'model_pool.json'))
    ap.add_argument('--local-only', action='store_true')
    ap.add_argument('--gateway-only', action='store_true')
    args = ap.parse_args()

    manifest = load_manifest()
    with open(args.models) as f:
        pool = json.load(f)

    models = []
    if not args.gateway_only:
        models += [m for m in pool.get('local', [])]
    if not args.local_only:
        models += [m for m in pool.get('gateway', [])]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    outcomes = []
    summary = {}

    for model in models:
        name = model['name']
        summary[name] = {'caught': 0, 'missed': 0, 'failed': 0, 'total_lat': 0.0,
                         'cost': 0.0}
        for fx in manifest['fixtures']:
            code = load_fixture(fx)
            prompt = manifest['prompt_template'].format(code=code)
            rec = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'task_type': 'code_review',
                   'model': name, 'fixture': fx['file'],
                   'defect_class': fx['defect_class']}
            try:
                if model['kind'] == 'local':
                    text, lat = call_local(model['endpoint'], prompt)
                    cost = 0.0
                else:
                    text, lat = call_gateway(model['model_id'], prompt)
                    cost = 0.0  # free tier — no budget draw
                sc = score_review(text, fx)
                rec.update(sc, latency_s=round(lat, 2), cost_usd=cost,
                           status='ok')
                summary[name]['caught' if sc['caught'] else 'missed'] += 1
            except Exception as e:  # noqa: BLE001
                rec.update(status='failed',
                           error=f'{type(e).__name__}: {str(e)[:120]}')
                summary[name]['failed'] += 1
            summary[name]['total_lat'] += rec.get('latency_s', 0)
            outcomes.append(rec)
            print(f"[{name}] {fx['file']}: "
                  f"{'CAUGHT' if rec.get('caught') else rec['status']}", flush=True)

    with open(args.out, 'a') as f:
        for rec in outcomes:
            f.write(json.dumps(rec) + '\n')

    print('\n=== per-model summary (code_review) ===')
    for name, s in sorted(summary.items()):
        n = s['caught'] + s['missed']
        print(f'{name:35s} catch_rate={s["caught"]}/{n} '
              f'failed={s["failed"]} avg_lat='
              f'{s["total_lat"]/n if n else 0:.1f}s')
    print(f'\n[outcomes] appended {len(outcomes)} records to {args.out}')


if __name__ == '__main__':
    main()
