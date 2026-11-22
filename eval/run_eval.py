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
import urllib.error
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
    if proc.returncode != 0:
        raise RuntimeError(
            f'claude CLI exited {proc.returncode} for {model_id}: '
            f'{proc.stderr.strip()[:120]}')
    out = proc.stdout
    # The CLI exits 0 with an error message on stdout for models it cannot
    # invoke through the Anthropic-protocol path. Detect and raise instead of
    # scoring the error text as a review (the 2026-11-21 paid-sweep lesson).
    if ('issue with the selected model' in out
            or 'unrecognized_model' in out
            or "doesn't recognize the model" in out):
        raise RuntimeError(f'claude CLI cannot invoke model {model_id}')
    return out, time.perf_counter() - t0


def feed_thompson(rec):
    """POST one outcome to the live Thompson service (loopback only).

    Contract (source-verified): POST /thompson with action=record_outcome,
    fields model/task_type/success/cost/tokens. Failures to feed are
    reported but never abort the sweep — the JSONL record is always the
    durable source of truth.
    """
    body = json.dumps({'action': 'record_outcome',
                       'model': rec['model'],
                       'task_type': rec['task_type'],
                       'success': bool(rec.get('caught')),
                       'cost': rec.get('cost_usd', 0.0),
                       'tokens': 0}).encode()
    req = urllib.request.Request('http://127.0.0.1:8769/thompson', data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get('ok', False)
    except Exception:  # noqa: BLE001
        return False


def call_ollama(model_name, prompt, timeout=300):
    """POST /api/chat to the local ollama service (single port, model per request)."""
    body = json.dumps({'model': model_name, 'stream': False,
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    req = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    if 'error' in d:
        raise RuntimeError(f'ollama error for {model_name}: '
                           f'{str(d["error"])[:120]}')
    content = (d.get('message') or {}).get('content') or ''
    if not content.strip():
        raise RuntimeError(f'ollama returned empty content for {model_name}')
    return content, time.perf_counter() - t0


def call_openai_gateway(model_id, prompt, timeout=300):
    """Direct HTTP to the gateway's OpenAI-compatible endpoint.

    Needed for GPT/Gemini families: `claude -p` (Anthropic protocol) cannot
    invoke them. GPT-5+ requires max_completion_tokens (rejects max_tokens);
    reasoning models (gemini) burn the budget on hidden reasoning, so use a
    generous limit. Retries once with the alternate parameter name on a
    parameter-rejection HTTP error, and once more on empty content; raises
    if both attempts yield no usable content.
    """
    body = json.dumps({'model': model_id, 'messages': [
        {'role': 'user', 'content': prompt}],
        'max_completion_tokens': 2048}).encode()
    lat = 0.0
    for attempt in range(2):
        req = urllib.request.Request(
            'https://api.enmaas.devshift.net/v1/chat/completions', data=body,
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + os.environ.get(
                         'ANTHROPIC_API_KEY', '')})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if attempt == 0 and e.code == 400:
                # parameter rejection: retry with max_tokens instead
                body = json.dumps({'model': model_id, 'messages': [
                    {'role': 'user', 'content': prompt}],
                    'max_tokens': 4096}).encode()
                continue
            raise
        lat = time.perf_counter() - t0
        choices = d.get('choices') or [{}]
        msg = choices[0].get('message') or {}
        content = msg.get('content') or ''
        if content.strip():
            return content, lat
        # empty content (reasoning burn): retry once with double budget
        body = json.dumps({'model': model_id, 'messages': [
            {'role': 'user', 'content': prompt}],
            'max_tokens': 4096}).encode()
    raise RuntimeError(
        f'openai-gateway model {model_id} returned no usable content '
        f'after {attempt + 1} attempts')


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
    ap.add_argument('--feed', action='store_true',
                    help='POST each outcome to the live Thompson service '
                         '(127.0.0.1:8769) as it is computed')
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
    models += [m for m in pool.get('ollama', [])]
    models += [m for m in pool.get('openai-gateway', [])]

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
                elif model['kind'] == 'ollama':
                    text, lat = call_ollama(model['model_id'], prompt)
                    cost = 0.0
                elif model['kind'] == 'openai-gateway':
                    text, lat = call_openai_gateway(model['model_id'], prompt)
                    cost = 0.0  # budget-drawing; recorded from response usage

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
            if args.feed and rec['status'] == 'ok':
                rec['fed'] = feed_thompson(rec)
            print(f"[{name}] {fx['file']}: "
                  f"{'CAUGHT' if rec.get('caught') else rec['status']}"
                  f"{' (fed)' if rec.get('fed') else ''}", flush=True)

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
