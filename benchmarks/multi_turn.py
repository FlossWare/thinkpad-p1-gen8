#!/usr/bin/env python3
"""M1: sustained multi-turn conversation benchmark on the NPU (roadmap M1).

Measures, per MAX_PROMPT_LEN bucket (256, 512, 1024):
  - cold compile time and warm load time (one pipeline instance per bucket)
  - conversation growth: per-turn prompt tokens, TTFT, decode tok/s
    (token-ID accounting: numerator = generated token IDs - 1, interval =
    first streamer callback -> generate() return), turn wall
  - sessions run to 4, 8, and 16 turns; a session stops at the KV-capacity
    boundary (prompt + 64 > mpl + mrl) and records stop_reason, never retried
  - steady-state: 20 sequential short-prompt requests
  - short-prompt baseline: 5 repeats
  - thermal annex: samples readable thermal zones / RAPL energy counters at
    1 s intervals; if nothing is readable, that fact is recorded (no estimates)

Raw output: benchmarks/results/m1-multi-turn.json (single JSON document).

Methodology: docs/roadmap.md shared measurement rules; token accounting per
benchmarks/token_accounting.py.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import threading
import time

import numpy as np
import openvino as ov
import openvino_genai as genai

BUCKETS = [256, 512, 1024]
TURN_TARGETS = [4, 8, 16]
SESSIONS_PER_CELL = 3
REPLY_TOKENS = 64
MRL = 64
STEADY_REQUESTS = 20
SHORT_REPEATS = 5

SYSTEM_PROMPT = ('You are a concise technical assistant. Answer directly and '
                 'stay on topic.')

# User prompts of increasing size (targets ~32, ~64, ~128 tokens; actual counts
# recorded in the results). Cycled for turns beyond 8.
USER_PROMPTS = [
    'What is the difference between latency and throughput?',
    'Explain how a CPU cache hierarchy works, including L1, L2, and L3, and '
    'why locality matters for performance.',
    ('Describe the design of a modern solid-state drive: NAND cells, pages '
     'and blocks, the flash translation layer, wear leveling, garbage '
     'collection, TRIM, and the performance consequences of each for both '
     'sequential and random workloads on a typical consumer laptop.'),
    'Compare interpreted and compiled languages.',
    ('Walk through what happens when a browser loads a web page: DNS '
     'resolution, TCP and TLS handshakes, the HTTP request, HTML parsing, '
     'the DOM, CSS object model, script execution blocking, async loading, '
     'rendering, layout, paint, and compositing.'),
    'Explain consensus in distributed systems.',
    ('Explain the memory hierarchy of a modern computer in depth: registers, '
     'L1/L2/L3 caches with typical sizes and latencies, main memory, NVMe '
     'storage, and network attached storage; include the implications for '
     'algorithm design and the arithmetic of cache misses.'),
    'What is a hash collision and how is it handled?',
]


class ThermalMonitor:
    """Samples readable thermal/power telemetry; records availability."""

    def __init__(self):
        self.samples: list[dict] = []
        self.zones = sorted(glob.glob('/sys/class/thermal/thermal_zone*/temp'))
        self.rapl = sorted(glob.glob('/sys/class/powercap/intel-rapl*/energy_uj'))
        self.available = bool(self.zones or self.rapl)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self, path):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def _run(self):
        while not self._stop.is_set():
            s = {'t': time.time()}
            for z in self.zones:
                v = self._read(z)
                if v is not None:
                    s[z.rsplit('/', 2)[-2]] = v  # milli-C
            for r in self.rapl:
                v = self._read(r)
                if v is not None:
                    s[r.rsplit('/', 2)[-2]] = v  # micro-J
            self.samples.append(s)
            self._stop.wait(1.0)

    def start(self):
        if self.available:
            self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        if self.available and self._t.is_alive():
            self._t.join(timeout=2)
        stats = {}
        keys = [k for k in (self.samples[0] if self.samples else {}) if k != 't']
        for k in keys:
            vals = [s[k] for s in self.samples if k in s]
            stats[k] = {'min': min(vals), 'max': max(vals),
                        'first': vals[0], 'last': vals[-1]}
        return {
            'available': self.available,
            'thermal_zones': [z.rsplit('/', 2)[-2] for z in self.zones],
            'rapl_domains': [r.rsplit('/', 2)[-2] for r in self.rapl],
            'n_samples': len(self.samples),
            'per_zone_stats': stats,
            'note': '' if self.available else
                    'no readable thermal zone or RAPL counter found on this '
                    'machine (values are NOT estimated)',
        }


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class TimingStreamer:
    def __init__(self):
        self.times: list[float] = []

    def __call__(self, chunk):
        self.times.append(time.perf_counter())
        return None


def gen_turn(pipe, ids, max_new):
    """One greedy turn; token-ID accounting per the shared rules."""
    import numpy as _np
    arr = _np.array([ids], dtype=_np.int64)
    inputs = genai.TokenizedInputs(
        input_ids=ov.Tensor(arr), attention_mask=ov.Tensor(_np.ones_like(arr)))
    cfg = genai.GenerationConfig(max_new_tokens=max_new, do_sample=False)
    st = TimingStreamer()
    t0 = time.perf_counter()
    res = pipe.generate(inputs, cfg, streamer=st)
    wall = time.perf_counter() - t0
    toks = list(res.tokens[0])
    n_gen = len(toks)
    ttft = (st.times[0] - t0) if st.times else None
    decode = None
    if st.times and n_gen > 1:
        # interval: first streamer callback -> generate() return
        interval = wall - (st.times[0] - t0)
        decode = (n_gen - 1) / interval if interval > 0 else None
    return toks, {'prompt_tokens': len(ids), 'requested': max_new,
                  'generated': n_gen, 'ttft_s': round(ttft, 4) if ttft else None,
                  'decode_tok_s': round(decode, 2) if decode else None,
                  'wall_s': round(wall, 3)}


def run_bucket(bucket, results, thermal):
    print(f'=== bucket mpl={bucket} ===', flush=True)
    model_dir = os.environ['NPU_MODEL_DIR']
    cache = os.environ.get('NPUW_CACHE_DIR')

    t0 = time.perf_counter()
    pipe = genai.LLMPipeline(model_dir, device='NPU',
                             MAX_PROMPT_LEN=bucket, MIN_RESPONSE_LEN=MRL,
                             GENERATE_HINT='BEST_PERF', CACHE_DIR=cache)
    cold_s = time.perf_counter() - t0
    # second load from cache measures warm load for the same config
    t0 = time.perf_counter()
    pipe2 = genai.LLMPipeline(model_dir, device='NPU',
                              MAX_PROMPT_LEN=bucket, MIN_RESPONSE_LEN=MRL,
                              GENERATE_HINT='BEST_PERF', CACHE_DIR=cache)
    warm_s = time.perf_counter() - t0
    del pipe2
    print(f'[compile] cold={cold_s:.1f}s warm_reload={warm_s:.1f}s', flush=True)

    tok = pipe.get_tokenizer()
    sys_ids = list(np.asarray(tok.encode(SYSTEM_PROMPT).input_ids.data).ravel())
    user_ids = [list(np.asarray(tok.encode(p).input_ids.data).ravel())
                for p in USER_PROMPTS]
    print(f'[script] system={len(sys_ids)}t users={[len(u) for u in user_ids]}',
          flush=True)

    # warm-up (uncounted)
    pipe.generate('Hello', max_new_tokens=2, do_sample=False)

    capacity = bucket + MRL
    bucket_res = {'bucket': bucket, 'mrl': MRL, 'capacity': capacity,
                  'cold_compile_s': round(cold_s, 2),
                  'warm_reload_s': round(warm_s, 2),
                  'script_token_counts': {'system': len(sys_ids),
                                          'users': [len(u) for u in user_ids]},
                  'sessions': [], 'steady': {}, 'short_baseline': {}}

    for target in TURN_TARGETS:
        for sess_i in range(SESSIONS_PER_CELL):
            history = list(sys_ids)
            turns = []
            stop = None
            for turn in range(target):
                u = user_ids[turn % len(user_ids)]
                prompt = history + u
                if len(prompt) + REPLY_TOKENS > capacity:
                    stop = {'reason': 'capacity', 'at_turn': turn,
                            'prompt_tokens': len(prompt)}
                    break
                toks, m = gen_turn(pipe, prompt, REPLY_TOKENS)
                m['turn'] = turn
                turns.append(m)
                history = prompt + toks
                if len(toks) < REPLY_TOKENS:
                    stop = {'reason': 'early_stop', 'at_turn': turn,
                            'generated': len(toks)}
                    break
            bucket_res['sessions'].append(
                {'target_turns': target, 'session': sess_i,
                 'turns': turns, 'stop': stop})
            last = turns[-1] if turns else None
            print(f"[session {target}t/{sess_i}] turns={len(turns)} "
                  f"last_ttft={last['ttft_s'] if last else '-'} "
                  f"last_decode={last['decode_tok_s'] if last else '-'} "
                  f"stop={stop['reason'] if stop else 'completed'}", flush=True)

    # steady-state: 20 sequential short requests
    lat = []
    for i in range(STEADY_REQUESTS):
        toks, m = gen_turn(pipe, sys_ids + user_ids[0], 16)
        lat.append(m)
    bucket_res['steady'] = {
        'n': STEADY_REQUESTS,
        'median_ttft_s': sorted(x['ttft_s'] for x in lat)[STEADY_REQUESTS // 2],
        'median_decode_tok_s': sorted(x['decode_tok_s'] for x in lat)[STEADY_REQUESTS // 2],
        'median_wall_s': sorted(x['wall_s'] for x in lat)[STEADY_REQUESTS // 2],
    }
    print(f"[steady] {bucket_res['steady']}", flush=True)

    # short-prompt baseline, 5 repeats
    base = []
    for _ in range(SHORT_REPEATS):
        toks, m = gen_turn(pipe, sys_ids + user_ids[0], REPLY_TOKENS)
        base.append(m)
    bucket_res['short_baseline'] = {
        'n': SHORT_REPEATS,
        'median_ttft_s': sorted(x['ttft_s'] for x in base)[SHORT_REPEATS // 2],
        'median_decode_tok_s': sorted(x['decode_tok_s'] for x in base)[SHORT_REPEATS // 2],
    }
    results['buckets'].append(bucket_res)


def main():
    model_dir = os.environ['NPU_MODEL_DIR']
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'm1-multi-turn.json')
    results = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'harness': 'benchmarks/multi_turn.py',
        'config': {'model': 'qwen3-0.6b-int4', 'mrl': MRL,
                   'generate_hint': 'BEST_PERF', 'device': 'NPU',
                   'compiler': os.environ.get('IE_NPU_COMPILER_TYPE'),
                   'reply_tokens': REPLY_TOKENS,
                   'buckets': BUCKETS, 'turn_targets': TURN_TARGETS,
                   'sessions_per_cell': SESSIONS_PER_CELL},
        'model_manifest': {
            p: sha256(os.path.join(model_dir, p))
            for p in ('openvino_model.xml', 'openvino_model.bin')
            if os.path.exists(os.path.join(model_dir, p))},
        'buckets': [],
    }

    thermal = ThermalMonitor()
    thermal.start()
    t_start = time.time()
    for bucket in BUCKETS:
        run_bucket(bucket, results, thermal)
    results['wall_total_s'] = round(time.time() - t_start, 1)
    results['thermal'] = thermal.stop()

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] wrote {out_path} (total {results["wall_total_s"]}s)', flush=True)


if __name__ == '__main__':
    main()
