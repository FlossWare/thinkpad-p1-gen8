#!/usr/bin/env python3
"""Repeatable NPUW performance benchmark for Qwen3-0.6B INT4 on Intel NPU.

Measures, per configuration:
  - load+compile time (cold: fresh CACHE_DIR, warm: reused CACHE_DIR)
  - time to first token (TTFT) via streamer timestamps
  - prefill throughput (prompt tokens / TTFT)
  - decode tokens/second (tokens after the first / their wall time)
  - peak RSS
  - NPU execution evidence (/proc/self/maps)

Usage:
  benchmark.py [--label NAME] [--repeats N] [--max-new N] [--prompt STR]
               [--cold]            # use a throwaway cache dir (cold compile)
               [--set KEY=VALUE]   # pipeline property override, repeatable
               [--out FILE.json]

Results are appended to the JSON output file (or stdout as JSON lines).
All config changes are explicit; nothing is mutated on disk except the
cache dir given via CACHE_DIR/--cold.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time

# GenAI scripts MUST guard against forkserver re-import of __main__.
if __name__ != '__main__':  # pragma: no cover
    raise SystemExit(0)  # imported by fork worker: do nothing

import openvino_genai as genai

PROMPTS = {
    'short': 'The capital of France is',
    'medium': 'Explain the difference between a neural processing unit and a graphics processing unit in the context of running large language models.',
}


def parse_set(items: list[str]) -> dict:
    out: dict = {}
    for it in items or []:
        k, _, v = it.partition('=')
        out[k.strip()] = v.strip()
    return out


class TimingStreamer:
    """Records wall-clock time of every generated token chunk."""

    def __init__(self) -> None:
        self.t_start: float | None = None
        self.token_times: list[float] = []

    def __call__(self, chunk: str) -> int | None:
        now = time.perf_counter()
        if self.t_start is None:
            self.t_start = now
        self.token_times.append(now)
        return None  # continue generation


def npu_evidence() -> list[str]:
    try:
        with open('/proc/self/maps') as f:
            mapped = {line.split()[-1] for line in f if 'npu' in line.lower() or 'ze_' in line.lower()}
    except OSError:
        return []
    keys = ('libopenvino_intel_npu', 'libnpu', 'ze_intel_npu')
    return sorted(x for x in mapped if any(k in x for k in keys))


def run_benchmark(args: argparse.Namespace) -> dict:
    ai_root = os.environ.get('FLOSSWARE_AI_ROOT', os.path.expanduser('~/AI'))
    model_dir = os.environ.get('NPU_MODEL_DIR', f'{ai_root}/models/openvino/qwen3-0.6b-int4')
    cache_dir = os.environ.get('NPUW_CACHE_DIR', f'{ai_root}/npuw-cache')

    if args.cold:
        cache_dir = tempfile.mkdtemp(prefix='npuw-bench-cold-')

    props: dict = {
        'MAX_PROMPT_LEN': args.max_prompt_len,
        'MIN_RESPONSE_LEN': args.min_response_len,
        'CACHE_DIR': cache_dir,
    }
    props.update(parse_set(args.set))

    prompt = PROMPTS.get(args.prompt, args.prompt)

    result: dict = {
        'label': args.label,
        'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {k: str(v) for k, v in props.items()},
        'cold_cache': args.cold,
        'max_new_tokens': args.max_new,
        'prompt_kind': args.prompt,
        'prompt_chars': len(prompt),
    }

    print(f"=== {args.label} | cold={args.cold} | extra={parse_set(args.set)}", flush=True)
    print(f'genai version: {genai.get_version()}', flush=True)

    t0 = time.perf_counter()
    pipe = genai.LLMPipeline(model_dir, device='NPU', **props)
    result['load_compile_s'] = round(time.perf_counter() - t0, 2)
    print(f"[load+compile] {result['load_compile_s']}s (cache: {cache_dir})", flush=True)

    # Tokenize once to know the prompt token count (prefill denominator).
    n_prompt_tokens = None
    try:
        tok = pipe.get_tokenizer()
        ids = tok.encode(prompt).input_ids
        n_prompt_tokens = int(ids.get_size())  # batch=1, so element count == token count
    except Exception as e:  # noqa: BLE001
        print(f'[warn] token count unavailable: {e}', flush=True)
    result['prompt_tokens'] = n_prompt_tokens

    # Uncounted warm-up: first generate pays lazy graph init / NPU clock ramp.
    pipe.generate(prompt, max_new_tokens=2, do_sample=False)

    runs = []
    for i in range(args.repeats):
        streamer = TimingStreamer()
        t_gen0 = time.perf_counter()
        out = pipe.generate(prompt, max_new_tokens=args.max_new,
                            streamer=streamer, do_sample=False)
        t_gen = time.perf_counter() - t_gen0
        tt = streamer.token_times
        n_tok = len(tt)
        ttft = (tt[0] - t_gen0) if tt else None
        decode_t = (tt[-1] - tt[0]) if n_tok > 1 else None
        run = {
            'run': i,
            'wall_s': round(t_gen, 3),
            'streamer_chunks': n_tok,
            'ttft_s': round(ttft, 4) if ttft is not None else None,
            'decode_s': round(decode_t, 4) if decode_t is not None else None,
            'decode_tok_s': round((n_tok - 1) / decode_t, 2) if decode_t and n_tok > 1 else None,
            'prefill_tok_s': round(n_prompt_tokens / ttft, 1) if (ttft and n_prompt_tokens) else None,
        }
        runs.append(run)
        print(f"[run {i}] wall={run['wall_s']}s ttft={run['ttft_s']}s "
              f"decode={run['decode_tok_s']} tok/s prefill={run['prefill_tok_s']} tok/s", flush=True)
        if i == 0:
            result['output_preview'] = out[:160]

    result['runs'] = runs

    def med(key: str):
        vals = [r[key] for r in runs if r.get(key) is not None]
        vals.sort()
        return round(vals[len(vals) // 2], 4) if vals else None

    result['median'] = {
        'ttft_s': med('ttft_s'),
        'decode_tok_s': med('decode_tok_s'),
        'prefill_tok_s': med('prefill_tok_s'),
        'wall_s': med('wall_s'),
    }
    result['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    ev = npu_evidence()
    result['npu_evidence'] = ev[:6]
    result['npu_confirmed'] = any('libopenvino_intel_npu_plugin' in x for x in ev)
    print(f"[median] {result['median']}", flush=True)
    print(f"[rss] {result['peak_rss_mb']} MB | npu_confirmed={result['npu_confirmed']}", flush=True)

    if args.cold:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='unnamed')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-new', type=int, default=32)
    p.add_argument('--max-prompt-len', type=int, default=256)
    p.add_argument('--min-response-len', type=int, default=64)
    p.add_argument('--prompt', default='short', help='short|medium|<literal text>')
    p.add_argument('--cold', action='store_true', help='fresh throwaway cache dir')
    p.add_argument('--set', action='append', help='pipeline property KEY=VALUE, repeatable')
    p.add_argument('--out', default=None, help='append result JSON line to this file')
    args = p.parse_args()

    res = run_benchmark(args)

    line = json.dumps(res)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(line + '\n')
    else:
        print(line)


if __name__ == '__main__':
    main()
