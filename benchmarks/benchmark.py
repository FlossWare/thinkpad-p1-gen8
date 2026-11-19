#!/usr/bin/env python3
"""Repeatable NPUW performance benchmark for Qwen3-0.6B INT4 on Intel NPU.

Measures, per configuration:
  - load+compile time (cold: fresh CACHE_DIR, warm: reused CACHE_DIR)
  - time to first token (TTFT) via streamer timestamps
  - prefill throughput (prompt tokens / TTFT)
  - decode tokens/second, counted from the generated token IDs in the GenAI
    result object (EncodedResults.tokens) over the first-callback ->
    generate()-return interval; streamer callbacks are timing boundaries
    only, never token counts (see token_accounting.py, issue #1)
  - peak RSS
  - NPU evidence in layers (issue #6): requested device, resolved-device
    query, plugin libraries mapped in process. Library presence proves the
    NPU plugin is loaded, NOT that each measured inference executed on the
    NPU; no per-inference execution telemetry exists in this build.

TODO(re-runs, issues #1/#6): the rows already in benchmarks/results/
results.jsonl were produced with the OLD accounting (streamer-callback
count as the decode numerator) and the OLD npu_confirmed semantics. Re-run
on the NPU workstation with this script and refresh: baseline rows
(baseline-warm/cold), BEST_PERF rows (e1/e2/e2b/e3/e4/e5/e5b), and the
figures in docs/npuw-tuning.md + README.md (docs owned separately). Do not
mix old and new rows in one comparison.

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

import numpy as np
import openvino as ov
import openvino_genai as genai

import token_accounting

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
    """Records wall-clock time of every streamer text chunk.

    Chunks are NOT 1:1 with generated tokens in this build (they can be
    merged or skipped), so these timestamps are used for timing boundaries
    only; token counts come from the result object (issue #1).
    """

    def __init__(self) -> None:
        self.t_start: float | None = None
        self.token_times: list[float] = []

    def __call__(self, chunk: str) -> int | None:
        now = time.perf_counter()
        if self.t_start is None:
            self.t_start = now
        self.token_times.append(now)
        return None  # continue generation


def npu_mapped_plugin_libs() -> list[str]:
    """NPU plugin/compiler/Level-Zero libraries mapped into this process.

    Reads /proc/self/maps. This establishes only that those libraries are
    LOADED in the process; it does NOT prove that any particular inference
    operation executed on the NPU (issue #6).
    """
    try:
        with open('/proc/self/maps') as f:
            mapped = {line.split()[-1] for line in f if 'npu' in line.lower() or 'ze_' in line.lower()}
    except OSError:
        return []
    keys = ('libopenvino_intel_npu', 'libnpu', 'ze_intel_npu')
    return sorted(x for x in mapped if any(k in x for k in keys))


def npu_device_query() -> dict:
    """Best-effort resolved-device query against a fresh OpenVINO Core.

    Records whichever NPU device properties could be read. This is
    device-resolution evidence (the NPU is enumerable and queryable from
    this process); it is NOT per-inference execution evidence.
    """
    out: dict = {}
    try:
        core = ov.Core()
    except Exception as e:  # noqa: BLE001 - Core() unavailable
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    for key in ('FULL_DEVICE_NAME', 'DEVICE_NAME'):
        try:
            out[key] = str(core.get_property('NPU', key))
        except Exception:  # noqa: BLE001 - property unsupported/renamed
            pass
    return out


def tokenize_prompt(pipe: genai.LLMPipeline, prompt: str) -> tuple[int, object]:
    """Tokenize the prompt into genai.TokenizedInputs for measured runs.

    Verified API facts for this build: pipe.generate() with TokenizedInputs
    returns EncodedResults with .tokens (list of per-sequence token-ID
    lists); with a plain string prompt it returns DecodedResults (no
    tokens). tokenizer.encode(str).input_ids is an ov.Tensor whose .data is
    a numpy view; genai.TokenizedInputs(input_ids=ov.Tensor(...),
    attention_mask=ov.Tensor(...)) is accepted by generate().

    Returns (n_prompt_tokens, tokenized); batch=1, so the element count is
    the token count.
    """
    tok = pipe.get_tokenizer()
    enc = tok.encode(prompt)
    ids = np.array(enc.input_ids.data, dtype=np.int64)
    if getattr(enc, 'attention_mask', None) is not None:
        mask = np.array(enc.attention_mask.data, dtype=np.int64)
    else:
        mask = np.ones(ids.shape, dtype=np.int64)
    tokenized = genai.TokenizedInputs(
        input_ids=ov.Tensor(ids), attention_mask=ov.Tensor(mask))
    return int(ids.size), tokenized


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

    # Tokenize once: the prompt token count (prefill denominator) and the
    # TokenizedInputs needed to get generated token IDs back from generate()
    # (a string-prompt generate returns DecodedResults, which has no tokens).
    n_prompt_tokens = None
    tokenized = None
    try:
        n_prompt_tokens, tokenized = tokenize_prompt(pipe, prompt)
    except Exception as e:  # noqa: BLE001
        print(f'[warn] tokenization unavailable, falling back to string '
              f'prompt (no token-ID accounting): {e}', flush=True)
    result['prompt_tokens'] = n_prompt_tokens

    # Uncounted warm-up: first generate pays lazy graph init / NPU clock ramp.
    pipe.generate(prompt, max_new_tokens=2, do_sample=False)

    runs = []
    for i in range(args.repeats):
        streamer = TimingStreamer()
        cfg = genai.GenerationConfig(max_new_tokens=args.max_new, do_sample=False)
        cfg.streamer = streamer
        t_gen0 = time.perf_counter()
        if tokenized is not None:
            out = pipe.generate(tokenized, cfg)  # EncodedResults: has .tokens
            n_generated = len(out.tokens[0]) if getattr(out, 'tokens', None) else None
        else:
            out = pipe.generate(prompt, max_new_tokens=args.max_new,
                                streamer=streamer, do_sample=False)
            n_generated = None
        t_gen_end = time.perf_counter()
        run = {
            'run': i,
            'requested_tokens': args.max_new,
            'wall_s': round(t_gen_end - t_gen0, 3),
        }
        run.update(token_accounting.measure_run(
            streamer.token_times, t_gen0, t_gen_end, n_generated,
            n_prompt_tokens))
        for note in token_accounting.sanity_check(run, args.max_new):
            print(f'[note] {note}', flush=True)
        runs.append(run)
        print(f"[run {i}] wall={run['wall_s']}s ttft={run['ttft_s']}s "
              f"decode={run['decode_tok_s']} tok/s "
              f"({run['generated_tokens']} generated / {run['streamer_chunks']} callbacks) "
              f"prefill={run['prefill_tok_s']} tok/s", flush=True)
        if i == 0:
            try:
                result['output_preview'] = out.texts[0][:160]
            except (AttributeError, IndexError, TypeError):
                result['output_preview'] = str(out)[:160]

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

    # NPU evidence, in distinct layers (issue #6):
    #   1. requested device: what the pipeline was constructed with
    #   2. resolved device: NPU properties queryable via a fresh ov.Core
    #   3. plugin libraries mapped in process: /proc/self/maps
    # Execution-level evidence: this build of OpenVINO GenAI exposes no
    # per-inference device telemetry (no perf-metrics API, no per-request
    # device accounting), so benchmark runs cannot independently verify that
    # each measured operation executed on the NPU. We record that limitation
    # explicitly instead of presenting library presence as proof.
    libs = npu_mapped_plugin_libs()
    result['device_requested'] = 'NPU'
    result['npu_plugin_libs_mapped'] = libs[:6]
    result['npu_plugin_loaded'] = any('libopenvino_intel_npu_plugin' in x for x in libs)
    result['npu_device_query'] = npu_device_query() or None
    result['npu_execution_evidence'] = None  # unavailable in this build (see above)
    print(f"[median] {result['median']}", flush=True)
    print(f"[rss] {result['peak_rss_mb']} MB", flush=True)
    print(f"[npu] requested=NPU | plugin_loaded={result['npu_plugin_loaded']} | "
          f"device_query={result['npu_device_query']}", flush=True)
    print('[npu] limitation: /proc/self/maps shows the NPU plugin libraries are '
          'mapped in this process; that is plugin-load evidence only. No '
          'per-inference execution telemetry exists in this build, so NPU '
          'execution of the measured operations is not independently verified.',
          flush=True)

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
