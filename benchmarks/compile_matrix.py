#!/usr/bin/env python3
"""M2 compile matrix: compile-only cells for a model across NPUW configs.

For each (MAX_PROMPT_LEN, GENERATE_HINT) cell: fresh cache dir, load the
pipeline, generate nothing, record compiled/cold_compile_s/cache_size_mb.
Appends JSONL records to benchmarks/results/npu-large-models.jsonl.

Usage (from repo root, scripts/env.sh sourced):
  python3 benchmarks/compile_matrix.py [--model-dir DIR] [--mpl 256 1024]
      [--hints FAST_COMPILE BEST_PERF] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--mpl', type=int, nargs='+', default=[256, 1024])
    ap.add_argument('--hints', nargs='+', default=['FAST_COMPILE', 'BEST_PERF'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--timeout-s', type=float, default=600,
                    help='compile bound; longer compiles are recorded as '
                         'timeouts (per the M2 spec)')
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'npu-large-models.jsonl')
    device = os.environ.get('M2_DEVICE', 'NPU')

    import openvino_genai as genai

    for mpl in args.mpl:
        for hint in args.hints:
            cache = tempfile.mkdtemp(prefix=f'm2-compile-{mpl}-{hint}-')
            rec = {
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'kind': 'compile_cell', 'model_dir': args.model_dir,
                'device': device, 'mpl': mpl, 'generate_hint': hint,
            }
            t0 = time.perf_counter()
            try:
                # The compile runs in this process; a hang would block the
                # whole run, so the timeout is enforced by the caller (e.g.
                # `timeout 600 python3 ...`). Within the process we only
                # measure.
                pipe = genai.LLMPipeline(
                    args.model_dir, device=device,
                    MAX_PROMPT_LEN=mpl, MIN_RESPONSE_LEN=64,
                    GENERATE_HINT=hint, CACHE_DIR=cache)
                rec['compiled'] = True
                rec['cold_compile_s'] = round(time.perf_counter() - t0, 2)
                del pipe
            except Exception as e:  # noqa: BLE001
                rec['compiled'] = False
                rec['error_class'] = type(e).__name__
                rec['error'] = str(e)[:300]
            cache_mb = 0.0
            for root, _dirs, files in os.walk(cache):
                for f in files:
                    try:
                        cache_mb += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            rec['cache_size_mb'] = round(cache_mb / 1e6, 1)
            shutil.rmtree(cache, ignore_errors=True)
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"[{mpl}/{hint}] compiled={rec['compiled']} "
                  f"compile={rec.get('cold_compile_s')}s "
                  f"cache={rec['cache_size_mb']}MB "
                  f"{rec.get('error_class', '')}", flush=True)


if __name__ == '__main__':
    main()
