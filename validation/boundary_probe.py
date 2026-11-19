#!/usr/bin/env python3
"""NPU-machine boundary probes for issue #13 (early-stop on near-limit prompts).

Config identical to the validation battery: mpl=256, mrl=64, BEST_PERF,
greedy. Builds exact-token-length prompts from a filler text and compares
CPU vs NPUW generated token counts for prompt/max_new combinations around
the suspected prompt+generated = 320 (= MAX_PROMPT_LEN + MIN_RESPONSE_LEN)
boundary. Prompts are passed as TokenizedInputs (no chat template), so
prompt_tokens is the exact model input length.

Run on the NPU machine (environment from scripts/env.sh, NPU_MODEL_DIR set):
    python3 validation/boundary_probe.py [OUT_JSON]
OUT_JSON defaults to validation/results/boundary-probes.json. The committed
result (2026-11-19) was produced by this script with the output path
hard-coded to a scratch location; the probe logic is otherwise unchanged.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

if __name__ != '__main__':  # GenAI forkserver re-import guard
    raise SystemExit(0)

import numpy as np
import openvino as ov
import openvino_genai as genai

FILLER = ('The quick brown fox jumps over the lazy dog while the sun sets '
          'behind the mountains and rivers flow gently through the valley. ')

CASES = [
    (5, 128), (5, 190), (5, 200),
    (129, 128), (191, 128), (192, 128), (193, 128), (200, 128),
    (193, 64), (250, 64), (256, 64),
]


def ids_to_inputs(ids):
    arr = np.array([ids], dtype=np.int64)
    return genai.TokenizedInputs(input_ids=ov.Tensor(arr),
                                 attention_mask=ov.Tensor(np.ones_like(arr)))


def gen(pipe, ids, max_new):
    cfg = genai.GenerationConfig(max_new_tokens=max_new, do_sample=False, logprobs=1)
    res = pipe.generate(ids_to_inputs(ids), cfg)
    return list(res.tokens[0])


def main():
    model_dir = os.environ['NPU_MODEL_DIR']
    cache = os.environ.get('NPUW_CACHE_DIR')

    cpu = genai.LLMPipeline(model_dir, device='CPU')
    npu = genai.LLMPipeline(model_dir, device='NPU',
                            MAX_PROMPT_LEN=256, MIN_RESPONSE_LEN=64,
                            GENERATE_HINT='BEST_PERF', CACHE_DIR=cache)
    tok = cpu.get_tokenizer()

    filler_ids = list(np.asarray(tok.encode(FILLER * 40).input_ids.data).ravel())
    print(f'filler pool: {len(filler_ids)} tokens', flush=True)

    results = []
    for n_prompt, max_new in CASES:
        if n_prompt > 256:
            print(f'[{n_prompt},{max_new}] skipped: prompt exceeds mpl=256', flush=True)
            continue
        ids = filler_ids[:n_prompt]
        try:
            c = gen(cpu, ids, max_new)
        except Exception as e:  # noqa: BLE001
            c = None
            print(f'[{n_prompt},{max_new}] CPU error: {e}', flush=True)
        try:
            n = gen(npu, ids, max_new)
        except Exception as e:  # noqa: BLE001
            n = None
            print(f'[{n_prompt},{max_new}] NPU error: {e}', flush=True)
        row = {
            'prompt_tokens': n_prompt, 'max_new': max_new,
            'cpu_len': len(c) if c is not None else None,
            'npu_len': len(n) if n is not None else None,
        }
        if c and n:
            k = min(len(c), len(n))
            row['prefix_match'] = c[:k] == n[:k]
            row['sum_prompt_npu'] = n_prompt + len(n)
        results.append(row)
        print(f"[{n_prompt},{max_new}] cpu={row['cpu_len']} npu={row['npu_len']} "
              f"prefix={row.get('prefix_match')} prompt+npu={row.get('sum_prompt_npu')}",
              flush=True)

    out = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else
           pathlib.Path(__file__).resolve().parent / 'results' / 'boundary-probes.json')
    with open(out, 'w') as f:
        json.dump({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'config': 'mpl=256 mrl=64 BEST_PERF greedy',
                   'cases': results}, f, indent=1)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()
