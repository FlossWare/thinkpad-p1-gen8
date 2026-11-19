#!/usr/bin/env python3
"""M2/M3 correctness sanity: CPU vs GPU greedy token agreement.

Same battery subset and caveats as validation/validate.py (which covers the
NPUW path): short/medium/near-limit/repeated-token prompts, 32-token greedy
generations, token-sequence comparison. GPU runs the plain stateful path
(no NPUW properties).

Usage: validate_gpu.py [--model-dir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-new', type=int, default=32)
    ap.add_argument('--f32', action='store_true',
                    help='force f32 inference + KV-cache precision on the GPU '
                         '(exact-match mode; default is the iGPU f16 path)')
    args = ap.parse_args()

    import numpy as np
    import openvino as ov
    import openvino_genai as genai

    model_dir = args.model_dir or os.environ['NPU_MODEL_DIR']
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results',
        f'gpu-correctness-{os.path.basename(model_dir.rstrip("/"))}.json')

    FILLER = ('The quick brown fox jumps over the lazy dog while the sun sets '
              'behind the mountains and rivers flow gently through the valley. ')
    prompts = {
        'short-geography': 'The capital of France is',
        'short-code': 'def fibonacci(n):\n    ',
        'repeated-token': 'the ' * 16,
        'number-pattern': ' '.join(str(i) for i in range(1, 33)),
        'medium-nlp': ('Explain the difference between a neural processing unit '
                       'and a graphics processing unit in the context of '
                       'running large language models.'),
        'near-limit-193t': FILLER * 8,
    }

    t0 = time.perf_counter()
    cpu = genai.LLMPipeline(model_dir, device='CPU')
    if args.f32:
        gpu = genai.LLMPipeline(model_dir, device='GPU',
                                **{'INFERENCE_PRECISION_HINT': 'f32',
                                   'KV_CACHE_PRECISION': 'f32'})
    else:
        gpu = genai.LLMPipeline(model_dir, device='GPU')
    print(f'[load] cpu+gpu ready in {time.perf_counter()-t0:.1f}s', flush=True)
    tok = cpu.get_tokenizer()

    def gen(pipe, ids):
        arr = np.array([ids], dtype=np.int64)
        inputs = genai.TokenizedInputs(
            input_ids=ov.Tensor(arr), attention_mask=ov.Tensor(np.ones_like(arr)))
        cfg = genai.GenerationConfig(max_new_tokens=args.max_new, do_sample=False)
        return list(pipe.generate(inputs, cfg).tokens[0])

    results = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'model_dir': model_dir, 'max_new': args.max_new,
               'note': 'GPU plain stateful path; same caveats as validate.py',
               'tests': []}
    for name, prompt in prompts.items():
        ids = list(np.asarray(tok.encode(prompt).input_ids.data).ravel())
        c = gen(cpu, ids)
        g = gen(gpu, ids)
        n = min(len(c), len(g))
        matches = sum(1 for i in range(n) if c[i] == g[i])
        first_div = next((i for i in range(n) if c[i] != g[i]), None)
        entry = {'test': name, 'prompt_tokens': len(ids),
                 'cpu_len': len(c), 'gpu_len': len(g),
                 'match_pct': round(100.0 * matches / n, 1) if n else None,
                 'first_divergence': first_div,
                 'identical': len(c) == len(g) and first_div is None,
                 'cpu_tokens': c, 'gpu_tokens': g}
        results['tests'].append(entry)
        print(f"[{name}] cpu={len(c)} gpu={len(g)} match={entry['match_pct']}% "
              f"first_div={first_div}", flush=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
