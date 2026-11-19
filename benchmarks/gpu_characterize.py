#!/usr/bin/env python3
"""M3 phase-1: measured GPU (Arc 140T) characterization of Qwen3-0.6B INT4.

Token-ID accounting identical to the NPU benchmarks (generated tokens from
EncodedResults.tokens; interval = first streamer callback -> generate() return).
"""
import json
import os
import time


def main():
    import numpy as np
    import openvino as ov
    import openvino_genai as genai

    model_dir = os.environ['NPU_MODEL_DIR']
    t0 = time.perf_counter()
    pipe = genai.LLMPipeline(model_dir, device='GPU')
    load_s = time.perf_counter() - t0
    print(f'[load] {load_s:.1f}s', flush=True)
    tok = pipe.get_tokenizer()

    FILLER = ('The quick brown fox jumps over the lazy dog while the sun sets '
              'behind the mountains and rivers flow gently through the valley. ')
    prompts = {
        'short': 'The capital of France is',
        'medium': 'Explain the difference between a neural processing unit '
                  'and a graphics processing unit in the context of running '
                  'large language models.',
        'long-289t': FILLER * 12,
    }

    class ST:
        def __init__(self):
            self.times = []

        def __call__(self, chunk):
            self.times.append(time.perf_counter())
            return None

    results = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'device': 'GPU (Intel(R) Arc(TM) Graphics iGPU)',
               'model_dir': model_dir,
               'load_s': round(load_s, 2), 'cases': []}
    pipe.generate('Hello', max_new_tokens=2, do_sample=False)  # warm-up

    for name, prompt in prompts.items():
        ids = list(np.asarray(tok.encode(prompt).input_ids.data).ravel())
        arr = np.array([ids], dtype=np.int64)
        inputs = genai.TokenizedInputs(
            input_ids=ov.Tensor(arr), attention_mask=ov.Tensor(np.ones_like(arr)))
        for max_new in (32, 128):
            runs = []
            for i in range(5):
                st = ST()
                cfg = genai.GenerationConfig(max_new_tokens=max_new,
                                             do_sample=False)
                t0 = time.perf_counter()
                res = pipe.generate(inputs, cfg, streamer=st)
                wall = time.perf_counter() - t0
                n = len(res.tokens[0])
                ttft = st.times[0] - t0 if st.times else None
                dec = ((n - 1) / (wall - ttft)) if (st.times and n > 1
                                                    and wall > ttft) else None
                runs.append({'wall_s': round(wall, 3),
                             'ttft_s': round(ttft, 4) if ttft else None,
                             'decode_tok_s': round(dec, 1) if dec else None,
                             'generated': n})
            med = lambda k: sorted(r[k] for r in runs)[2]  # noqa: E731
            case = {'case': f'{name}@{max_new}', 'prompt_tokens': len(ids),
                    'median_ttft_s': med('ttft_s'),
                    'median_decode_tok_s': med('decode_tok_s'),
                    'median_wall_s': med('wall_s'), 'runs': runs}
            results['cases'].append(case)
            print(f"[{case['case']}] ttft={case['median_ttft_s']*1000:.0f}ms "
                  f"decode={case['median_decode_tok_s']} tok/s "
                  f"wall={case['median_wall_s']}s gen={runs[0]['generated']}",
                  flush=True)

    model_name = os.path.basename(model_dir.rstrip('/'))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'gpu-{model_name}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] wrote {out}', flush=True)


if __name__ == '__main__':
    main()
