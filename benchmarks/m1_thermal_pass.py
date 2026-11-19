#!/usr/bin/env python3
"""M1 thermal annex: one sustained 16-turn 1024-bucket session with 1 s
thermal-zone and RAPL sampling. Writes benchmarks/results/m1-thermal-1024.json.
Run from the repository root with scripts/env.sh sourced.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import openvino_genai as genai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from multi_turn import (SYSTEM_PROMPT, USER_PROMPTS, ThermalMonitor, gen_turn)  # noqa: E402


def main():
    thermal = ThermalMonitor()
    thermal.start()
    pipe = genai.LLMPipeline(os.environ['NPU_MODEL_DIR'], device='NPU',
                             MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=64,
                             GENERATE_HINT='BEST_PERF',
                             CACHE_DIR=os.environ.get('NPUW_CACHE_DIR'))
    pipe.generate('Hello', max_new_tokens=2, do_sample=False)  # warm-up
    tok = pipe.get_tokenizer()
    history = list(np.asarray(tok.encode(SYSTEM_PROMPT).input_ids.data).ravel())
    users = [list(np.asarray(tok.encode(p).input_ids.data).ravel())
             for p in USER_PROMPTS]
    turns = []
    for t in range(16):
        prompt = history + users[t % len(users)]
        if len(prompt) + 64 > 1024 + 64:
            break
        toks, m = gen_turn(pipe, prompt, 64)
        m['turn'] = t
        turns.append(m)
        history = prompt + toks
        print(f"[turn {t}] prompt={m['prompt_tokens']}t "
              f"ttft={m['ttft_s']} decode={m['decode_tok_s']}", flush=True)
    out = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'config': 'mpl=1024 mrl=64 BEST_PERF, 16-turn sustained session',
           'turns': turns, 'thermal': thermal.stop()}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'm1-thermal-1024.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    z = out['thermal']['per_zone_stats'].get('thermal_zone0')
    r = out['thermal']['per_zone_stats'].get('intel-rapl:0')
    print(f'[thermal] {len(turns)} turns; zone0={z} rapl0={r}', flush=True)
    print(f'[done] wrote {path}', flush=True)


if __name__ == '__main__':
    main()
