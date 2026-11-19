#!/usr/bin/env python3
"""CPU vs NPUW correctness validation for Qwen3-0.6B INT4 on Intel NPU.

Compares the same GenAI build running the same model on CPU (reference) and
NPU (NPUW, tuned config). For each test case captures:
  - greedy token IDs per step (argmax decisions)
  - sequence sum of top-1 logprobs (scores[0], logprobs=1)
  - per-step top-1 logprobs via incremental single-token generation
  - KV-cache variable shapes/dtypes from the original IR (documentation)

Reports token agreement, first divergence, per-step logprob deltas
(abs/mean/max/relative), argmax flips, and context-boundary behavior.

Full per-step logits are NOT exposed by this GenAI build (no logit_processor
hook; scores is a sequence sum under greedy). The manual static-blob path
(~/AI/ovfix-static-blob) remains the source of direct logit-level comparison
(cosine similarity of the last-position logit vector 0.9916 vs CPU, measured
2026-11-18). The 'cosine' reported by THIS script is different: see
compare_logprobs().

Usage: validate.py [--out results.json] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

if __name__ != '__main__':  # GenAI forkserver re-import guard
    raise SystemExit(0)

import openvino as ov
import openvino_genai as genai

NPU_CONFIG = dict(MAX_PROMPT_LEN=256, MIN_RESPONSE_LEN=64,
                  GENERATE_HINT='BEST_PERF')


def build_prompts(tokenizer) -> dict[str, str]:
    filler = ('The quick brown fox jumps over the lazy dog while the sun sets '
              'behind the mountains and rivers flow gently through the valley. ')
    return {
        'short-geography': 'The capital of France is',
        'short-code': 'def fibonacci(n):\n    ',
        'repeated-token': 'the ' * 16,
        'number-pattern': ' '.join(str(i) for i in range(1, 33)),
        'medium-nlp': ('Explain the difference between a neural processing unit '
                       'and a graphics processing unit in the context of running '
                       'large language models.'),
        'near-limit-250t': filler * 8,   # ~193 tokens
        'overflow-340t': filler * 12,    # ~349 tokens: must be rejected at mpl=256
    }


def token_count(tokenizer, text: str) -> int:
    return int(tokenizer.encode(text).input_ids.get_size())


def make_pipe(device: str, npu: bool):
    ai_root = os.environ.get('FLOSSWARE_AI_ROOT', os.path.expanduser('~/AI'))
    model_dir = os.environ.get('NPU_MODEL_DIR', f'{ai_root}/models/openvino/qwen3-0.6b-int4')
    if npu:
        return genai.LLMPipeline(model_dir, device=device,
                                 CACHE_DIR=os.environ.get('NPUW_CACHE_DIR'),
                                 **NPU_CONFIG)
    return genai.LLMPipeline(model_dir, device=device)


def gen_greedy(pipe, tokenizer, prompt: str, max_new: int):
    """Continuous greedy generation; returns (token_ids, sum_logprob)."""
    cfg = genai.GenerationConfig(max_new_tokens=max_new, do_sample=False, logprobs=1)
    res = pipe.generate(tokenizer.encode(prompt), cfg)
    return list(res.tokens[0]), float(res.scores[0]) if res.scores else None


def ids_to_inputs(ids: list[int]) -> 'genai.TokenizedInputs':
    """Exact token-ID inputs (no decode/re-encode roundtrip drift)."""
    import numpy as np
    arr = np.array([ids], dtype=np.int64)
    input_ids = ov.Tensor(arr)
    attention_mask = ov.Tensor(np.ones_like(arr))
    return genai.TokenizedInputs(input_ids=input_ids, attention_mask=attention_mask)


def gen_stepwise(pipe, tokenizer, prompt: str, steps: int):
    """Incremental single-token greedy generations.

    Each call re-prefills prompt+generated-so-far and returns that step's
    top-1 logprob. Equivalent to continuous greedy decode for a causal LM;
    divergence between the two modes is itself a state-handling finding.
    """
    import numpy as np
    ids = list(np.asarray(tokenizer.encode(prompt).input_ids.data).ravel())
    out_tokens, out_logprobs = [], []
    for _ in range(steps):
        cfg = genai.GenerationConfig(max_new_tokens=1, do_sample=False, logprobs=1)
        res = pipe.generate(ids_to_inputs(ids), cfg)
        tok = int(res.tokens[0][0])
        lp = float(res.scores[0]) if res.scores else None
        out_tokens.append(tok)
        out_logprobs.append(lp)
        ids.append(tok)
    return out_tokens, out_logprobs


def compare_tokens(ref, test):
    n = min(len(ref), len(test))
    matches = sum(1 for i in range(n) if ref[i] == test[i])
    first_div = next((i for i in range(n) if ref[i] != test[i]), None)
    return {
        'n': n,
        'match_pct': round(100.0 * matches / n, 1) if n else None,
        'first_divergence': first_div,
        'identical': len(ref) == len(test) and first_div is None,
    }


def compare_logprobs(ref, test):
    pairs = [(a, b) for a, b in zip(ref, test) if a is not None and b is not None]
    if not pairs:
        return None
    diffs = [abs(a - b) for a, b in pairs]
    rel = [abs(a - b) / max(abs(a), 1e-6) for a, b in pairs]
    # 'cosine' = cosine similarity between the two vectors of per-step top-1
    # log-probabilities (one element per compared step; each element is
    # log P of the token that device chose at that step). NOT a logit or
    # full-vocabulary cosine. Key name kept for compatibility with
    # validation/results/validation-results.json.
    dot = sum(a * b for a, b in pairs)
    na = math.sqrt(sum(a * a for a, _ in pairs))
    nb = math.sqrt(sum(b * b for _, b in pairs))
    return {
        'steps_compared': len(pairs),
        'mean_abs_delta': round(sum(diffs) / len(diffs), 5),
        'max_abs_delta': round(max(diffs), 5),
        'mean_rel_delta': round(sum(rel) / len(rel), 5),
        'max_rel_delta': round(max(rel), 5),
        'cosine': round(dot / (na * nb), 5) if na and nb else None,
    }


def kv_cache_report() -> dict:
    """KV-cache variables as declared in the original IR."""
    core = ov.Core()
    model = core.read_model(os.path.join(os.environ['NPU_MODEL_DIR'], 'openvino_model.xml'))
    vars_out = []
    for v in model.get_variables():
        info = v.get_info()
        vars_out.append({'variable_id': str(info.variable_id),
                         'data_type': str(info.data_type),
                         'data_shape': str(info.data_shape)})
    return {
        'n_variables': len(vars_out),
        'example_variables': vars_out[:4],
        'npuw_storage_precision': 'f16 (source-verified: choose_kv_cache_storage_type '
                                   'default, NPUW_LLM_OPTIMIZE_FP8=false)',
        'activation_conversion': 'bf16 -> f16 model-wide (llm_compiled_model.cpp:965)',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='validation-results.json')
    ap.add_argument('--quick', action='store_true', help='smaller battery')
    ap.add_argument('--steps', type=int, default=16, help='per-step logprob depth')
    args = ap.parse_args()

    results = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'tests': [], 'kv_cache': kv_cache_report()}
    print(f"[kv-cache] {results['kv_cache']['n_variables']} variables in IR; "
          f"NPUW storage: {results['kv_cache']['npuw_storage_precision']}", flush=True)

    print('[load] CPU reference pipeline...', flush=True)
    t0 = time.perf_counter()
    cpu = make_pipe('CPU', npu=False)
    print(f'[load] CPU ready in {time.perf_counter()-t0:.1f}s', flush=True)
    print('[load] NPU pipeline...', flush=True)
    t0 = time.perf_counter()
    npu = make_pipe('NPU', npu=True)
    print(f'[load] NPU ready in {time.perf_counter()-t0:.1f}s', flush=True)

    tokenizer = cpu.get_tokenizer()
    prompts = build_prompts(tokenizer)

    # 1. Overflow behavior at the 256 bucket (expected rejection)
    ovf_tokens = token_count(tokenizer, prompts['overflow-340t'])
    entry = {'test': 'overflow-prompt', 'prompt_tokens': ovf_tokens}
    try:
        gen_greedy(npu, tokenizer, prompts['overflow-340t'], 4)
        entry['npu_result'] = 'ACCEPTED (unexpected!)'
    except Exception as e:  # noqa: BLE001
        entry['npu_result'] = f'rejected as expected: {str(e)[:120]}'
    try:
        gen_greedy(cpu, tokenizer, prompts['overflow-340t'], 4)
        entry['cpu_result'] = 'accepted (CPU has no bucket limit)'
    except Exception as e:  # noqa: BLE001
        entry['cpu_result'] = f'rejected: {str(e)[:120]}'
    results['tests'].append(entry)
    print(f"[overflow] {ovf_tokens} tokens -> {entry['npu_result'][:80]}", flush=True)

    # 2. Continuous-generation battery
    battery = [
        ('short-geography', 32), ('short-geography', 64), ('short-geography', 128),
        ('short-code', 32),
        ('repeated-token', 32),
        ('number-pattern', 32),
        ('medium-nlp', 32), ('medium-nlp', 64),
        ('near-limit-250t', 32), ('near-limit-250t', 128),
    ]
    if args.quick:
        battery = [battery[0], battery[3], battery[8]]

    for name, max_new in battery:
        prompt = prompts[name]
        n_tok = token_count(tokenizer, prompt)
        cpu_ids, cpu_lp = gen_greedy(cpu, tokenizer, prompt, max_new)
        npu_ids, npu_lp = gen_greedy(npu, tokenizer, prompt, max_new)
        entry = {
            'test': f'{name}@{max_new}',
            'prompt_tokens': n_tok,
            'cpu_tokens': cpu_ids,
            'npu_tokens': npu_ids,
            'cpu_sum_logprob': cpu_lp,
            'npu_sum_logprob': npu_lp,
            'sum_logprob_delta': round(npu_lp - cpu_lp, 4) if cpu_lp is not None and npu_lp is not None else None,
            'token_comparison': compare_tokens(cpu_ids, npu_ids),
        }
        results['tests'].append(entry)
        tc = entry['token_comparison']
        print(f"[{entry['test']}] prompt={n_tok}t match={tc['match_pct']}% "
              f"first_div={tc['first_divergence']} ΔΣlogP={entry['sum_logprob_delta']}", flush=True)

    # 3. Per-step logprobs (incremental single-token) on selected prompts
    for name in ('short-geography', 'repeated-token', 'near-limit-250t'):
        prompt = prompts[name]
        _, cpu_lps = gen_stepwise(cpu, tokenizer, prompt, args.steps)
        _, npu_lps = gen_stepwise(npu, tokenizer, prompt, args.steps)
        cmp = compare_logprobs(cpu_lps, npu_lps)
        entry = {
            'test': f'{name}@stepwise{args.steps}',
            'cpu_logprobs': cpu_lps,
            'npu_logprobs': npu_lps,
            'logprob_comparison': cmp,
        }
        results['tests'].append(entry)
        print(f"[{entry['test']}] {cmp}", flush=True)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
