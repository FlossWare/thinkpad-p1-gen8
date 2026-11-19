# NPUW Tuning Experiments — Qwen3-0.6B INT4 on Intel NPU (ThinkPad P1 Gen 8)

Date: 2026-11-19. Methodology: `benchmarks/benchmark.py`, 3–5 repeats per config,
median reported, one uncounted warm-up generation per process, TTFT measured via
streamer callback (this build has no `get_perf_metrics`). NPU execution confirmed via
`/proc/self/maps` for every row. Raw results: `benchmarks/results/results.jsonl`.

## Baseline correction

The previous session's "13 tok/s" divided total wall time by token count, folding TTFT
and process overhead into decode. With per-token streamer timing the true baseline
decode is **32.9 tok/s**. All comparisons below use the corrected methodology.

## Results

| # | Config (delta vs baseline) | Compile (cold) | Warm load | TTFT | Decode tok/s | Prefill tok/s | Peak RSS |
|---|---|---|---|---|---|---|---|
| 0 | baseline: mpl=256, FAST_COMPILE | 6.2 s | 1.0 s | 266 ms | 32.9 | 18.8 (5-tok prompt) | 1216 MB |
| 1 | + `GENERATE_HINT=BEST_PERF` | 19.8 s | 1.1 s | 262 ms | **41.1** | 19.1 | 1743 MB |
| 2 | `MAX_PROMPT_LEN=64` only | 5.9 s | — | 172 ms | 30.7 | 29.0 | ~1216 MB |
| 3 | mpl=64 + BEST_PERF | 19.8 s | 1.1 s | **164 ms** | **41.1** | 30.4 | 1248 MB |
| 4 | (#3) sustained 128-token gen | (cached) | 1.1 s | 164 ms | 41.1 | 30.7 | 1248 MB |
| 5 | mpl=256 + BEST_PERF, 23-tok prompt | (cached) | 1.1 s | 293 ms | 37.1 | 78.4 | — |
| 6 | mpl=512 + chunked prefill (chunk=128, DYNAMIC), 349-tok prompt | 21.3 s | — | 643 ms | 41.2 | 476 | 1803 MB |
| 7 | mpl=512 + static prefill, same 349-tok prompt | 21.1 s | — | **457 ms** | 41.1 | **691** | 1814 MB |

## Findings

1. **`GENERATE_HINT=BEST_PERF` is the single biggest decode win: 32.9 → 41.1 tok/s
   (+25%)**, stable across 32- and 128-token generations and warm/cold caches.
   Mechanism (source-verified): replaces `NPUW_UNFOLD_IREQS=YES` with
   `NPUW_ONLINE_PIPELINE=NONE` in the decode graph config. Costs: cold compile
   6.2 s → 19.8 s (one-time, then cached at ~1.1 s warm load), peak RSS
   +~500 MB. Only valid with the default generate config (throws otherwise).
2. **`MAX_PROMPT_LEN=64` cuts TTFT 266 → 164 ms (−38%)** for short prompts. The static
   prefill graph pads to the bucket, so the bucket size dominates TTFT; prompt length
   itself is a minor term (5→23 tokens only added ~27 ms at mpl=256). `MAX_PROMPT_LEN`
   is aligned up to multiples of 64 by the plugin, so 64 is the floor. Prompts longer
   than the bucket are rejected (`input_ids.get_size() <= m_max_prompt_len` check in
   `pipeline_stateful.cpp:360`) — size the bucket to your workload.
3. **Chunked prefill (`PREFILL_HINT=DYNAMIC` + `NPUW_LLM_PREFILL_CHUNK_SIZE`) works but
   is slower than static** at this model size: 476 vs 691 tok/s prefill on a 349-token
   prompt. Per-chunk overhead dominates any dynamism benefit for 0.6B. Not recommended.
4. `MIN_RESPONSE_LEN` is aligned to 64 by the plugin; the default 64 already sustains
   128-token generations without measurable degradation (row 4), so no change needed.
5. Prefill throughput scales with prompt length (18.8 tok/s @ 5 tokens → 691 tok/s @
   349 tokens): fixed per-inference overhead, not per-token prefill cost.

## Recommended configuration

```python
genai.LLMPipeline(
    model_dir, device='NPU',
    MAX_PROMPT_LEN=64,       # 256 if prompts can be long; TTFT scales with bucket
    MIN_RESPONSE_LEN=64,
    GENERATE_HINT='BEST_PERF',
    CACHE_DIR='<persistent cache dir>',
)
```

Delivered as `run-npuw.sh`. Trade-off to remember: BEST_PERF's 19.8 s cold compile is
paid once per (model, config, compiler) cache key; with a persistent `CACHE_DIR` warm
load is ~1.1 s.

## Failures recorded

- Prompt exceeding `MAX_PROMPT_LEN` → `RuntimeError: Check 'data->input_ids.get_size()
  <= m_max_prompt_len' failed at src/cpp/src/llm/pipeline_stateful.cpp:360` (reproduced
  with 700-token prompt at mpl=512). Serving layer must enforce the bucket limit.
- `GENERATE_HINT` with a custom generate config → `GENERATE_HINT only works with
  default generate config!` (source-verified, not reproduced here).
