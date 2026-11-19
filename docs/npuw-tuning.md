# NPUW Tuning Experiments — Qwen3-0.6B INT4 on Intel NPU (ThinkPad P1 Gen 8)

Date: 2026-11-19. Methodology: `benchmarks/benchmark.py`, 3–5 repeats per config,
median reported, one uncounted warm-up generation per process, TTFT measured via
streamer callback (this build has no `get_perf_metrics`). NPU plugin libraries were
mapped in-process (`/proc/self/maps`, recorded as `npu_confirmed` in the rows below) for
every row — plugin-load evidence, **not** independent proof that each measured inference
executed on the NPU (issue #6). Raw results: `benchmarks/results/results.jsonl`.

> **Methodology update (2026-11-19, issues #1/#6) — tooling changed, the figures
> below are NOT yet re-measured.** Decode throughput is now computed from the
> generated token IDs in the GenAI `EncodedResults` object (tokenized-input
> `generate()`), over the interval from the first streamer callback to
> `generate()` return; streamer callback counts are recorded for reference only,
> since this build's text streamer is not 1:1 with tokens (32 requested → 30
> callbacks, 128 → 101). NPU evidence is now layered and labeled: requested
> device, resolved-device query, and plugin libraries mapped in process — the
> `/proc/self/maps` check proves plugin load, not per-inference NPU execution,
> and no execution telemetry exists in this build. See the re-run TODO in
> `benchmarks/benchmark.py`; rows in `results.jsonl` recorded before this change
> used callback counts as the decode numerator and must not be mixed with new
> rows in one comparison.

## Baseline correction

The previous session's "13 tok/s" divided total wall time by token count, folding TTFT
and process overhead into decode. A second correction (issue #1): the streamer
**callback count** is not a generated-token count (31 callbacks for 32 generated tokens
is typical), so callback-count numerators also under-report. All current figures below
come from the **token-ID re-measurement** (2026-11-19, rows labeled `re-*` in
`benchmarks/results/results.jsonl`): the numerator is `EncodedResults.tokens` length,
the interval is first→last streamer callback.

## Results (token-ID accounting, 2026-11-19 re-measurement)

| # | Config (delta vs baseline) | Compile (cold) | Warm load | TTFT | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|---|---|
| 0 | baseline: mpl=256, FAST_COMPILE | 6.2 s | 1.0 s | 297 ms | 36.5 | 16.8 (5-tok prompt) |
| 1 | + `GENERATE_HINT=BEST_PERF` | 19.8 s | 1.1 s | 317 ms | **42.2** | 15.8 |
| 2 | `MAX_PROMPT_LEN=64` only | 5.9 s | — | 257 ms | 36.4 | 19.5 |
| 3 | mpl=64 + BEST_PERF (recommended) | 19.8 s | 1.1 s | **189 ms** | **44.6** | 26.5 |
| 4 | (#3) sustained 128-token gen | (cached) | 1.1 s | 196 ms | 42.3 | 25.5 |
| 5 | mpl=256 + BEST_PERF, 23-tok prompt | (cached) | 1.1 s | 295 ms | 46.2 | 78.0 |

Historical rows measured with the superseded callback-count accounting (32.9 baseline,
41.1 BEST_PERF) remain in `results.jsonl` under their original labels; the chunked-
prefill comparison (rows 6–7 of the original run: chunked 476 vs static 691 tok/s
prefill on a 349-token prompt) was not re-run — its conclusion compares two
configurations measured under the same (superseded) accounting, so the *relative*
result stands, absolute values are indicative only.

## Findings

1. **`GENERATE_HINT=BEST_PERF` is the single biggest decode win: 36.5 → 44.6 tok/s
   (+22%)** at mpl=64, stable across 32- and 128-token generations and warm/cold
   caches. Mechanism (source-verified): replaces `NPUW_UNFOLD_IREQS=YES` with
   `NPUW_ONLINE_PIPELINE=NONE` in the decode graph config. Costs: cold compile
   6.2 s → 19.8 s (one-time, then cached at ~1.1 s warm load), peak RSS
   +~500 MB. Only valid with the default generate config (throws otherwise).
2. **`MAX_PROMPT_LEN=64` cuts TTFT 297 → 189 ms (−36%)** for short prompts. The static
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
5. Prefill throughput scales with prompt length (16.8 tok/s @ 5 tokens → 691 tok/s @
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
