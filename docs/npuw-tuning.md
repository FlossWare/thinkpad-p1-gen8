# NPUW Tuning Experiments — Qwen3-0.6B INT4 on Intel NPU (ThinkPad P1 Gen 8)

Date: 2026-11-19. Methodology: `benchmarks/benchmark.py`, 3–5 repeats per config,
median reported, one uncounted warm-up generation per process, TTFT measured via
streamer callback (this build has no `get_perf_metrics`). NPU plugin libraries were
mapped in-process (`/proc/self/maps`; recorded as `npu_plugin_loaded` /
`npu_plugin_libs_mapped` in the `re-*` rows, as `npu_confirmed` in historical rows) for
every row — plugin-load evidence, **not** independent proof that each measured inference
executed on the NPU (issue #6). Raw results: `benchmarks/results/results.jsonl`.

> **Methodology (2026-11-19, issues #1/#6).** Decode throughput is computed from the
> generated token IDs in the GenAI `EncodedResults` object (tokenized-input
> `generate()`), numerator `generated_tokens − 1`, over the interval from the first
> streamer callback to `generate()` return (definitions: `benchmarks/token_accounting.py`).
> Streamer callback counts are recorded for reference only, since this build's text
> streamer is not 1:1 with tokens (old rows: 32 requested → 30 callbacks, 128 requested
> → 101 callbacks, a run that was itself capacity-truncated — the `re-*` re-run of that
> config generated 123 tokens; `re-*` rows: 31 callbacks for 32 tokens, 122 for 123). NPU evidence is layered: requested device, device query
> (`npu_device_query`, `FULL_DEVICE_NAME = "Intel(R) AI Boost"`), and plugin libraries
> mapped in process — the `/proc/self/maps` check proves plugin load, not
> per-inference NPU execution, and no execution telemetry exists in this build
> (`npu_execution_evidence: null`). Rows recorded before this change used callback
> counts as the decode numerator and must not be mixed with `re-*` rows in one
> comparison.

## Baseline correction

The previous session's "13 tok/s" divided total wall time by token count, folding TTFT
and process overhead into decode. A second correction (issue #1): the streamer
**callback count** is not a generated-token count (31 callbacks for 32 generated tokens
is typical), so callback-count numerators also under-report. All current figures below
come from the **token-ID re-measurement** (2026-11-19, rows labeled `re-*` in
`benchmarks/results/results.jsonl`): the numerator is `EncodedResults.tokens` length − 1,
the interval is first streamer callback → `generate()` return. Exception: the
**Compile (cold)** column was not re-measured — those values come from the historical
cold-compile rows (`baseline-cold` 6.15 s, `e1-bestperf-cold` 19.77 s, `e2-mp64-bestperf`
19.82 s, `e2b-mp64-only` 5.92 s); compile time does not depend on token accounting.

## Results (token-ID accounting, 2026-11-19 re-measurement)

| # | Config (delta vs baseline) | Compile (cold) | Warm load | TTFT | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|---|---|
| 0 | baseline: mpl=256, FAST_COMPILE | 6.2 s | 1.0 s | 297 ms | 36.5 | 16.8 (5-tok prompt) |
| 1 | + `GENERATE_HINT=BEST_PERF` | 19.8 s | 1.1 s | 317 ms | **42.2** | 15.8 |
| 2 | `MAX_PROMPT_LEN=64` only | 5.9 s | — | 257 ms | 36.4 | 19.5 |
| 3 | mpl=64 + BEST_PERF (recommended) | 19.8 s | 1.1 s | **189 ms** | **44.6** | 26.5 |
| 4 | (#3), 128 requested → **123 generated** (capacity-truncated, see finding 4) | (cached) | 1.1 s | 196 ms | 42.3 | 25.5 |
| 5 | mpl=256 + BEST_PERF, 23-tok prompt | (cached) | 1.1 s | 295 ms | 46.2 | 78.0 |

Historical rows measured with the superseded callback-count accounting (32.9 baseline,
41.1 BEST_PERF) remain in `results.jsonl` under their original labels; the chunked-
prefill comparison (rows 6–7 of the original run: chunked 476 vs static 691 tok/s
prefill on a 289-token prompt, rows `e5-chunk128-mpl512-cold` / `e5b-static-mpl512-cold`) was not re-run — its conclusion compares two
configurations measured under the same (superseded) accounting, so the *relative*
result stands, absolute values are indicative only.

## Findings

1. **`GENERATE_HINT=BEST_PERF` is the single biggest decode win: 36.5 → 44.6 tok/s
   (+22%)** at mpl=64 (vs the mpl=256 baseline; 36.4 → 44.6 against mpl=64 alone),
   stable across 32- and 123-token generations (128 requested; the `re-*` rows are warm-cache runs). Mechanism (source-verified): replaces `NPUW_UNFOLD_IREQS=YES` with
   `NPUW_ONLINE_PIPELINE=NONE` in the decode graph config. Costs: cold compile
   6.2 s → 19.8 s (one-time, then cached at ~1.1 s warm load). Peak RSS: the
   historical rows that compiled in-process reached ~1.6–1.8 GB (vs 1.2 GB warm), but
   warm-cache `re-*` runs show only ~+35 MB for BEST_PERF (1249–1252 vs 1215–1216 MB),
   so the larger figure is a compile-time cost, not a steady-state one. Only valid with the default generate config (throws otherwise).
2. **`MAX_PROMPT_LEN=64` cuts TTFT 297 → 189 ms (−36%)** for short prompts. The static
   prefill graph pads to the bucket, so the bucket size dominates TTFT; prompt length
   itself is a minor term (at mpl=256 + BEST_PERF, 5 vs 23 tokens: 317 vs 295 ms, rows
   `re-bestperf-mpl256` / `re-medium-mp256-bestperf` — a difference within run-to-run
   variance). `MAX_PROMPT_LEN`
   is aligned up to multiples of 64 by the plugin, so 64 is the floor. Prompts longer
   than the bucket are rejected (`input_ids.get_size() <= m_max_prompt_len` check in
   `pipeline_stateful.cpp:360`) — size the bucket to your workload.
3. **Chunked prefill (`PREFILL_HINT=DYNAMIC` + `NPUW_LLM_PREFILL_CHUNK_SIZE`) works but
   is slower than static** at this model size: 476 vs 691 tok/s prefill on a 289-token
   prompt (historical rows, same accounting on both sides). Per-chunk overhead dominates any dynamism benefit for 0.6B. Not recommended.
4. `MIN_RESPONSE_LEN` is aligned to 64 by the plugin. **Correction (issue #13):** the
   default 64 does *not* sustain a 128-token generation at `MAX_PROMPT_LEN=64` — the
   KV window is mpl + mrl = 128 tokens total (prompt + generated combined), and the
   re-measured row 4 (`re-mp64-bestperf-128tok`) generated **123 of 128 requested
   tokens** with a 5-token prompt (5 + 123 = 128). The earlier "sustains 128" claim
   came from the superseded callback-count rows, which never counted tokens. To get
   full responses, size the window so that
   **`prompt_tokens + max_new_tokens ≤ MAX_PROMPT_LEN + MIN_RESPONSE_LEN`**; anything
   beyond it is silently truncated (prefix-exact, no error, not a natural EOS). Confirmed on the
   NPU (2026-11-19, `validation/results/boundary-probes.json`: at 256 + 64, 192 × 128
   completes, 193 × 128 → 127, 200 × 128 → 120). The plugin aligns `MIN_RESPONSE_LEN`
   up to a multiple of 64 but GenAI's stop check uses the configured value, so use
   multiples of 64. Source trace:
   [npu-early-stop-investigation.md](npu-early-stop-investigation.md).
5. Prefill throughput scales with prompt length (16.8 tok/s @ 5 tokens in `re-baseline-warm`
   → 691 tok/s @ 289 tokens in the historical `e5b-static-mpl512-cold` row; prefill =
   prompt tokens / TTFT, which the token-accounting change does not affect): fixed
   per-inference overhead, not per-token prefill cost.

## Recommended configuration

```python
genai.LLMPipeline(
    model_dir, device='NPU',
    MAX_PROMPT_LEN=64,       # 256 if prompts can be long; TTFT scales with bucket
    MIN_RESPONSE_LEN=64,     # window = 64 + 64 = 128 tokens, prompt + generated
    GENERATE_HINT='BEST_PERF',
    CACHE_DIR='<persistent cache dir>',
)
```

**Capacity rule:** the NPUW KV window holds `MAX_PROMPT_LEN + MIN_RESPONSE_LEN` tokens,
prompt and generated combined. With this config a request must satisfy
`prompt_tokens + max_new_tokens ≤ 128` (e.g. a 5-token prompt gets at most 123 new
tokens), or the generation is **silently truncated** to `128 − prompt_tokens` tokens.
For longer responses raise `MIN_RESPONSE_LEN` (multiples of 64); its effect on TTFT,
decode speed and compile time has not been measured here.

Delivered as `run-npuw.sh`. Trade-off to remember: BEST_PERF's 19.8 s cold compile is
paid once per (model, config, compiler) cache key; with a persistent `CACHE_DIR` warm
load is ~1.1 s.

## Failures recorded

- Prompt exceeding `MAX_PROMPT_LEN` → `RuntimeError: Check 'data->input_ids.get_size()
  <= m_max_prompt_len' failed at src/cpp/src/llm/pipeline_stateful.cpp:360` (reproduced
  with 700-token prompt at mpl=512). Serving layer must enforce the bucket limit.
- `GENERATE_HINT` with a custom generate config → `GENERATE_HINT only works with
  default generate config!` (source-verified, not reproduced here).
