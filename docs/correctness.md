# Correctness Validation — Qwen3-0.6B INT4, CPU vs NPU (NPUW)

Date: 2026-11-19. Harness: `validation/validate.py`; raw data:
`validation/results/validation-results.json`. Reference: same OpenVINO 2026.5 build + same model on the
CPU plugin of the same wheel. NPU side: NPUW with the tuned config
(`MAX_PROMPT_LEN=256, GENERATE_HINT=BEST_PERF`). Greedy decoding, `logprobs=1`.

## KV-cache representation and precision (source-verified, not inferred)

- The original IR declares **56 stateful KV-cache variables** (Qwen3-0.6B: 28 layers ×
  K/V), dynamic shape `[?,8,?,128]` — see `validate.py` `kv_cache_report()` output.
- NPUW storage precision for this configuration is **f16**:
  `choose_kv_cache_storage_type()` (`llm_compiled_model.cpp:587`) defaults to
  `ov::element::f16`; the only automatic override, `NPUW_LLM_OPTIMIZE_FP8`, defaults to
  **false** (option table, `npuw_option_defs.inc:71`); we set no
  `ov::hint::kv_cache_precision`. Integer KV storage (key i8/u8, value i4) exists in
  `convert_kvcache_to_precision.cpp` but is a hint-only path (and i4 value-cache is
  marked broken in a source TODO). **The KV cache is f16 regardless of the INT4
  weights.**
- Additionally the whole kvcache model gets a **bf16 → f16 activation conversion**
  (`llm_compiled_model.cpp:965`). So NPU-side numerics differ from CPU reference by:
  INT4 weight dequantization on both, f16 activations and f16 KV storage on NPU.
- Context limit is enforced against the bucket: prompts longer than `MAX_PROMPT_LEN`
  are rejected (`pipeline_stateful.cpp:360`), reproduced at 289 tokens vs mpl=256.

## Continuous-generation battery (token-level, argmax decisions)

| Test | Prompt tokens | Gen len | Token match | First divergence | ΔΣlogP |
|---|---|---|---|---|---|
| short-geography@32 | 5 | 32 | **100%** | — | −0.190 |
| short-geography@64 | 5 | 64 | **100%** | — | −0.219 |
| short-geography@128 | 5 | 128 | **100%** | — | −0.375 |
| short-code@32 | 5 | 32 | **100%** | — | −0.018 |
| repeated-token@32 | 17 | 32 | 93.8% | step 2 | −5.058 |
| number-pattern@32 | 86 | 32 | **100%** | — | +0.007 |
| medium-nlp@32 | 23 | 32 | **100%** | — | +0.076 |
| medium-nlp@64 | 23 | 64 | **100%** | — | +0.101 |
| near-limit-193t@32 | 193 | 32 | **100%** | — | −0.034 |
| near-limit-193t@128 | 193 | 128 CPU / **127 NPU** | 100% of the 127 common tokens (not identical: NPU 1 token shorter) | — | −0.031 |

Raw labels: the `near-limit-193t` rows are stored as `near-limit-250t@32` / `@128` in
`validation-results.json` (the harness name predates the measured 193-token count).
**Strict count: 8 of 10 battery tests are token-identical.** `near-limit-193t@128` is
*not* identical: the NPU produced 127 tokens to the CPU's 128 (`token_comparison`
`{n: 127, match_pct: 100.0, identical: false}`); the NPU sequence is an exact prefix of
the CPU sequence. 193 + 127 = 320 = `MAX_PROMPT_LEN` (256) + `MIN_RESPONSE_LEN` (64): the
NPUW KV-cache window is exactly mpl + mrl tokens (prompt + generated combined), enforced
by GenAI's per-sequence capacity check (`lm_encoding.cpp:134` with
`utils.cpp:933`) and by the plugin's own guards (`llm_infer_request.cpp:1505, 1544`);
the NPU run requested 321 tokens of window — one more than configured — so it stopped
gracefully at the limit. Source-verified, not just arithmetic: see
[npu-early-stop-investigation.md](npu-early-stop-investigation.md). **Empirically
confirmed** (NPU machine, 2026-11-19, 11 boundary probes, raw:
[`validation/results/boundary-probes.json`](../validation/results/boundary-probes.json),
script `validation/boundary_probe.py`): the 193 × 128 case reproduces (127 tokens),
192 × 128 completes all 128 (192 + 128 = 320), 200 × 128 stops at 120, and every
truncating case is prefix-exact. This is expected capacity behavior, not a bug; the
battery result above is unchanged — it stays a 127-token prefix match, **not** a
128-token match, and the strict count stays 8 of 10.

ΔΣlogP = NPU sum-of-top-1-logprobs minus CPU, over the full generation. Magnitudes
grow with length (−0.19 → −0.38 for 32 → 128 tokens) roughly linearly (~3e-3/token),
consistent with per-step numerical drift, not accumulating corruption.

## The one divergence: repeated-token prompt

Prompt `"the " × 16`. CPU: `[220, 16, 15, 15, 15, …]`; NPU: `[220, 16, 62, 16, 15,
15, 15, …]`. Both continue the degenerate repetition; 30/32 positions match; outputs
are semantically identical.

Interpretation: at step 2 the top-2 logits were within the f16 precision envelope;
the NPU's f16 activations/KV flipped the near-tie (CPU chose token 15, NPU token 62,
then NPU re-emitted 16 before joining the `15,15,…` pattern). This is an argmax
flip on a tie-ish distribution — the standard, expected behavior of reduced-precision
inference, not cache corruption or context mishandling. Prompts of repeated identical
tokens maximize attention-score quantization sensitivity (all keys identical), which
is why only this prompt diverges.

## Per-step numerical drift (incremental single-token generation)

Each step re-prefills prompt+generated and records that step's top-1 logprob
(`logit − logsumexp(vocab)`). Continuous-greedy equivalence was confirmed implicitly:
stepwise token choices matched the continuous sequences for all three prompts.

The "cosine" column below is **cosine similarity over the two 16-element vectors of per-step top-1 log-probabilities** (log P of the greedy token at each step, 16 steps, CPU vs NPU) — **not** full-vocabulary logit cosine. It is computed by `compare_logprobs()` in `validation/validate.py` and stored as the `cosine` field in `validation/results/validation-results.json`. Because each element is the log-probability of the token *each device chose*, a step where the argmax differs compares log-probabilities of different tokens. See Limitations for the separate last-position logit metric, which exists only on the manual static-blob fallback path.

| Prompt | Steps | Mean \|ΔlogP\| | Max \|ΔlogP\| | Cosine similarity (16-step top-1 logprob vectors, CPU vs NPU) |
|---|---|---|---|---|
| short-geography | 16 | 0.0206 | 0.0702 | 0.99956 |
| repeated-token | 16 | 0.0365 | 0.1489 | 0.99929 |
| near-limit-193t | 16 | 0.0008 | 0.0050 | 0.99999 |

Drift is per-step small and non-accumulating. The 193-token prompt shows the *smallest*
absolute drift — long prompts don't degrade; if anything the prefill averaging reduces
it. The repeated-token prompt shows the largest, matching the argmax-flip above.

## Practical impact

- For real prompts, greedy NPU output matched CPU token-for-token over every common
  position in the tested battery (prompts up to 193 tokens, generations up to 128).
  One such case (`near-limit-193t@128`) ended one token early on NPU (127 vs 128,
  prompt + generation reaching the 320-token mpl + min-response capacity), so the
  strictly identical count is 8 of 10.
- **Capacity rule (confirmed, issue #13):** NPUW generates at most
  `MAX_PROMPT_LEN + MIN_RESPONSE_LEN − prompt_tokens` tokens. A larger
  `max_new_tokens` is silently truncated (prefix-exact, no error). Size
  `MIN_RESPONSE_LEN` so that `prompt_tokens + max_new_tokens ≤ MAX_PROMPT_LEN +
  MIN_RESPONSE_LEN`.
- Divergence risk concentrates in near-tie top-2 logits, which occur on degenerate
  (repeated-token) prompts; the flip produces a locally different but semantically
  equivalent continuation.
- No evidence of KV-cache corruption, position-ID errors, or context-boundary
  mishandling: near-limit prompts (193/256) match CPU on every common token, and
  overflow is cleanly rejected.

## Limitations (explicit)

- This GenAI build exposes no per-step logits (no `logit_processor` hook;
  `EncodedResults.scores` is a sequence **sum** under greedy decoding). Per-step
  numerical comparison is therefore on top-1 logprobs only; top-5 agreement is not
  measurable through `LLMPipeline`. For the same reason no NPUW full-vocabulary
  logit comparison is reported: the API does not permit one in this build (issue #8's
  optional full-logit item is not achievable through `LLMPipeline` here).
- Direct logit-level NPU-vs-CPU comparison exists only via the **separate manual
  static-blob fallback path** (not NPUW): last-token argmax match and **cosine
  similarity of the last-position logit vector 0.9916** (measured 2026-11-18,
  `~/AI/ovfix-static-blob/compare*.py`; script and raw output are outside this repo).
  This is a different path and a different metric from the 16-step top-1-logprob
  cosine similarities above — token agreement, top-1-logprob drift, and logit-vector
  similarity are three distinct measures.
- RoPE/RMSNorm/softmax op-level precision was not instrumented (would require source
  patches, out of scope per project rules); the f16 activation conversion applies
  model-wide as cited above.
- Tolerances are **not** claimed as universal: the measured envelope is
  mean \|Δ top-1 logP\| ≤ 0.04, max ≤ 0.15 per step, for this model/build/NPU.
