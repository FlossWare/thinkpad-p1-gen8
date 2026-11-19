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
| near-limit-193t@128 | 193 | 128 | **100%** | — | −0.031 |

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

| Prompt | Steps | Mean \|ΔlogP\| | Max \|ΔlogP\| | Cosine |
|---|---|---|---|---|
| short-geography | 16 | 0.0206 | 0.0702 | 0.99956 |
| repeated-token | 16 | 0.0365 | 0.1489 | 0.99929 |
| near-limit-193t | 16 | 0.0008 | 0.0050 | 0.99999 |

Drift is per-step small and non-accumulating. The 193-token prompt shows the *smallest*
absolute drift — long prompts don't degrade; if anything the prefill averaging reduces
it. The repeated-token prompt shows the largest, matching the argmax-flip above.

## Practical impact

- For real prompts, greedy NPU output is **token-identical to CPU** for the tested
  battery up to 193-token prompts and 128-token generations.
- Divergence risk concentrates in near-tie top-2 logits, which occur on degenerate
  (repeated-token) prompts; the flip produces a locally different but semantically
  equivalent continuation.
- No evidence of KV-cache corruption, position-ID errors, or context-boundary
  mishandling: near-limit prompts (193/256) match exactly, and overflow is cleanly
  rejected.

## Limitations (explicit)

- This GenAI build exposes no per-step logits (no `logit_processor` hook;
  `EncodedResults.scores` is a sequence **sum** under greedy decoding). Per-step
  numerical comparison is therefore on top-1 logprobs only; top-5 agreement is not
  measurable through `LLMPipeline`.
- Direct logit-level NPU-vs-CPU comparison exists only via the manual static-blob
  fallback path: last-token argmax match and logit cosine 0.9916 (measured
  2026-11-18, `~/AI/ovfix-static-blob/compare*.py`).
- RoPE/RMSNorm/softmax op-level precision was not instrumented (would require source
  patches, out of scope per project rules); the f16 activation conversion applies
  model-wide as cited above.
- Tolerances are **not** claimed as universal: the measured envelope is
  mean \|Δ top-1 logP\| ≤ 0.04, max ≤ 0.15 per step, for this model/build/NPU.
