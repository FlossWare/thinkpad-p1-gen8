# NPUW Early-Stop on Near-Limit Prompts — Source Investigation (issue #13)

Date: 2026-11-19. Method: read-only source trace of the OpenVINO / GenAI trees at
`~/AI/src/openvino` (intel_npu NPUW plugin) and `~/AI/src/openvino.genai` (GenAI), both
on the `releases/2026/5` branches matching the installed `openvino-2026.5` wheel
(plugin-load evidence in `benchmarks/results/results.jsonl` points at
`~/AI/install/openvino-2026.5-python` and the compiler loader built from
`~/AI/src/openvino/bin`). The source trace and the numbers in it were derived from data
already recorded in this repo (`docs/correctness.md`, `benchmarks/results/results.jsonl`).
The conclusion was then **confirmed empirically on the NPU machine** by 11 boundary
probes (2026-11-19, raw: [`validation/results/boundary-probes.json`](../validation/results/boundary-probes.json));
see "Empirical confirmation" below.

## Reproduction configuration (as recorded)

- Model: Qwen3-0.6B INT4, OpenVINO GenAI `LLMPipeline`, device `NPU` (NPUW).
- Pipeline config (`validation/validate.py:37`): `MAX_PROMPT_LEN=256`,
  `MIN_RESPONSE_LEN=64`, `GENERATE_HINT=BEST_PERF`, `CACHE_DIR=~/AI/npuw-cache`.
- Greedy decoding (`do_sample=False`), `max_new_tokens=128`, 193-token prompt
  (`near-limit-250t` in `validation-results.json`, ~193 tokens measured).
- Observed (recorded in `docs/correctness.md:39`): CPU 128 tokens, NPU 127 tokens,
  NPU output an exact prefix of CPU. 193 + 127 = 320 = 256 + 64.

## The counter-example re-examined (it dissolves)

The issue's counter-example — "a 5-token prompt at `MAX_PROMPT_LEN=64` completed a
128-token generation (5+128=133 > 128)" — is **not supported by the recorded data**.
The token-ID re-measurement of exactly that configuration
(`benchmarks/results/results.jsonl`, row `re-mp64-bestperf-128tok`:
`MAX_PROMPT_LEN=64, MIN_RESPONSE_LEN=64`, 5-token prompt, `max_new_tokens=128`)
records **`generated_tokens: 123`** in all three runs, and 5 + 123 = 128 = 64 + 64 —
the same boundary arithmetic as the 193-token case. The output preview ends
mid-sentence ("…But wait, maybe I should "), i.e. no natural EOS.

The earlier belief that 128 tokens "completed fine" came from the superseded
callback-count accounting (historical row `e3-mp64-bestperf-128tok` records only
`streamer_chunks: 101` and no token count; this streamer is documented as not 1:1
with tokens — `docs/npuw-tuning.md:16-18`). `docs/npuw-tuning.md` finding #4
("the default 64 already sustains 128-token generations") was inferred from those
rows and is corrected by this investigation (see "Documentation corrections").

So the capacity theory has **zero recorded counter-examples** and two independent
confirmations: 193 + 127 = 320 (mpl=256, mrl=64) and 5 + 123 = 128 (mpl=64, mrl=64).

## Source trace

### 1. Capacity: total KV window = MAX_PROMPT_LEN + MIN_RESPONSE_LEN

Plugin side (`~/AI/src/openvino/src/plugins/intel_npu/src/plugin/npuw/`):

- `llm_compiled_model.cpp:829-830` — both options are read and **aligned up to
  multiples of 64**: `max_prompt_len = align_to(NPUW_LLM_MAX_PROMPT_LEN, 64)`,
  `min_response_len = align_to(NPUW_LLM_MIN_RESPONSE_LEN, 64)`. For 256/64 both are
  already aligned; no bucket-alignment artifact enters this case.
- `llm_compiled_model.cpp:1012-1013` — the KV-cache descriptor is built as
  `KVCacheDesc{max_prompt_len, max_prompt_len + min_response_len, ...}`, i.e.
  **`total_size = mpl + mrl = 320`** for our config. This is where 320 comes from.
- Plugin-level defaults (for reference): `npuw_option_defs.inc:62-64` —
  `NPUW_LLM_MAX_PROMPT_LEN` default 1024, `NPUW_LLM_MIN_RESPONSE_LEN` default 128,
  `NPUW_LLM_MAX_GENERATION_TOKEN_LEN` default **1**.
- GenAI-level defaults: `openvino.genai/src/cpp/src/utils.cpp:780-781` —
  `MAX_PROMPT_LEN` default 1024, `MIN_RESPONSE_LEN` default **128**. Our harness sets
  both explicitly, so the defaults don't apply here.
- The generate model is reshaped by
  `ReshapeToStatic(max_generation_token_len, kv_size, ...)` 
  (`llm_compiled_model.cpp:688`), and `reshape_to_static.cpp:150-151` sizes the
  past-KV inputs to `kvcache_size - input_size` = 320 − 1 = **319 past slots**, with
  1 in-flight input token per step. The window 320 therefore decomposes as
  319 storable past tokens + 1 current-step input slot.

### 2. The GenAI stopping condition (the one that actually fired)

- `openvino.genai/src/cpp/src/utils.cpp:923-934` — `get_npu_kv_cache_capacity()`
  returns `max_prompt_len + min_response_len - max_generation_token_len`
  (= 256 + 64 − 1 = **319**; the `- max_generation_token_len` reserves the in-flight
  slot, with an explicit comment that the property is assumed 1 because
  `NPUW_LLM_MAX_GENERATION_TOKEN_LEN` is not exposed to GenAI).
- `openvino.genai/src/cpp/src/llm/pipeline_stateful.cpp:489-492` — for NPU devices
  the stateful pipeline passes this value as `max_kv_cache_size` into
  `get_lm_encoded_results()` (NPU routes through `StatefulLLMPipeline`;
  `llm/pipeline.cpp:287`).
- `openvino.genai/src/cpp/src/lm_encoding.cpp:131-138` — after every sampling step
  (called at `lm_encoding.cpp:204` after prefill, `:314` mid-infer for streaming, and
  `:329` after each decode sample):

  ```cpp
  if (sg->get_prompt_len() + seq->get_generated_len() - 1 == max_kv_cache_size) {
      seq->set_status(SequenceStatus::OUT_OF_MEMORY);
  }
  ```

  Substituting: `prompt_len + generated_len - 1 == 319`, i.e. the sequence is killed
  exactly when **`prompt_len + generated_len == mpl + mrl`**. For a 193-token prompt:
  193 + g = 320 → g = **127**. For a 5-token prompt at mpl=64/mrl=64: 5 + g = 128 →
  g = **123**. Both match the recorded data exactly.
- The stop is **graceful**: an `OUT_OF_MEMORY` sequence is no longer `is_running()`
  (`sequence_group.hpp:151-160, 695-710`), so its group counts as finished and is
  dropped from the active set (`lm_encoding.cpp:139-143`); all tokens generated so
  far are still published through the generation handle
  (`sequence_group.hpp:1139-1147, 1179-1185` — OOM maps to `GenerationStatus::IGNORED`
  and `notify_handle_oom()` still streams the accumulated output). This is why the
  NPU result is a clean 127-token **exact prefix** of the CPU result rather than an
  error.

### 3. The plugin hard limit (preempted by GenAI, same boundary)

- `llm_infer_request.cpp:1505-1512` — before each generate step, if
  `num_stored_tokens >= get_current_variant_capacity()`, the plugin tries
  `try_switch_to_larger_variant()` and otherwise **throws** `KV-Cache is full`.
  `get_current_variant_capacity()` (`llm_infer_request.cpp:920-932`) returns
  `kv_size - max_generation_token_len` = 319 — the past-slot count, matching the
  reshape above.
- `llm_infer_request.cpp:1543-1546` — a second guard throws when
  `num_stored_tokens + input_tokens_len > total_size` (320).
- `llm_infer_request.cpp:1610` — `num_stored_tokens += input_tokens_len` after each
  step; prefill adds the prompt length at `:1312`.

Token accounting: after prefill, `S = prompt_len`; generate step *k* feeds generated
token *k* (entry `S = prompt_len + k − 1`) and samples token *k+1*. The step-*k*
entry check `S >= 319` admits *k* only while `prompt_len + k − 1 ≤ 318`, so the last
admissible step samples token `g = 320 − prompt_len`. For prompt 193 that is 127 —
**the same number GenAI's check produces**. GenAI's OOM check fires one sampling step
before the plugin would throw, so the plugin exception is never reached through
`LLMPipeline`; the two layers enforce the identical boundary
(`prompt + generated = mpl + mrl`), GenAI just fails gracefully instead of throwing.

### 4. Why the cache is never extended

- `llm_compiled_model.cpp:643-673` — with `NPUW_LLM_GENERATE_PYRAMID` (default
  **true**, `npuw_option_defs.inc:66`) the plugin pre-compiles larger generate-model
  variants at sizes `1024 + mrl, 2048 + mrl, …` while they fit under `total_size`.
  For total = 320 the loop body never executes (`1024 + 64 > 320`), so
  `m_kvcache_sizes = {320}` — a single variant, nothing larger to switch to.
- `try_switch_to_larger_variant()` (`llm_infer_request.cpp:934-956`) therefore
  returns false at the boundary. There is no runtime cache extension path; capacity
  is fixed at compile time by mpl + mrl. The 5-token mpl=64 case was never an
  extension — it was the counter-example being mis-measured (see above).

## Verdict

**CONFIRMED (source-backed):** the 320 boundary is **expected capacity behavior, not
a bug and not an off-by-one**. The NPUW KV window is exactly
`MAX_PROMPT_LEN + MIN_RESPONSE_LEN` tokens (prompt + generated combined), enforced
identically at two layers: GenAI's per-step OOM check
(`lm_encoding.cpp:134` + `utils.cpp:933`) stops the sequence gracefully when
`prompt_len + generated_len` reaches the window, and the plugin's own guards
(`llm_infer_request.cpp:1505, 1544`) would throw at the same point. Requesting
`max_new_tokens=128` with a 193-token prompt asks for 321 tokens of window — one
more than the configuration provides — so the NPU "ends one token early" by design;
CPU has no such finite window and completes 128. The `-1`s that look suspicious
(`- max_generation_token_len` in `utils.cpp:933`, `- 1` in `lm_encoding.cpp:134`,
`kv_size - input_size` in `reshape_to_static.cpp:151`) all reserve the same single
in-flight input slot and cancel out: the boundary lands exactly on
`prompt + generated = mpl + mrl`, as both recorded runs show (193+127=320,
5+123=128).

- **Not a bucket-alignment artifact**: 193 is never rounded to 256; the boundary is
  computed from actual token counts (`num_stored_tokens`, `get_prompt_len()`,
  `get_generated_len()`). mpl/mrl alignment to 64 (`llm_compiled_model.cpp:829-830`)
  is a no-op for 256/64.
- **Not a streamer/accounting artifact**: the 127 count comes from the EncodedResults
  token IDs (token-ID accounting, `docs/npuw-tuning.md` methodology note), and the
  stop is a `SequenceStatus::OUT_OF_MEMORY` transition, not EOS and not a callback
  miscount.

**EMPIRICALLY CONFIRMED (NPU machine, 2026-11-19):** the boundary probes below
reproduce the original case (193 × 128 → 127) and land on
`prompt + generated = mpl + mrl = 320` in every case that reaches it, while the
last-fitting case (192 × 128) completes all 128 tokens. What the probes do *not*
observe directly is the internal `SequenceStatus::OUT_OF_MEMORY` transition (no
status is exposed through the Python API); that attribution rests on the source trace,
which predicts every probe length exactly.

**UNEXPLAINED:** nothing remains for this specific observation. One adjacent
open question, out of scope for this issue: whether a *natural* EOS could ever be
confused with the capacity stop in the recorded battery (it cannot for these two
cases — both outputs end mid-sentence).

## Empirical confirmation (NPU machine, 2026-11-19)

Script: [`validation/boundary_probe.py`](../validation/boundary_probe.py) (run with the
environment from `scripts/env.sh` and `NPU_MODEL_DIR` set:
`python3 validation/boundary_probe.py`). Config identical to the battery: mpl=256,
mrl=64, `GENERATE_HINT=BEST_PERF`, greedy, `CACHE_DIR` persistent. Prompts are exact
token-length slices of a filler text passed as `TokenizedInputs` (no chat template),
so `prompt_tokens` is the exact model input length; counts are `EncodedResults.tokens`
lengths. Raw: [`validation/results/boundary-probes.json`](../validation/results/boundary-probes.json).

| prompt | requested | CPU | NPU | prompt + NPU | predicted NPU = min(req, 320 − prompt) | common-prefix match |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 128 | 128 | 128 | 133 | 128 | no (see note) |
| 5 | 190 | 190 | 190 | 195 | 190 | no (see note) |
| 5 | 200 | 200 | 200 | 205 | 200 | no (see note) |
| 129 | 128 | 128 | 128 | 257 | 128 | yes |
| 191 | 128 | 128 | 128 | 319 | 128 | yes |
| **192** | 128 | 128 | **128** | **320** | 128 (fits exactly) | yes |
| **193** | 128 | 128 | **127** | **320** | 127 (original case reproduced) | yes |
| **200** | 128 | 128 | **120** | **320** | 120 | yes |
| 193 | 64 | 64 | 64 | 257 | 64 | yes |
| 250 | 64 | 64 | 64 | 314 | 64 | yes |
| **256** | 64 | 64 | **64** | **320** | 64 (fits exactly) | yes |

All 11 NPU lengths equal the prediction; CPU always produced the requested count. The
early stop is prefix-exact in all three truncating cases. **Note on the 5-token rows:**
`prefix_match` is false there — the CPU and NPU token sequences diverge somewhere in
these long generations from a repetitive filler prompt. This is a content divergence
(the FP16 near-tie class documented in `docs/correctness.md`), not a length effect:
lengths match exactly and 5 + 200 = 205 is far from the 320 window. It was not
investigated further here.

Together with the recorded mpl=64 row (`re-mp64-bestperf-128tok`: 5 + 123 = 128 =
64 + 64), the boundary is confirmed at two different `MAX_PROMPT_LEN` values. Of the
planned probes below, #1, #2 and the #3 subset 129/191/200 were run, #5 is covered by
the recorded mpl=64 row; #3 (255), #4 (316 requested), #6 (mrl=128), #7 (explicit EOS
check) and #8 were **not** run. #6 is the remaining direct test that the window moves
with `MIN_RESPONSE_LEN` (the source and the mpl=64 data point say it does).

## Planned boundary probes (pre-run design)

All greedy, same model/config as the battery. Prediction in every case:
**generated = min(requested, mpl + mrl − prompt_len)**, output an exact prefix of the
CPU run, ending mid-sentence (no EOS).

| # | Probe | Prediction | What the outcome proves/disproves |
|---|---|---|---|
| 1 | mpl=256, mrl=64; prompt 192 × requested 128 | **exactly 128** (192+128=320, fits exactly) | The sharpest discriminator: 192 is the *last* prompt length that fits 128 tokens. If this also stops early (127), the boundary formula is wrong. |
| 2 | mpl=256, mrl=64; prompt 193 × requested 128 | **127** (reproduces the recorded case) | Reproducibility of the original observation. |
| 3 | mpl=256, mrl=64; prompts 129 / 191 / 200 / 255 × requested 128 | 128 / 128 / 120 / 65 | 191 is the last "comfortable" length (320−191=129 ≥ 128); 200 and 255 test the linear `320 − prompt` law away from the edge. |
| 4 | mpl=256, mrl=64; 5-token prompt × requested 316 | **315** | Tests the law far from the prompt limit, on the generation side. |
| 5 | mpl=64, mrl=64; 5-token prompt × requested 128 | **123** | Re-runs the dissolved counter-example with token-ID accounting; 128 here would disprove the theory. |
| 6 | mpl=256, **mrl=128**; prompt 193 × requested 128 | **128** (capacity 384) | Proves the boundary follows mpl + mrl (moves with mrl), not mpl alone. |
| 7 | Any probe: decode the last generated token and check `!= eos_token_id`, and confirm output ends mid-phrase | last token is not EOS | Rules out a natural-EOS explanation for the stop. |
| 8 | mpl=256, mrl=64; prompt 193 × requested 127 | **127**, byte-identical tokens to probe 2's output | Confirms the stop is capacity-bound, not `max_new_tokens`-interaction: requesting exactly the capacity yields the same sequence. |

A single probe contradicting its prediction (e.g. #1 stopping at 127, or #5 completing
128) would falsify the confirmed mechanism and reopen the investigation.

## Documentation corrections made alongside this note

- Capacity rule now stated wherever a configuration is recommended:
  `prompt_tokens + max_new_tokens ≤ MAX_PROMPT_LEN + MIN_RESPONSE_LEN`, otherwise the
  generation is silently (prefix-exactly) truncated — `docs/npuw-tuning.md`
  (finding 4, results row 4, recommended configuration), `README.md`,
  `docs/serving.md`. `serving/server.py` now rejects requests beyond the window with
  HTTP 400 `exceeds_kv_capacity` instead of truncating silently.

- `docs/correctness.md:46-48` — the capacity explanation was labeled "a hypothesis
  from the arithmetic, not verified against the plugin source here"; it is now
  source-verified (before/after quoted in the commit summary).
- `docs/npuw-tuning.md` finding #4 — "the default 64 already sustains 128-token
  generations" contradicted by the repo's own re-measured row
  (`re-mp64-bestperf-128tok`: 123 generated of 128 requested); corrected to state
  the capacity cap and its arithmetic.
