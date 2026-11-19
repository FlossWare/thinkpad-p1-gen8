# Post-Project-1 Roadmap — Local AI Milestones for the ThinkPad P1 Gen 8

Status: active plan, created 2026-11-19. Supersedes nothing; it is the forward-looking
companion to the completed Project 1 (see [Completed](#completed-project-1)).

This document tracks the next investigations into what local LLM and AI workloads are
practically useful on **this** ThinkPad P1 Gen 8 (Core Ultra 7 265H, NPU platform 3720,
integrated Intel Arc GPU). It is a plan, not a results page: **none of the active
milestones has started, and no measurements exist for them yet.** Results pages will be
linked from here as each milestone completes.

Defects in Project 1's measurement, serving, or documentation are tracked separately in
issues #1–#8, and documentation reconciliation is #11; both are explicitly out of scope
here (see [Out of scope](#out-of-scope)).

## How to read this document

| Status | Meaning |
|---|---|
| **COMPLETE** | Finished; code, docs, and raw results are linked. |
| **ACTIVE** | Scoped and justified, queued, **not started**. |
| **BACKLOG** | Deferred; one-line rationale for why it is not scheduled. |

Active milestones carry a fixed methodology written *before* the work runs, so that
results cannot be retrofitted to a preferred conclusion. If a milestone's scope changes
after data exists, the change is recorded in this file with a date rather than silently
rewritten.

## Priority: practical value × feasibility on this machine

Scores are judgment calls (1–5), not measurements. They are shown so the ranking can be
argued with rather than guessed at.

| Rank | Milestone | Practical value | Feasibility | Score | Effort | Why this rank |
|---|---|---|---|---|---|---|
| 1 | **M1 — Sustained multi-turn / longer-context serving** | 5 | 5 | 25 | M | Turns the working 0.6B path into something usable: at the current 64-token bucket a session dies after ~3 short exchanges ([docs/serving.md](serving.md)). COMPLETE 2026-11-19: mpl=1024 recommended for conversations; decode is context-independent; session isolation verified under history growth (results in the M1 section). |
| 2= | **M2 — Larger model on the NPU** | 5 | 3 | 15 | L | The biggest capability lever: 0.6B is a working path, not necessarily a useful assistant. Feasibility is uncertain because a new INT4 IR must be converted and the 3720 NPU may not compile or fit it. |
| 2= | **M4 — Driver/runtime compatibility matrix** | 3 | 5 | 15 | S, recurring | Cheap insurance. The 3720 compiler-selection defect is closed upstream without a fix ([docs/driver-compiler.md](driver-compiler.md)), so every stack upgrade is a risk to a working machine. Reuses existing smoke tests. Runs in parallel with the capability track and does not compete for it. |
| 4 | **M3 — Intel Arc GPU characterization** | 4 | 3 | 12 | M | Answers the other half of the platform question, but with the same 0.6B artifact it changes device choice more than capability. The GPU path is entirely untested and its runtime may not be installed. Best done after M1's harness exists and alongside/after M2 so the larger IR can also be tried on the GPU. |

Deliberately **not** scheduled: embeddings, dedicated thermal characterization, and the
other candidates in the [Backlog](#future--backlog). Four active milestones are enough
to occupy the machine and the engineering time without producing half-finished
comparisons.

**Execution order:** M1 first; capture the M4 baseline immediately afterwards (a
snapshot of the current working stack, before anything is upgraded); then M2; then M3.
M3's second phase reuses M2's artifact, so slipping M3 behind M2 costs nothing and
avoids converting the same model twice.

## What the repository already establishes

Everything below is **measured on this individual machine** unless marked otherwise.
These facts define the starting point and the open questions; the active milestones are
designed around them.

| Fact | Source |
|---|---|
| NPUW Qwen3-0.6B INT4 decode is **44.6 tok/s** with `GENERATE_HINT=BEST_PERF` at mpl=64; baseline `FAST_COMPILE` is 36.5 tok/s (token-ID accounting, re-measured 2026-11-19). | [docs/npuw-tuning.md](npuw-tuning.md) |
| TTFT is dominated by the shape bucket: **189 ms** at `MAX_PROMPT_LEN=64` (BEST_PERF), **297 ms** at 256 (baseline) — rows `re-mp64-bestperf`, `re-baseline-warm`. The plugin aligns the bucket up to multiples of 64; prompts longer than it are rejected, not truncated. | [docs/npuw-tuning.md](npuw-tuning.md) |
| `BEST_PERF` costs **19.8 s cold compile** (per model/config/compiler cache key) and **~1.1 s warm load** from a persistent `CACHE_DIR`. | [docs/npuw-tuning.md](npuw-tuning.md) |
| Chunked prefill is **slower** than static prefill at 0.6B (476 vs 691 tok/s on a 289-token prompt; historical rows, not re-measured). | [docs/npuw-tuning.md](npuw-tuning.md) |
| NPUW stores the KV cache in **FP16** (source-verified), independent of INT4 weights; CPU-vs-NPU greedy output was strictly token-identical on 8/10 battery tests: one near-tie argmax flip, and one 193-token-prompt run where the NPU matched all 127 common tokens but stopped one token short of 128 (prompt + output = 320 = mpl + min-response). Per-step top-1-logprob drift: mean ≤ 0.04, max 0.15; cosine similarity of the 16-step top-1-logprob vectors ≥ 0.9992 (raw minimum 0.99929; not a full-logit cosine). | [docs/correctness.md](correctness.md) |
| One pipeline + FIFO queue serves **1.8–1.9 successful req/s** (16-token requests, clean `re-w1-*` runs); concurrency adds latency linearly and no throughput. Clean two-pipeline runs: sequential a wash (1.88 req/s), ~16% faster under 4 clients (2.17 vs 1.86 req/s). At `mpl=64` a session holds only ~3 short exchanges. | [docs/serving.md](serving.md) |
| Platform 3720 defaults to the **driver** NPU compiler on Linux and fails model-independently; `IE_NPU_COMPILER_TYPE=PLUGIN` is mandatory. Upstream openvino#38738 is closed "not planned"; no Intel driver release through v1.38.0 documents a fix. | [docs/driver-compiler.md](driver-compiler.md) |
| openvino#39184 (open) reports Qwen3.5-4B INT8 failing on NPU37XX due to dynamic hybrid KV-cache shapes — relevant to model/format choice in M2. | [docs/driver-compiler.md](driver-compiler.md) |
| Platform: Core Ultra 7 265H, NPU PCI `8086:7d1d` (platform 3720), Fedora 44 kernel `7.3.4-200.fc44.x86_64`, intel-npu-driver 1.32.0, Level Zero 26.35.39758.21, OpenVINO 2026.5.0 commit `88f8c6bf7c`, GenAI `releases/2026/5` @ `2de7224b`. The Intel Arc GPU has **not** been identified or tested in this repository yet. | [docs/environment.md](environment.md), [hardware/thinkpad-p1-gen8.md](../hardware/thinkpad-p1-gen8.md) |

### Measured facts vs. interpretation vs. hypothesis

The distinction is maintained throughout, and every results page must keep it:

- **Measured facts** — the numbers in the table above, reproducible from the linked
  scripts and raw JSON.
- **Interpretation** (supported by those measurements, but still specific to this
  model/build/device): TTFT is governed by bucket size rather than prompt length; the
  single token divergence is an FP16 near-tie flip rather than cache corruption; throughput
  serializes because one stateful pipeline owns the NPU.
- **Hypotheses, untested** — the active milestones exist to test these and must be free
  to refute them (H6 belongs to the backlog embeddings item, not to an active
  milestone):
  - H1: a 1024-token bucket remains interactive for multi-turn chat on this NPU.
  - H2: Qwen3-1.7B INT4 will compile with the plugin compiler and fit NPU memory.
  - H3: decode throughput scales roughly inversely with active parameter count.
  - H4: the Arc GPU beats the NPU on prefill for long prompts.
  - H5: sustained generation within long sessions produces measurable clock/thermal
    drift.
  - H6: embedding models are viable on the NPU or GPU with useful throughput.

Results in this repository characterize **one P1 Gen 8 configuration**. They are not
claims about all P1 Gen 8 systems, all Arrow Lake NPUs, or Intel hardware generally.
Component variance (firmware, binning, memory configuration, chassis/thermal paste) is
not controlled for and cannot be inferred from these results.

## Evidence conventions (shared by all milestones)

These follow the patterns already used by `benchmarks/results/`,
`validation/results/`, and `serving/results/`.

1. **Raw, machine-readable results are checked in.** One JSON object per run,
   configuration, or session, appended to a `.jsonl` file (or a single `.json` for a
   structured battery, as `validation/results/validation-results.json` does). Summaries
   in Markdown are derived from these files and link to them.
2. **Proposed paths** (created by the milestone, not present today):
   `serving/results/multiturn-results.jsonl`, `benchmarks/results/gpu-results.jsonl`,
   `benchmarks/results/npu-large-models.jsonl`,
   `validation/results/compat/<date>-<stack>.json`.
3. **Every record carries its environment**: repository `git_commit`, timestamp,
   device, model name and IR hash, and the stack versions (OpenVINO, GenAI, NPU driver,
   Level Zero, kernel) read from the live system at run time. Results without this
   metadata are not accepted into `results/`.
4. **Device evidence is recorded in layers, not assumed**: requested device, any
   resolved-device query, and the `/proc/self/maps` plugin check (extended with the GPU
   plugin, `libopenvino_intel_gpu_plugin.so`). Plugin presence is recorded as
   plugin-load evidence only — never as proof that each measured inference ran on that
   device — following the convention already used by `benchmarks/benchmark.py`.
5. **Failures are data.** Compile errors, OOM, timeouts, rejected prompts, and garbage
   output are recorded with `status: "failed"`, an `error_class`, and a short stderr
   excerpt. Failed runs are never deleted or overwritten; they are the negative-result
   record the issue asks for.
6. **Scripts and commands are committed with the results.** A results file whose
   generating command cannot be reconstructed from the repository is treated as an
   anecdote, not evidence.
7. **Raw files are append-only.** Corrections are new records with a note, not edits to
   history.

## Measurement rules (fixed before any comparison)

These rules follow the validated token-accounting methodology implemented in
[`benchmarks/token_accounting.py`](../benchmarks/token_accounting.py) and
[`benchmarks/benchmark.py`](../benchmarks/benchmark.py) (Project 1, issue #1).
Raw measured results for that implementation are in
[`benchmarks/results/results.jsonl`](../benchmarks/results/results.jsonl) (rows
`re-*`; older `baseline-*`/`e*` rows used the superseded callback-count
numerator and must not be mixed with `re-*` in one comparison) and
[`serving/results/serving-results.jsonl`](../serving/results/serving-results.jsonl).

- **Warm-up:** one uncounted generation (`max_new_tokens=2`, greedy) per pipeline
  instance before timed runs. Cold compile is measured once per (model, config,
  compiler) cache key with a throwaway `CACHE_DIR`; warm load is measured with the
  persistent cache. First-inference effects on a new device (notably GPU kernel
  compilation) are reported as a separate `first_infer_s`, not folded into TTFT.
- **Timing boundaries — four distinct metrics:**
  - *load/compile* — from immediately before the `LLMPipeline(...)` constructor to
    immediately after it returns.
  - *TTFT (time to first token)* — from `generate()` entry (`t_gen0`) until the
    first streamer callback (`token_times[0]`). Tokenization happens outside the
    timed region; prompt token count is recorded separately.
  - *Prefill throughput* — `prompt_tokens / TTFT`, where `prompt_tokens` is the
    tokenizer `encode()` length of the prompt. No generated tokens are included in
    this numerator.
  - *Decode throughput* — `(generated_tokens − 1) / (t_gen_end − token_times[0])`,
    where `generated_tokens` is `len(result.tokens[0])` from the `EncodedResults`
    object returned by `generate(TokenizedInputs, ...)` and `t_gen_end` is the
    perf-counter timestamp immediately after `generate()` returns. The interval is
    **first streamer callback → `generate()` return**, not last-callback, and the
    numerator is **generated token IDs minus one**, not a callback count.
  - *End-to-end turn wall* — HTTP submit to response completion, reported
    separately from TTFT so queue wait and transport are visible. This is the only
    metric that includes queueing; TTFT and decode are measured inside
    `generate()` and do not include it.
- **Why callback/chunk counts are not token counts on this build:** the text
  streamer delivers text chunks, not tokens, and on this build the callback count
  differs from the generated-token count: the `re-*` rows record 31 callbacks for
  32 generated tokens and 122 for 123; the superseded rows recorded 30 callbacks
  for 32 requested and 101 for 128 requested (that config actually generates only
  123 tokens — it is capacity-truncated, issue #13 — so "128" there was never a
  token count either). See `benchmarks/results/results.jsonl`. Using `n_chunks`
  or `len(token_times)` as the decode numerator therefore miscounts tokens, by an
  amount that varies with streamer version and content. Callback counts are therefore
  recorded only as `streamer_chunks` reference data, never as a throughput
  numerator. See `benchmarks/token_accounting.py` for the accounting rules and
  the known limitation that the first callback is assumed to mark the first
  generated token.
- **Repetitions:** at least 5 timed generations per configuration, across at least 2
  process launches where load behavior matters; report **median and min–max**, never a
  single run. M1 additionally uses at least 3 independent sessions per configuration.
- **Noise:** the baseline spread from these repetitions defines the resolution. A
  difference smaller than the baseline min–max is reported as "unresolved", not as a
  win.
- **Decoding is greedy** (`do_sample=False`) and token counts come from the model's own
  tokenizer, matching Project 1's methodology.
- **Reporting:** measured facts, interpretation, and hypotheses are separate sections in
  every results document, and the single-machine limitation is restated there.
- **Requested vs generated tokens:** `max_new_tokens` is not a token count either.
  NPUW stops silently once `prompt_tokens + generated_tokens` reaches
  `MAX_PROMPT_LEN + MIN_RESPONSE_LEN` (issue #13,
  [npu-early-stop-investigation.md](npu-early-stop-investigation.md)), and EOS can
  stop earlier; every milestone records `generated_tokens` and sizes its NPUW
  config so the requested length fits the window.
- **Methodology-change rule:** if a future runtime exposes a different reliable
  timing interface (e.g., per-token timestamps, a new result field), the change
  must be documented in this file with a date and rationale, old and new results
  kept in separate, clearly labeled result sets, and never silently mixed in one
  throughput comparison. Historical results keep their original accounting label.

---

## Active milestones

### M1 — Sustained multi-turn and longer-context serving on the NPU

**Status:** COMPLETE (2026-11-19; results at the end of this section). **Effort:** medium.

**Question.** With the existing Qwen3-0.6B INT4 NPUW pipeline and the single-worker
server, what happens to TTFT, decode rate, per-turn latency, and memory as a realistic
conversation grows through the 256-, 512-, and 1024-token buckets — and at what point
does it stop being usable or fail outright?

**Why it matters.** Project 1 proved a request works, not that a conversation works.
The measured serving limit is stark: at `MAX_PROMPT_LEN=64` a session holds ~3 short
exchanges before HTTP 400, and TTFT is known to scale with the bucket, not the prompt.
A laptop assistant needs dozens of turns. Choosing between "fast but amnesiac" (64) and
"slower but coherent" (256–1024) is the first practical decision this project should
make, and nobody has measured the 512/1024 side.

**Scope.**

- In: NPU only; one pipeline, one worker; the existing server; buckets 256, 512, 1024;
  up to 16 turns; 64-token greedy replies; session reset/isolation behavior under
  history growth.
- Out: model-quality evaluation, speculative decoding, prefix caching (only available
  under chunked prefill, which measured slower at 0.6B), contexts above 1024 tokens,
  multi-user fairness, and device comparisons (M3).

**Configuration and artifacts.**

- Model: the existing `qwen3-0.6b-int4` stateful OpenVINO IR (56 KV variables). Record
  `sha256sum` of the IR `.xml`/`.bin` and keep the manifest in the results file.
- Stack: exactly the README-tested stack — OpenVINO 2026.5.0 @ `88f8c6bf7c`, GenAI
  `releases/2026/5` @ `2de7224b`, intel-npu-driver 1.32.0, Level Zero
  `26.35.39758.21`, kernel `7.3.4-200.fc44.x86_64`, `IE_NPU_COMPILER_TYPE=PLUGIN`.
- Pipeline config: `MIN_RESPONSE_LEN=64`, `GENERATE_HINT=BEST_PERF`, persistent
  `CACHE_DIR`; `MAX_PROMPT_LEN` is the independent variable.
- Server change required: `serving/server.py` hard-codes `HISTORY_LIMIT_TOKENS = 1024`
  and counts template tokens, so the 1024 bucket needs a matching, documented limit
  change; otherwise sessions reject before exercising the bucket. The change and its
  rationale go in the M1 results document.

**Workload and warm-up.**

- Fixed conversation script committed with the harness: one recorded system prompt,
  then 8 recorded user prompts of increasing size (targets ~32, ~64, ~128 prompt
  tokens, actual counts recorded), each followed by a 64-token greedy reply. Sessions
  run to 4, 8, and 16 turns, truncating when the bucket is reached; a rejection is
  recorded as `fail_at_turn`, not retried.
- One uncounted warm-up generation per pipeline instance. Each bucket gets one cold
  compile measurement and one warm-load measurement before timed sessions.
- Thermal annex (basic): sample package/NPU temperature and, if the platform exposes
  it, RAPL power at 1 s intervals during the longest sessions. **If no sensor or power
  telemetry is readable, record that fact — do not estimate or substitute synthetic
  values.** A full thermal milestone remains in the backlog.

**Timing boundaries and repetitions.**

- Per-turn TTFT, decode, and turn wall as defined in the shared rules; queue wait is
  reported separately (the HTTP client measures submit→worker start).
- 3 independent sessions per (bucket, turn-count) cell; median and min–max per turn
  index. 20 sequential requests per bucket for steady-state throughput.
- Include a 5-repeat short-prompt run per bucket so the conversation numbers can be
  tied back to the Project 1 baseline (189 ms / 44.6 tok/s at mpl=64).

**Metrics.** Per turn: `prompt_tokens`, `ttft_s`, `decode_tok_s`, `turn_wall_s`,
`queue_wait_s`, `rss_mb`. Per session: `errors`, `fail_at_turn`, `peak_rss_mb`,
optional `temp_c` / `power_w`. Per bucket: cold compile, warm load, throughput, p50/p95.

**Evidence plan.**

- New `serving/multiturn_bench.py` — drives the conversation script against the running
  server and writes one JSONL record per session, extending the existing record shape
  (`serving/results/serving-results.jsonl`) with per-turn entries.
- Commands (the script does not exist yet):
  ```bash
  source scripts/env.sh
  python3 serving/server.py --port 8931 --workers 1 --max-prompt-len 256 &
  python3 serving/multiturn_bench.py --port 8931 --bucket 256 --turns 16 \
      --max-new 64 --out serving/results/multiturn-results.jsonl
  python3 serving/bench_serving.py --mode sequential --n 20 \
      --out serving/results/multiturn-serving.jsonl
  ```
- New `docs/multiturn-serving.md` with the results, the measured/interpreted/hypothesis
  split, the bucket recommendation, and explicit limitations.

**Exit criteria.** A table of per-turn latency and failure points for 256/512/1024,
a stated recommended bucket for conversation use, all raw JSONL checked in, and any
negative result (e.g., 1024 rejected earlier than expected, thermal drift observed)
recorded rather than omitted.

**Results (measured 2026-11-19 — M1 COMPLETE).** Harnesses as built:
`benchmarks/multi_turn.py` (direct pipeline) and `serving/m1_server_probe.py`
(HTTP; queue wait + isolation), plus `benchmarks/m1_thermal_pass.py` (thermal
annex). Raw data: `benchmarks/results/m1-multi-turn.json`,
`serving/results/m1-server-256.json`, `serving/results/m1-server-1024.json`,
`benchmarks/results/m1-thermal-1024.json` (model IR sha256 manifest included in
the first). Conversation shape: ~14-token system prompt + 8 user prompts
(6–54 tokens), 64-token greedy replies → ~87 tokens of growth per turn.

| Bucket (mpl) | KV capacity | Turns before stop | Steady TTFT | Steady decode | Cold compile |
|---|---|---|---|---|---|
| 256 | 320 | 3 (direct) / 4 (HTTP) | 291 ms | 47.6 tok/s | 1.3 s (cached) |
| 512 | 576 | 6 | 441 ms | 48.6 tok/s | 21.0 s |
| 1024 | 1088 | 11 (direct) / 14 (HTTP) | 1,204 ms | 43.0 tok/s | 22.3 s |

Measured findings:

1. **Decode rate is context-independent**: 38–48 tok/s at every bucket and every
   turn depth; conversation growth does not degrade decode.
2. **TTFT is bucket-dominated and flat across turns**: at mpl=1024 TTFT is
   ~1.14 s at turn 1 and turn 11 alike — every turn pays the full padded-bucket
   prefill. TTFT is a function of the configured bucket, not the conversation.
3. **Conversation length follows the KV-capacity rule exactly** (no
   earlier-than-expected failure): turns ≈ (bucket + mrl − first prompt) / ~87.
   The server rejects the over-capacity turn cleanly (`prompt_too_long`, exact
   token count, HTTP 400) — recorded as a stop, never a crash.
4. **Queue wait and isolation (HTTP)**: 2–5 ms queue wait solo and with two
   interleaved sessions; interleaved per-turn latencies match solo — session
   isolation holds under history growth.
5. **Thermal annex**: thermal zones readable (10 zones); during a sustained
   11-turn mpl=1024 session the sampled zone ranged 55–59 °C — no throttling
   signature. RAPL domain `intel-rapl:0` was NOT readable (recorded as
   unavailable; no power values are estimated).
6. **Cold compile is a one-time per-bucket cost** (21–22 s for 512/1024, then
   ~1 s warm reload from the persistent cache).

**Recommended bucket for conversation use: mpl=1024.** ~1.2 s TTFT and ~3.0–3.4 s
per 64-token turn remain interactively usable, and the bucket holds 11–14 turns of
this conversation shape; mpl=256 is snappier (~1.9 s/turn) but holds only 3–4
turns. Serve with `--max-prompt-len 1024 --history-limit 1088` (the
`--history-limit` flag was added for this milestone and is documented in
docs/serving.md). Longer conversations need either a larger bucket (compile cost
and TTFT grow with the bucket) or client-side history truncation.

---

### M2 — Larger models on the NPU

**Status:** COMPLETE (2026-11-19; results below). **Effort:** large (conversion + compile risk).

**Question.** What is the largest dense Qwen3-family model this 3720 NPU can compile and
run usefully, and at what context length and generation speed? The concrete first
candidate is **Qwen3-1.7B INT4**; escalation beyond it is conditional on success.

**Why it matters.** Qwen3-0.6B was chosen to make the pipeline work, not because it is
useful. The practical question — "can this laptop run a model worth talking to?" — is
unanswered. NPU memory is not directly readable from the host in this stack (compiled
blobs are device-resident; see [docs/serving.md](serving.md)), the compiler has known
stateful/dynamic-shape limitations on NPU37XX (openvino#39184), and compile time and
cache behavior may change with model size. This milestone decides
whether the NPU is a viable host for a useful model or a 0.6B-class accelerator.

**Scope.**

- In: NPU only; Qwen3-1.7B INT4 weight-only OpenVINO IR first, Qwen3-4B INT4 only if
  1.7B passes; buckets 256 and 1024; compile behavior, memory, TTFT, decode, and a
  correctness sanity check; negative results for models that fail.
- Out: GPU placement (M3 phase 2), quantization-quality research, fine-tuning, MoE and
  reasoning-model families (backlog), hybrid NPU+GPU execution.

**Configuration and artifacts.**

- Target model: **Qwen3-1.7B**, converted to an INT4 weight-only OpenVINO IR
  (stateful) with the optimum-intel/NNCF conversion path. **No IR for this model exists
  yet**, and this plan makes no claim about its size or behavior. The conversion
  recipe, the exact optimum-intel/NNCF versions, the pinned Hugging Face revision, and
  the resulting IR hashes must be recorded in the M2 results document before any
  benchmark is run. The conversion command itself is committed so the artifact is
  reproducible.
- Control: the existing `qwen3-0.6b-int4` IR, re-measured on the same day in the same
  session so that machine-state drift is not attributed to the model.
- Stack: identical to M1. `IE_NPU_COMPILER_TYPE=PLUGIN` is mandatory.
- Conditional escalation: Qwen3-4B INT4 only if 1.7B compiles, fits, and generates
  coherently. Note that openvino#39184 reports Qwen3.5-4B **INT8** failing on NPU37XX;
  INT4 weight-only is the path to try, and reproducing that failure is a legitimate
  negative result, not a detour.

**Workload and warm-up.**

- Compile matrix first, generating nothing: `MAX_PROMPT_LEN` ∈ {256, 1024} ×
  `GENERATE_HINT` ∈ {FAST_COMPILE, BEST_PERF}, each with a fresh cache. Record
  success/failure, compile time, and on-disk cache size.
- Generation: the fixed Project 1 prompts (`short` 5-token, `medium` 23-token) plus the
  289-token long prompt, `max_new_tokens` 32 and 128, greedy. One uncounted warm-up
  generation per configuration.
- Correctness sanity: a subset of `validation/validate.py` (short, medium, and
  near-limit prompts) against the CPU plugin on the same build; token match and top-1
  logprob drift reported with the same caveats as Project 1.

**Timing boundaries and repetitions.**

- Load/compile, TTFT, decode as defined in the shared rules. Compile failures are timed
  to timeout and recorded; a compile that takes longer than 10 minutes is recorded as a
  failure with that bound.
- 5 timed generations per configuration after warm-up, over 2 process launches; median
  and min–max. One cold compile per matrix cell.

**Metrics.** `compiled` (bool), `cold_compile_s`, `warm_load_s`, `cache_size_mb`,
`ttft_s`, `decode_tok_s`, `prefill_tok_s`, `peak_rss_mb`, `prompt_tokens`,
`correctness.token_match`, `error_class`, and an output preview for the sanity check.
NPU memory itself is not directly readable from the host in this stack; report host RSS
and NPU execution evidence, and state the limitation rather than inventing a device
memory figure.

**Evidence plan.**

- Harness: `benchmarks/benchmark.py` already accepts `NPU_MODEL_DIR`; extend records
  with `model_dir`, IR hashes, and `stack{}` (the current file records config but not
  model identity). A small `benchmarks/compile_matrix.py` drives the compile-only
  matrix.
- Raw results: `benchmarks/results/npu-large-models.jsonl` (append-only, including
  failed rows). Conversion recipe and hashes: `docs/models/qwen3-1.7b-int4.md`.
- Commands (planned):
  ```bash
  source scripts/env.sh
  NPU_MODEL_DIR=$HOME/AI/models/openvino/qwen3-1.7b-int4 \
    python3 benchmarks/benchmark.py --label qwen3-1.7b-int4-mpl256 \
    --max-prompt-len 256 --set GENERATE_HINT=BEST_PERF --repeats 5 \
    --out benchmarks/results/npu-large-models.jsonl
  python3 benchmarks/compile_matrix.py --model-dir "$NPU_MODEL_DIR" \
    --out benchmarks/results/npu-large-models.jsonl
  ```

**Exit criteria.** A size/format feasibility statement backed by checked-in results:
which model compiled, at which buckets, how fast, how much host memory, and — equally
important — the exact failure mode for anything that did not. If 1.7B fails, that
negative result closes the milestone and informs whether the GPU (M3) becomes the
primary path for larger models.

---

**Results (measured 2026-11-19 — M2 COMPLETE).** Two 1.7B INT4 artifacts were
tested (both recorded, reproducible):

- **Official** `OpenVINO/Qwen3-1.7B-int4-ov` @ `6f32d81e` (u4:f16 weights):
  **fails to compile on this NPU in every tested cell** (mpl 256/1024 ×
  FAST_COMPILE/BEST_PERF; `Unsupported DynamicDequantizeOp configuration:
  inElemType=!quant.uniform<u4:f16...>, arch=NPU37XX` — the openvino#39184
  failure class). Negative result, recorded.
- **Local conversion** (Qwen3-1.7B → INT4 via optimum 2.3.0 + nncf 3.4.0 +
  torch 2.15.0, `stateful=True`, `weight_format=int4`, in the isolated
  venv `~/AI/venvs/ov-convert`; tokenizer IRs copied from the official
  export): **compiles in all four cells** — 9.3/43.8 s (mpl 256,
  FAST_COMPILE/BEST_PERF), 13.2/49.8 s (mpl 1024), caches 322–350 MB.

Measured performance (mpl=256, BEST_PERF, token-ID accounting; raw rows in
`benchmarks/results/npu-large-models.jsonl`):

| Model | Device | TTFT | Decode | Verdict |
|---|---|---|---|---|
| Qwen3-0.6B INT4 | NPU | 297 ms | 36.5 tok/s | usable (Project 1 baseline) |
| Qwen3-1.7B INT4 (local) | NPU | **18.6 s** | **0.17 tok/s** | **unusable** — 201 s per 32-token generation, 3.2 GB RSS |
| Qwen3-1.7B INT4 | **GPU** | **59–64 ms** | **54–58 tok/s** | **the practical path** |

Correctness (CPU reference, `validation/validate_gpu.py`): the GPU's default
f16 path diverges from CPU token-wise on some prompts (near-tie argmax flips;
outputs remain textually equivalent — spot-checked) on BOTH models, i.e. it
is GPU-path numerics, not the 1.7B IR. Forcing
`INFERENCE_PRECISION_HINT=f32` + `KV_CACHE_PRECISION=f32` on the GPU restores
**100% token agreement** across the full battery (measured on 0.6B; the same
knob applies to 1.7B). Note: f32 with default KV precision hits a GenAI GPU
Paged-Attention block-size error on this build — both hints are required.

**M2 conclusion:** the NPU on this machine is a 0.6B-class accelerator — 1.7B
compiles (with the right quantization recipe) but runs ~330× slower than the
iGPU. **Qwen3-1.7B on the Arc iGPU is the first model on this laptop that is
both worth talking to and measured usable** (~56 tok/s, ~60 ms TTFT).
The 4B escalation is therefore a GPU question, not an NPU question (moved to
M3 phase 2). An OpenVINO master build (661 commits newer, including NPUW
dequant-chain folding) is being tested to see whether the official u4:f16 IR
compiles on a newer stack — recorded when complete; it cannot change the
0.17 tok/s verdict for the local IR.

### M3 — Intel Arc GPU characterization

**Status:** IN PROGRESS — phase 1 complete (2026-11-19; results below). **Effort:** medium.

**Question.** For the same model and IR, how does the integrated Intel Arc GPU compare
with the NPU and CPU on load/compile, first inference, TTFT, decode, and memory at
256/512/1024 tokens — and which device should host which workload?

**Why it matters.** The README lists GPU inference as untested; the hardware document
does not even record the GPU's device ID. The Arc iGPU is the only other accelerator in
the machine, and the obvious hypothesis (H4) is that it will beat the NPU on prefill and
long contexts while possibly losing on efficiency. If true, a device-selection rule
falls out of this milestone; if false, the NPU-only recommendation becomes much
stronger. Either way the answer is currently unknown.

**Scope.**

- In: GPU plugin via `genai.LLMPipeline(device='GPU')`; the existing `qwen3-0.6b-int4`
  IR; buckets 256/512/1024; CPU and NPU as references; correctness check; device
  capability pre-flight; negative results if the GPU stack is unavailable.
- Out: NPUW on GPU (NPUW is an NPU-plugin feature; the comparison is "best available
  configuration per device", not identical tuning flags), GPU driver installation
  experiments on the host OS, discrete-GPU comparisons, and multi-device execution.

**Pre-flight (must happen before any benchmark).**

- Record the GPU device ID with `lspci -nn` in
  [hardware/thinkpad-p1-gen8.md](../hardware/thinkpad-p1-gen8.md) (currently absent).
- Record available OpenVINO devices (`Core().get_available_devices()`), GPU plugin load,
  and the GPU runtime stack (intel-compute-runtime / Level Zero versions). If OpenVINO
  does not expose a GPU device, or the plugin cannot be loaded, that is the milestone's
  result: documented as unsupported on this machine, with the error text.

**Configuration and artifacts.**

- Same IR and stack as M1 (hash recorded), device `GPU`; CPU reference via the CPU
  plugin; NPU reference numbers from Project 1 re-measured on the same day.
- GenAI configuration for GPU: default generate config; `MAX_PROMPT_LEN` is still a
  GenAI-level property, but the NPUW-only hints (`GENERATE_HINT`,
  `NPUW_LLM_*`) do not apply and must not be silently carried over. Record the exact
  properties passed.

**Workload and warm-up.**

- The three fixed prompts (5, 23, and 289 tokens), `max_new_tokens` 32 and 128, greedy;
  buckets 256 and 512 for the long prompt.
- One uncounted warm-up generation, plus a separately reported `first_infer_s` (GPU
  kernel compilation may land on the first inference).
- 5 timed generations per configuration across 2 process launches.

**Timing boundaries and repetitions.** As defined in the shared rules. Load and first
inference are separate metrics; TTFT excludes both.

**Metrics.** `load_s`, `first_infer_s`, `ttft_s`, `decode_tok_s`, `prefill_tok_s`,
`peak_rss_mb`, GPU memory if an observable source exists (record the tool and
limitation; if none, say so), `device_evidence[]`, and correctness parity versus CPU
(token match / drift) using the Project 1 methodology. The iGPU shares system memory,
so any memory claim must state whether it is host RSS, device-reported, or unknown.

**Evidence plan.**

- Extend `benchmarks/benchmark.py` with `--device {NPU,GPU,CPU}` and device-specific
  evidence collection; today the device is hard-coded to `NPU` and the evidence check
  assumes NPU libraries.
- Extend `validation/validate.py` with a `--device` reference option (today it compares
  CPU vs NPU only) for GPU correctness runs.
- Raw results: `benchmarks/results/gpu-results.jsonl`,
  `validation/results/gpu-validation.json`; results document `docs/gpu.md`.
- Commands (planned):
  ```bash
  source scripts/env.sh
  python3 benchmarks/benchmark.py --device GPU --label arc-gpu-mpl256 \
    --max-prompt-len 256 --repeats 5 \
    --out benchmarks/results/gpu-results.jsonl
  python3 validation/validate.py --device GPU \
    --out validation/results/gpu-validation.json
  ```

**Exit criteria.** A device comparison table with the same workload and clearly stated
configuration differences, an evidenced recommendation (NPU / GPU / split by workload),
and either GPU support confirmed or a documented unsupported path with the exact
failure. Phase 2 (optional, after M2): run the larger IR on the GPU if it exists and the
GPU path is functional.

---


**Phase-1 results (measured 2026-11-19).** Pre-flight: GPU PCI `8086:7d51`
(recorded in hardware/thinkpad-p1-gen8.md), OpenVINO devices `['CPU', 'GPU',
'NPU']`, GPU full name "Intel(R) Arc(TM) Graphics (iGPU)". Harness:
`benchmarks/gpu_characterize.py` (token-ID accounting, same rules as the NPU
benchmarks); raw data: `benchmarks/results/gpu-m3-phase1.json`. Qwen3-0.6B INT4,
`genai.LLMPipeline(device='GPU')` (the plain stateful path — NPUW properties
like MAX_PROMPT_LEN do not apply to GPU and are rejected by the plugin):

| Case (prompt @ max_new) | TTFT | Decode | Full generation? |
|---|---|---|---|
| short (5 t) @ 32 | 41 ms | 113.0 tok/s | yes (32/32) |
| short @ 128 | 40 ms | 102.7 tok/s | yes (128/128) |
| medium (23 t) @ 32 | 42 ms | 114.3 tok/s | yes |
| medium @ 128 | 39 ms | 105.5 tok/s | yes |
| long (289 t) @ 32 | 34 ms | 102.9 tok/s | yes |
| long @ 128 | 33 ms | 98.9 tok/s | yes (128/128) |

Phase-1 findings vs the NPU (same model, Project 1 / M1 numbers):

1. **The iGPU dominates the NPU on every axis for this model**: TTFT 33–42 ms
   vs 189–1,204 ms (5–35× faster; TTFT is also prompt-length-independent on
   GPU — dynamic shapes, no bucket padding); decode 99–114 tok/s vs 43–48
   (2.1–2.6×); load 1.9 s vs ~1.1 s warm (comparable) with no 20 s cold
   compiles and no NPUW workarounds required.
2. **No KV-capacity window on the GPU path**: full 128-token generations from
   a 289-token prompt (the NPUW path truncates at mpl+mrl).
3. Output is coherent greedy generation (spot-checked; a correctness battery
   against CPU is the remaining phase-2 item, along with 512/1024-context
   scaling, memory, and multi-turn behavior on GPU).
4. Device-selection implication (measured, this model only): the iGPU, not the
   NPU, is the practical host for local LLM inference on this machine; the
   NPU's case rests on efficiency and availability under GPU load — untested
   (backlog). M2's escalation decisions should account for this.

### 

**First re-test result (2026-11-19): OpenVINO master vs the working snapshot.**
Built OV master (`2027.0.0-23963`, 661 commits ahead) + GenAI master
(`2027.0.0.0-3690`) in isolated worktrees; the working installation was
untouched. Verdict — **newer is worse on this machine; do not upgrade**:

- **GPU**: works, but measured ~2× slower decode and ~4× TTFT for Qwen3-1.7B
  (single-run probe: 31.6 tok/s / 270 ms vs 54–58 tok/s / ~60 ms on the
  snapshot).
- **NPU**: broken. Master drops the in-tree compiler and downloads a prebuilt
  VCL compiler for Ubuntu only — Fedora 44 is explicitly skipped; even with
  the sha256-verified prebuilt v8.5.0 manually installed, NPU compilation
  fails identically (`vclAllocatedExecutableCreate2`), including on the
  known-good 0.6B control model.
- The wheel still omits the NPU compiler libs (workaround remains), and the
  GenAI 2026.5 ↔ OV master ABI breaks (`SDPAToPagedAttention`).

Raw record: [`benchmarks/results/ov-master-retest.json`](../benchmarks/results/ov-master-retest.json).
This is the first row of the M4 compatibility matrix: the 2026.5 snapshot is
pinned until a future release re-tests green on both accelerators.M4 — Driver and runtime re-testing matrix

**Status:** ACTIVE (standing), not started. **Effort:** small, recurring.

**Question.** Which combinations of Intel NPU driver, Level Zero, OpenVINO, and OpenVINO
GenAI still run this machine's workloads correctly, and what changes between releases?

**Why it matters.** The platform's defining defect is a compiler-selection bug that
OpenVINO closed without fixing and that no driver release through v1.38.0 documents
fixing. The working configuration is therefore fragile with respect to exactly the kind
of upgrade a user does routinely. A fixed smoke matrix turns "did the upgrade break my
laptop?" into a short, repeatable check with a checked-in baseline, and produces the
regression record the issue asks for.

**Scope.**

- In: a fixed smoke suite plus the key Project 1 numbers; one row per tested stack; a
  documented baseline row for the current environment.
- Out: exhaustive performance benchmarking of every release, driver/firmware
  development, and kernel changes beyond what Fedora ships.

**Matrix.** Rows are added as releases appear; the baseline is the current verified
stack.

| Axis | Baseline | Candidates |
|---|---|---|
| OpenVINO | 2026.5.0 @ `88f8c6bf7c` | next upstream release/commit |
| OpenVINO GenAI | `releases/2026/5` @ `2de7224b` | next release |
| intel-npu-driver | 1.32.0 (Fedora) | latest Fedora package; note v1.38.0 exists upstream |
| Level Zero | 26.35.39758.21 | whatever the above requires |

Driver and Level Zero changes are system-wide and may need a reboot. Before any such
row: record `rpm -qa` for the relevant packages, confirm the previous package version is
available for rollback, and record the reboot in the results. A matrix row that cannot
be rolled back is not run.

**Smoke tests (fixed, run in this order).**

1. Driver-compiler control: trivial Relu graph on NPU with the driver compiler —
   **expected to fail** with `ZE_RESULT_ERROR_UNKNOWN 0x7ffffffe`; confirms the known
   defect is still present.
2. Plugin-compiler control: same graph with `IE_NPU_COMPILER_TYPE=PLUGIN` — expected to
   pass.
3. Qwen3-0.6B NPUW load at `mpl=64`, `BEST_PERF`: cold compile and warm load.
4. Short-prompt benchmark, 5 repeats: TTFT and decode, compared with the baseline.
5. Correctness subset at `mpl=256`: the 193-token near-limit prompt vs CPU, token match.
6. Serving smoke: 20 sequential requests; throughput and p50/p95.
7. Conditional (once M2 exists): larger-model load + 32 tokens.

**Timing boundaries and repetitions.** Tests 1–3 and 5 are pass/fail. Test 4 uses the
shared warm-up and 5 repeats; test 6 matches `serving/bench_serving.py`. A cell that
fails is re-run once to rule out a flake; both outcomes are recorded.

**Metrics.** `pass`/`fail` per cell, `error_class`, version strings, cold compile, warm
load, TTFT, decode, throughput, correctness match, and percentage deltas versus the
baseline row. Deltas inside the baseline min–max spread are marked "unresolved".

**Evidence plan.**

- New `validation/compat_matrix.py` orchestrating the suite and writing one JSON object
  per stack row: `validation/results/compat/2026-11-19-baseline.json` and later
  `<date>-<stack>.json`.
- Reuses `benchmarks/benchmark.py`, `validation/validate.py`, and
  `serving/bench_serving.py`; the matrix script records their exact invocations.
- Results summary `docs/compatibility.md` (a table of rows, with the raw files linked).

**Exit criteria.** A baseline row exists; every subsequent release/driver change has a
row; regressions are visible as pass/fail flips or unresolved/confirmed deltas; and
upgrade guidance ("safe", "needs workaround X", "do not upgrade") is stated with
evidence.

---

## Sequencing and dependencies

```
M1 (multi-turn)  ──►  informs bucket defaults for everything else
M4 baseline      ──►  snapshot current stack before any upgrade
M2 (larger NPU)  ──┬─►  new IR + hashes
                   └─►  M3 phase 2 (larger IR on GPU), if GPU works
M3 (Arc GPU)     ──►  device-selection rule
M4 recurring     ──►  gates every stack change
```

- M1 is independent and first.
- The M4 baseline row should be captured immediately after M1, before any component is
  upgraded, so later rows have something to compare against.
- M2 and M3 both need small harness extensions (model identity/versions in records;
  `--device`). Do that work once, in whichever milestone starts first.
- No calendar dates are assigned. Milestones are pulled in order as time allows; the
  ranking, not a schedule, decides what is next.

## Future / backlog

Each item is deferred for a stated reason. Revisit after each completed milestone.

- **Embedding models (and rerankers)** — high practical value for RAG, but needs new
  converted IRs and a device-placement decision that M2/M3 will inform; scheduling it
  now would duplicate that work.
- **Dedicated sustained power/thermal characterization** — basic temperature/power
  sampling is folded into M1; a full milestone waits until sensor access and the
  sampling method are proven, and until there is a sustained workload worth
  characterizing.
- **CPU-plugin throughput characterization** — CPU currently serves as the correctness
  reference only; a full characterization is useful only if a third comparison point is
  needed, and can be added to M3 cheaply.
- **Speculative decoding / draft models** — no verified draft-model path in this
  NPUW/GenAI build; would need upstream support before it is worth measuring.
- **FP8 or integer KV cache** — `NPUW_LLM_OPTIMIZE_FP8` defaults off and the int4
  value-cache path is marked broken in a source TODO; not a practical lever today.
- **Prefix caching for long conversations** — only active under chunked prefill, which
  measured slower at 0.6B; revisit if a future build decouples it or a larger model
  changes the prefill/decode balance.
- **Continuous batching / multi-user serving** — the stateful `LLMPipeline` has no
  batching mode in this build and serving.md measured serialization; nothing to tune
  until upstream provides it.
- **Additional runtimes (llama.cpp/GGUF, IPEX-LLM)** — a third runtime is a large setup
  cost and does not answer the NPU/GPU capability questions; defer until the OpenVINO
  path's limits are known.
- **Other model families (MoE, reasoning-oriented)** — conversion and compile
  feasibility on NPU37XX is unknown; sequence after M2's dense-model result.
- **Multimodal, ASR/TTS, image generation** — different pipelines, models, and
  evaluation criteria; outside the LLM path this repository is currently building.
- **NPU+GPU split execution / pipeline parallelism** — no supported mechanism in this
  stack; the clean two-pipeline runs show only a modest concurrent-throughput gain
  (~16% under 4 clients, none sequential), which does not suggest large wins from
  splitting work without upstream scheduling support.
- **Contexts above 1024 tokens** — depends on M1's results and on NPU memory behavior;
  scheduling it before the 1024 data exists would be guessing.
- **Model hot-swap and cache management** — convenience engineering with no capability
  payoff; low priority relative to the active milestones.

## Completed: Project 1

Project 1 — Intel NPU inference with OpenVINO GenAI NPUW and Qwen3-0.6B INT4 — is
complete (2026-11-19). It is the baseline this roadmap builds on.

| Kind | Artifacts |
|---|---|
| Code | [scripts/env.sh](../scripts/env.sh), [scripts/run-npuw.sh](../scripts/run-npuw.sh), [benchmarks/benchmark.py](../benchmarks/benchmark.py), [validation/validate.py](../validation/validate.py), [serving/server.py](../serving/server.py), [serving/bench_serving.py](../serving/bench_serving.py) |
| Documentation | [README.md](../README.md), [docs/environment.md](environment.md), [docs/npuw-tuning.md](npuw-tuning.md), [docs/correctness.md](correctness.md), [docs/serving.md](serving.md), [docs/driver-compiler.md](driver-compiler.md), [hardware/thinkpad-p1-gen8.md](../hardware/thinkpad-p1-gen8.md) |
| Raw results | [benchmarks/results/results.jsonl](../benchmarks/results/results.jsonl), [validation/results/validation-results.json](../validation/results/validation-results.json), [serving/results/serving-results.jsonl](../serving/results/serving-results.jsonl) |

When an active milestone completes, it moves into this section with the same three
links: code, documentation, and raw results.

## Out of scope

- The defects and corrections tracked in issues **#1–#8**: this roadmap does not plan,
  duplicate, or re-describe them.
- The documentation reconciliation tracked in **#11**.
- Performance claims about other machines, other Intel NPU platforms (Meteor Lake,
  Lunar Lake), other driver generations, or Intel hardware in general.
- Model quality/leaderboard evaluation beyond the coherence and correctness sanity
  checks needed to trust a measurement.

## Maintaining this document

- Move completed milestones to [Completed](#completed-project-1) with code, docs, and
  raw results linked.
- Record refuted hypotheses (e.g., "1.7B did not compile") as results, not as silent
  scope changes.
- Re-rank the backlog after each completion; promote at most one item at a time.
- Keep the priority table's rationale honest: if a milestone proves infeasible, its
  score should change, not disappear.
