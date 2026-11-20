# Multi-Session Serving — NPUW Qwen3-0.6B on Intel NPU

Date: 2026-11-19. Prototype: `serving/server.py` (this repo); benchmark client:
`serving/bench_serving.py`; raw results: `serving/results/serving-results.jsonl`.

## Concurrency support in this release (source/API-inspected)

- `LLMPipeline` (stateful variant) holds per-generation state and is not
  documented thread-safe; there is no batched/continuous-batching mode for it in this
  build. Concurrent `generate()` calls on one pipeline are therefore **not** attempted.
- Multiple `LLMPipeline` instances in one process are supported (two verified), but
  they share the single NPU — see measurements below.
- The NPU plugin exposes no multi-command-stream scheduling we can influence from the
  GenAI Python API.

## Architecture (delivered)

Single persistent `LLMPipeline` owned by exactly **one worker thread**; HTTP front end
(`ThreadingHTTPServer`) validates and enqueues; the worker executes strictly
sequentially.

- **Request isolation**: every request re-prefills its full prompt; no pipeline state
  survives a request. Session history is explicit (`SessionStore`), appended only after
  a successful generation.
- **Predictable errors**: over-length prompts (session history included) are rejected
  pre-queue with HTTP 400 and a specific message (`prompt_too_long: N tokens >
  MAX_PROMPT_LEN 64`); malformed/missing/invalid fields are rejected with 400 before
  queueing; a request whose prompt tokens (history included) plus `max_new_tokens`
  exceed the NPUW KV window `MAX_PROMPT_LEN + MIN_RESPONSE_LEN` (64 + 64 = 128 by
  default; reported as `kv_capacity` in `/health`) is rejected with 400
  `exceeds_kv_capacity: ...` (see "KV-window capacity" below); pipeline and tokenizer exceptions surface as HTTP 500 JSON; a request
  not answered within 300 s gets HTTP 504 after 300 s (a still-queued request is then
  skipped; an in-flight one finishes but its result is discarded and never recorded); a full request queue (bounded, `--max-queue`/`QUEUE_MAX`, default 32) →
  HTTP 429 `overloaded` and the request is not queued; admitting a *new* session
  while all `--max-sessions` slots are active is likewise a 429 (see
  "Session capacity and eviction" below).
- **Ordering**: requests to the same session serialize in arrival order (history
  snapshot + generation + record are atomic; a request abandoned with 504 after the
  timeout is never recorded); different sessions may run
  concurrently. `POST /reset` is ordered with the session's generations and waits
  for earlier work before clearing history. At most 8 operations may be pending
  (running or waiting) per session; further ones get HTTP 429 immediately.
- **Graceful shutdown**: `POST /shutdown`, SIGTERM, SIGINT stop accepting, cancel
  queued-not-started requests with 503, finish in-flight work, and join workers
  within `--drain-timeout`/`DRAIN_TIMEOUT_S` (default 30 s); a drain timeout is
  logged and reported; HTTP handlers get until the drain deadline + 2 s to deliver
  their responses; the process then exits normally (no `os._exit`).
- Endpoints: `POST /generate`, `POST /reset`, `GET /health`, `POST /shutdown`.
- **KV-window capacity (issue #13)**: NPUW generates at most
  `MAX_PROMPT_LEN + MIN_RESPONSE_LEN − prompt_tokens` tokens and stops there without
  any error (prefix-exact truncation; confirmed on the NPU, see
  [npu-early-stop-investigation.md](npu-early-stop-investigation.md)). The
  `max_new_tokens` range check alone ([1, 256]) would admit requests that are
  certain to be truncated at the default `--max-prompt-len 64` (window 128), so the
  server additionally rejects `prompt_tokens + max_new_tokens > kv_capacity` with 400
  before queueing; the error states the at-most token count the request would have
  received. Limitation: the server counts the raw prompt text, but `generate()`
  applies the model's chat template, which adds tokens. The check therefore rejects
  only certainly-truncated requests; an accepted request within a few template tokens
  of the window can still come back shorter than `max_new_tokens` (the response
  carries no truncation flag). For the same reason the `prompt_too_long` check is a
  lower bound on the real model input length.
- Known prototype limitations (not fixed): on drain timeout, workers still inside
  `generate()` are abandoned at interpreter exit (daemon threads), not finished;
  `/health.requests_done` counts every dequeued request, including ones skipped after
  a client timeout.

## Session capacity and eviction (issue #12)

Retained session state is now **bounded**: each stored session owns one history list
and one `SessionGate`, and the store keeps at most `--max-sessions`
(env `SESSIONS_MAX`, **default 1024**) of them. The selected policy is *idle-based
LRU eviction on admission* (not hard rejection of new sessions).

- **Idle-time origin**: an entry's timestamp is the monotonic time of its last
  *completed* operation (`SessionStore.finish`); a session that has not completed an
  operation yet uses its creation time.
- **Active means pinned**: an entry is active — and is never evicted or replaced —
  from the moment admission takes its gate ticket until that ticket is released.
  This covers waiting on the gate, work sitting in a worker queue, in-flight
  generation, and `/reset`. A generation abandoned with HTTP 504 after the 300 s
  timeout releases its ticket immediately (its result is never recorded), so it no
  longer pins state.
- **When eviction happens**: admitting a *new* session id while the store is at
  capacity removes the idle entry with the oldest completion time (ties broken by
  session id, so the choice is deterministic). Eviction removes the history and the
  gate together.
- **Atomic admission**: lookup-or-create, eviction and ticket acquisition happen in
  one critical section. Concurrent first requests for the same new session therefore
  share exactly one gate, and concurrent admissions can never exceed the configured
  limit.
- **All entries active**: if a request would need a new session and every stored
  session is active, nothing is evicted and that request is rejected with HTTP 429
  (`{"error": "overloaded: session capacity N reached; all N stored session(s) are
  active"}`). Requests for already-stored sessions are unaffected.
- **Client-visible behaviour**: eviction is not an error and is not reported to the
  client. An evicted session id remains valid: its next `/generate` or `/reset`
  creates fresh state (empty history) and is served normally (HTTP 200 / 200). The
  client only observes that earlier context is gone (the new prompt is sent without
  prior history). Clients that must preserve context across long idle gaps should
  re-send it or increase `--max-sessions`.
- **Diagnostics**: `GET /health` now includes `sessions` (currently stored entries),
  `sessions_active` (entries holding or waiting for a ticket), `session_capacity`
  (configured maximum) and `session_evictions` (cumulative since process start).
- **Operational limit**: the LRU scan is O(number of stored sessions) and runs only
  when a new session is admitted at capacity; with the default 1024 this is
  negligible. Memory is bounded by `max-sessions × (per-session history + gate)`.
  Per-session history is checked *before* a turn, and the answer is appended afterwards.
  The counted prompt must pass `prompt_too_long` (`MAX_PROMPT_LEN`, 64 by default) and
  `HISTORY_LIMIT_TOKENS` (1024), and prompt + `max_new_tokens` must fit the KV window.
  `HISTORY_LIMIT_TOKENS` is configurable (`--history-limit` / env) and must exceed
  `--max-prompt-len`; raise it to bucket + `MIN_RESPONSE_LEN` (e.g. 1088 for mpl=1024)
  when serving large buckets (roadmap M1).
  So a retained history is roughly one accepted prompt plus its answer: about
  `MAX_PROMPT_LEN + MIN_RESPONSE_LEN` tokens (128 by default). Re-tokenizing joined text
  makes this approximate, not exact. The total is therefore at most 1024 sessions of
  roughly that size each.

## Measurements

### Clean re-runs (2026-11-19, new client, corrected server; issue #2)

All rows: 20 requests, 16 new tokens each, short prompts, mpl=64, BEST_PERF, fresh
per-run sessions, **zero failures in every run**. Source:
[`serving/results/serving-results.jsonl`](../serving/results/serving-results.jsonl),
rows labeled `re-*`. "Nominal tok/s" = successful × 16 / wall (the server does not
report generated-token counts; labeled as such in the raw rows).

| Arrangement (raw label) | Attempted | Failed | Successful req/s (nominal tok/s) | p50 latency | max latency (raw `latency_p95_s`) |
|---|---|---|---|---|---|
| 1 pipeline, sequential (`re-w1-sequential`) | 20 | 0 | 1.93 (30.9) | 0.50 s | 0.87 s |
| 1 pipeline, 4 clients (`re-w1-concurrent-c4`) | 20 | 0 | 1.86 (29.8) | 2.03 s | 2.68 s |
| 1 pipeline, 8 clients (`re-w1-concurrent-c8`) | 20 | 0 | 1.82 (29.0) | 4.43 s | 4.54 s |
| 2 pipelines, sequential (`re-w2-sequential`) | 20 | 0 | 1.88 (30.1) | 0.51 s | 0.90 s |
| 2 pipelines, 4 clients (`re-w2-concurrent-c4`) | 20 | 0 | 2.17 (34.7) | 1.83 s | 1.91 s |

Percentile note: the client that produced these rows computed `latency_p95_s` as
element `int(0.95·n)` of the sorted latencies, which for n = 20 is the **maximum**
(`latency_p95_s == latency_max_s` in every `re-*` row), so the last column is labeled
as such. The client now uses nearest-rank p95 (19th of 20) and records
`latency_percentile_method`; p50 is element `n//2` (upper median).

Live SIGTERM shutdown was exercised after the two-pipeline runs: the server drained
(in-flight request finished, 0 queued requests cancelled), logged
`[shutdown] stopped workers [0, 1]`, and exited normally — the orderly-shutdown
behavior required by issue #3, verified on the real service (in addition to the
in-process test suite).

Findings (clean re-runs):

1. **Concurrency does not increase single-pipeline throughput** (1.93 → 1.86 → 1.82
   req/s for sequential/4/8 clients): requests serialize on the worker; per-client
   latency grows linearly with client count (4 clients → ~4× p50), exactly as a FIFO
   queue predicts.
2. **Single vs two pipelines: no consistent benefit.** Sequential throughput is a wash
   (1.88 vs 1.93 req/s); under 4-client concurrency two pipelines are ~16% faster
   (2.17 vs 1.86 req/s) with lower per-request latency (1.83 s vs 2.03 s p50). Two
   pipelines cost a second model load and NPU device memory for no sequential gain,
   so **one pipeline + FIFO queue remains the recommendation**; add a second worker
   only if concurrent-request latency matters more than simplicity.
3. Host RSS for the two-pipeline process was observed at only ~422 MB (a manual
   observation; RSS is not recorded in `serving-results.jsonl`) — the compiled blobs live in
   NPU device memory, not host heap (single-pipeline processes measured 1.2–1.8 GB
   peak RSS in `benchmark.py`, which includes Python-side tensors).
4. Session-history growth is the practical capacity limit: at mpl=64 a session can
   hold ~3 short exchanges before `prompt_too_long`. Serving real conversations needs
   mpl=256 (TTFT ~300 ms) or client-side history truncation.

### Historical runs (superseded; kept for the record)

The original `w1-*` rows were produced by the **old** benchmark client, which recorded
only an error count and computed throughput as `n / wall` — counting failed requests
as completed (issue #2). Each `w1-*` run had 4 of 20 requests fail before reaching the
pipeline (server `requests_done` advanced by 16, pipeline `errors` stayed 0); the old
client did not preserve per-request outcomes, so the exact cause is unrecoverable from
the repository. The earlier published claim that "two pipelines are 22% slower than
one" compared those error-inflated figures with clean two-pipeline runs and **has been
withdrawn**; the clean re-runs above supersede the whole comparison. See the raw file
for the full history (`sequential-n20-c4`, `concurrent-n20-c4`, `w1-*`, `w2-*`,
`w2clean-*`).

## Recommendation

**One pipeline + FIFO queue** (the default `--workers 1`). It is the simplest
arrangement, the clean re-runs show no sequential benefit from a second pipeline
(1.88 vs 1.93 req/s), and it is the only arrangement where error semantics are fully
predictable. A second worker gives a modest concurrent-latency improvement (16% under
4 clients) at the cost of a second model load and NPU device memory — add it only if
that trade is worth it for your workload. Scale by queueing, not by replicating
pipelines. Conventional batching is not available for this pipeline type in this
build and should not be assumed for future builds without re-measurement.

## GPU serving

`serving/server.py --device GPU` runs the same multi-session server on the
Intel Arc iGPU via the plain stateful path (M3 phase 1):

- **No KV-capacity window**: `kv_capacity` is `null` in `/health` and the
  `MAX_PROMPT_LEN` / `exceeds_kv_capacity` request checks are skipped — the
  GPU path is not bucket-bound (only `HISTORY_LIMIT_TOKENS` still bounds
  sessions). Measured (Qwen3-1.7B INT4): ~56 tok/s decode, ~60 ms TTFT,
  full-length generations.
- `--f32` (GPU only) forces `INFERENCE_PRECISION_HINT=f32` +
  `KV_CACHE_PRECISION=f32` for CPU-identical output (see the M2 correctness
  notes in the roadmap); both hints are required on this build.
- Model selection via `NPU_MODEL_DIR` as usual (e.g. the 1.7B/4B IRs in the
  FlossWare model store).

## Anthropic Messages-API shim (`POST /v1/messages`)

Text-only v1 of the Anthropic Messages API, for driving local models as
ensemble workers/arbiters from any HTTP client:

- Request: `model`, `max_tokens` (1–4096), `system`, `messages[]` (string or
  text-block content), optional `stream` (SSE). The conversation is flattened
  to `system + Human:/Assistant: turns`; each call is self-contained (no
  session history).
- Response (non-streaming): Anthropic message shape with `content` blocks and
  `usage`; streaming: standard SSE event sequence (message_start through
  message_stop; v1 emits the full text as one delta).
- Errors use the Anthropic error envelope: `400 invalid_request_error`
  (malformed request, `prompt_too_long`), **`400` with an explicit message
  when `tools` is present — tool use is NOT supported locally and is never
  silently ignored**, `429 overloaded_error` (queue saturation), `504` on
  generation timeout.
- Works on both `--device NPU` and `--device GPU`; one server serves one
  model (the `model` field is accepted as-is). `claude -p` integration
  requires tool-use emulation — documented limitation, not attempted in v1.

## Tests (no NPU required)

```bash
python3 -m pytest serving/ benchmarks/ -q   # or: python3 serving/test_server.py -v
```

## Reproduce

```bash
cd thinkpad-p1-gen8 && source scripts/env.sh
python3 serving/server.py --port 8931 --workers 1 &
python3 serving/bench_serving.py --mode sequential --n 20
python3 serving/bench_serving.py --mode concurrent --n 20 --clients 4
# result lines now carry attempted/successful/failed, errors_by_class,
# per-request outcomes and successful-only throughput
curl -s -X POST localhost:8931/generate -d \
  '{"session":"s1","prompt":"The capital of France is","max_new_tokens":16}'
curl -s -X POST localhost:8931/shutdown
```
