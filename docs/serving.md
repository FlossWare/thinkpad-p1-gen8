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
  queueing; pipeline exceptions surface as HTTP 500; 300 s in-queue timeout → HTTP
  504; a full request queue (bounded, `--max-queue`/`QUEUE_MAX`, default 32) →
  HTTP 429 `overloaded` and the request is not queued.
- **Ordering**: requests to the same session serialize in arrival order (history
  snapshot + generation + record are atomic); different sessions may run
  concurrently. `POST /reset` is ordered with the session's generations and waits
  for earlier work before clearing history. At most 8 operations may be pending
  (running or waiting) per session; further ones get HTTP 429 immediately.
- **Graceful shutdown**: `POST /shutdown`, SIGTERM, SIGINT stop accepting, cancel
  queued-not-started requests with 503, finish in-flight work, and join workers
  within `--drain-timeout`/`DRAIN_TIMEOUT_S` (default 30 s); a drain timeout is
  logged and reported; HTTP handlers get until the drain deadline + 2 s to deliver
  their responses; the process then exits normally (no `os._exit`).
- Endpoints: `POST /generate`, `POST /reset`, `GET /health`, `POST /shutdown`.

## Measurements (20 requests, 16 new tokens each, short prompts, mpl=64, BEST_PERF)

Source: [`serving/results/serving-results.jsonl`](../serving/results/serving-results.jsonl)
(row label in the first column). These rows were produced by the **old** benchmark
client, which recorded only an error count and computed `throughput_req_per_s` as
`n / wall` — i.e. it counted failed requests as completed (issue #2). The table below
is reconciled with the raw file: "Successful req/s" is derived as
`(n − errors) / wall_s` from the raw row; "nominal tok/s" assumes 16 tokens per
successful request (the server does not report generated-token counts). Latency
percentiles are over successful requests only.

| Arrangement (raw label) | Attempted | Failed | Successful req/s (nominal tok/s) | Old `n/wall` figure | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| 1 pipeline, sequential (`w1-sequential`) | 20 | **4** | 2.00 (32.0) | 2.50 | 0.50 s | 0.57 s |
| 1 pipeline, 4 concurrent clients (`w1-concurrent-c4`) | 20 | **4** | 2.02 (32.3) | 2.52 | 1.97 s | 2.00 s |
| 1 pipeline, 8 concurrent clients (`w1-concurrent-c8`) | 20 | **4** | 2.01 (32.2) | 2.52 | 3.95 s | 4.94 s |
| 2 pipelines (round-robin), sequential (`w2clean-sequential`) | 20 | 0 | 1.94 (31.0) | 1.94 | 0.49 s | 0.83 s |
| 2 pipelines, 4 concurrent clients (`w2clean-concurrent-c4`) | 20 | 0 | 1.97 (31.5) | 1.97 | 2.00 s | 2.54 s |

The raw file also contains earlier rows not summarized here (`sequential-n20-c4`:
8/20 failed; `concurrent-n20-c4`: 5/20 failed; `w2-*`: two-pipeline runs taken with a
stray idle server loaded, superseded by `w2clean-*`).

What the raw data says about the single-pipeline failures: in each `w1-*` run the
server's `requests_done` advanced by exactly 16 and its pipeline `errors` stayed 0, so
the 4 failed requests were rejected **before reaching the pipeline** (a pre-queue HTTP
4xx or a client-side failure) — not generation failures. The old client printed each
error message to the console only; the messages were not preserved, so the exact cause
cannot be recovered from the repository. (Four failures out of 20 matches one of the
five cycled prompts failing every time — a hypothesis, not a finding.) The new client
records per-request status and server error text, so re-runs will settle this.

Findings:

1. **Concurrency does not increase throughput** on one pipeline (2.00 → 2.02 → 2.01
   successful req/s for 1/4/8 clients, within noise): requests serialize on the worker;
   per-client latency grows linearly with client count (4 clients → ~4× p50), exactly as
   a FIFO queue predicts.
2. **Single vs two pipelines: no supported conclusion yet.** The previously published
   "two pipelines are ~22% slower than one" compared the old `n/wall` figure (2.50 req/s,
   counting 4 failed requests as completed) with clean two-pipeline runs. Counting
   successful requests only, single-pipeline (2.00–2.02 req/s) and two-pipeline
   (1.94–1.97 req/s) runs are within ~3% — and the single-pipeline runs are not clean.
   A clean, comparable re-run with the new client is required (issue #2); until then
   this comparison must not be cited.
3. Host RSS for the two-pipeline process is only ~411 MB — the compiled blobs live in
   NPU device memory, not host heap (single-pipeline processes measured 1.2–1.8 GB
   peak RSS in `benchmark.py`, which includes Python-side tensors).
4. Session-history growth is the practical capacity limit: at mpl=64 a session can
   hold ~3 short exchanges before `prompt_too_long`. Serving real conversations needs
   mpl=256 (TTFT 266 ms) or client-side history truncation.

## Recommendation

**One pipeline + FIFO queue** (the default `--workers 1`). It is the simplest
arrangement, a second pipeline showed no measured throughput benefit, and it is the only
one where error semantics are fully predictable. (Whether a second pipeline is slower
awaits the clean re-run noted above.) Scale by queueing, not by replicating pipelines. Conventional batching is
not available for this pipeline type in this build and should not be assumed for
future builds without re-measurement.

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
