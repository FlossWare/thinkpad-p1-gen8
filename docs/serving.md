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

## Measurements

### Clean re-runs (2026-11-19, new client, corrected server; issue #2)

All rows: 20 requests, 16 new tokens each, short prompts, mpl=64, BEST_PERF, fresh
per-run sessions, **zero failures in every run**. Source:
[`serving/results/serving-results.jsonl`](../serving/results/serving-results.jsonl),
rows labeled `re-*`. "Nominal tok/s" = successful × 16 / wall (the server does not
report generated-token counts; labeled as such in the raw rows).

| Arrangement (raw label) | Attempted | Failed | Successful req/s (nominal tok/s) | p50 latency | p95 latency |
|---|---|---|---|---|---|
| 1 pipeline, sequential (`re-w1-sequential`) | 20 | 0 | 1.93 (30.9) | 0.50 s | 0.87 s |
| 1 pipeline, 4 clients (`re-w1-concurrent-c4`) | 20 | 0 | 1.86 (29.8) | 2.03 s | 2.68 s |
| 1 pipeline, 8 clients (`re-w1-concurrent-c8`) | 20 | 0 | 1.82 (29.0) | 4.43 s | 4.54 s |
| 2 pipelines, sequential (`re-w2-sequential`) | 20 | 0 | 1.88 (30.0) | 0.51 s | 0.89 s |
| 2 pipelines, 4 clients (`re-w2-concurrent-c4`) | 20 | 0 | 2.17 (34.7) | 1.83 s | 2.54 s |

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
3. Host RSS for the two-pipeline process is only ~422 MB — the compiled blobs live in
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
