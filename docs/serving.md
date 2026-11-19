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
  MAX_PROMPT_LEN 64`); pipeline exceptions surface as HTTP 500; 300 s in-queue
  timeout → HTTP 504; malformed JSON → 400.
- **Graceful shutdown**: `POST /shutdown`, SIGTERM, SIGINT all drain the queue, finish
  the in-flight request, and exit 0 (verified).
- Endpoints: `POST /generate`, `POST /reset`, `GET /health`, `POST /shutdown`.

## Measurements (20 requests, 16 new tokens each, short prompts, mpl=64, BEST_PERF)

| Arrangement | Throughput | p50 latency | p95 latency | Errors |
|---|---|---|---|---|
| 1 pipeline, sequential | **2.50 req/s (40 tok/s)** | 0.50 s | 0.57 s | 0 |
| 1 pipeline, 4 concurrent clients | **2.52 req/s (40 tok/s)** | 1.97 s | 2.00 s | 0 |
| 1 pipeline, 8 concurrent clients | 2.52 req/s (40 tok/s) | 3.95 s | 4.94 s | 0 |
| 2 pipelines (round-robin), sequential | 1.94 req/s (31 tok/s) | 0.49 s | 0.83 s | 0 |
| 2 pipelines, 4 concurrent clients | 1.97 req/s (31 tok/s) | 2.00 s | 2.54 s | 0 |

Findings:

1. **Concurrency does not increase throughput** on one pipeline (2.50 → 2.52 req/s,
   within noise): requests serialize on the worker; per-client latency grows linearly
   with client count (4 clients → 4× p50), exactly as a FIFO queue predicts.
2. **Two pipelines are ~22% slower than one** (1.94 vs 2.50 req/s). Both compiled
   models contend for the single NPU; per-request latency is unchanged, so the
   second pipeline buys nothing and costs NPU memory and a second 1.1 s load.
   (First two-worker measurements were taken with a stray idle server loaded; the
   clean re-run — nothing else running — confirmed the result.)
3. Host RSS for the two-pipeline process is only ~411 MB — the compiled blobs live in
   NPU device memory, not host heap (single-pipeline processes measured 1.2–1.8 GB
   peak RSS in `benchmark.py`, which includes Python-side tensors).
4. Session-history growth is the practical capacity limit: at mpl=64 a session can
   hold ~3 short exchanges before `prompt_too_long`. Serving real conversations needs
   mpl=256 (TTFT 266 ms) or client-side history truncation.

## Recommendation

**One pipeline + FIFO queue** (the default `--workers 1`). It is the fastest measured
arrangement, the simplest, and the only one where error semantics are fully
predictable. Scale by queueing, not by replicating pipelines. Conventional batching is
not available for this pipeline type in this build and should not be assumed for
future builds without re-measurement.

## Reproduce

```bash
cd flossware-thinkpad-p1-gen8 && source scripts/env.sh
python3 serving/server.py --port 8931 --workers 1 &
python3 serving/bench_serving.py --mode sequential --n 20
python3 serving/bench_serving.py --mode concurrent --n 20 --clients 4
curl -s -X POST localhost:8931/generate -d \
  '{"session":"s1","prompt":"The capital of France is","max_new_tokens":16}'
curl -s -X POST localhost:8931/shutdown
```
