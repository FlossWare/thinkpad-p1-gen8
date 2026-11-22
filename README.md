# thinkpad-p1-gen8

Reproducible AI engineering experiments and measured capabilities on the
**Lenovo ThinkPad P1 Gen 8** (Intel Core Ultra 7 265H, Intel NPU).

This is a hardware-platform engineering repository, not a model benchmark collection.
The machine is the stable reference platform; each project documents what was built,
how it was measured, and what the hardware and software stack actually delivered —
including the failures and workarounds.

**Project 1: Intel NPU inference with OpenVINO GenAI NPUW and Qwen3-0.6B INT4.**
Status: complete (2026-11-19). Everything below refers to that project unless noted.

**Project 2: fully-automated model measurement campaign** (`eval/`).
Status: complete (2026-11-22). Every reachable model tier — 13 local OpenVINO
models, 10 ollama models, 3 free gateway models, 3 Claude models, 28 paid
gateway models — was measured on the same 11 seeded-defect code-review
fixtures with computed scoring (term-match against ground truth; no arbiter,
no human in the loop), recording ~4,600 outcomes to a live Thompson-sampling
service. Findings: classic-defect capability is saturated (nearly everything
from 0.6B up scores 11/11); the real routing signal is time-budget failures
of small thinking models, runtime choice (same weights: 3/11 as an OpenVINO
IR vs 11/11 through ollama), and economics (qwen3-coder-30b-a3b matches
premium-model catch rate at ~9 s and $0 per request). All raw outcomes:
[`eval/results/outcomes.jsonl`](eval/results/outcomes.jsonl).

One hardware-relevant result: **stock ollama builds are CUDA/ROCm-only and
cannot use the Arc iGPU** — everything silently runs on CPU until ollama
0.40.17 (Vulkan backend) is installed as a full binary+runner tree with
`OLLAMA_VULKAN=1` and `OLLAMA_IGPU_ENABLE=1`. Verified by journal
`offloaded N/M layers to GPU` lines (37/37 for qwen3:4b, ~13 s/response vs
5-minute CPU timeouts). `eval/rerun_ollama_gpu.sh` refuses to run unless
GPU offload is evidenced in the journal.

`eval/` also contains fixture miners that turn real code history into
evaluation fixtures: `mine_own_repos.py` (git-log mining: a small
defect-named follow-up commit labels the change a reviewer should have
caught; aged untouched changes are clean false-positive baselines) and
`mine_public_github.py` (human review comments on merged PRs in public
repos as labeled findings). Mined fixtures live under the local data store,
not in this repository.

> All numbers on this page are measurements from **one specific machine** (see
> [hardware/thinkpad-p1-gen8.md](hardware/thinkpad-p1-gen8.md)). They are not claims
> about Intel NPU performance in general.

## What works today

- **NPUW path (primary)**: OpenVINO GenAI `LLMPipeline` on `device='NPU'`, running the
  original stateful Qwen3-0.6B INT4 OpenVINO IR end-to-end on the NPU, tuned to
  **45 tokens/s decode** and **~190 ms time-to-first-token** for short prompts.
- **Correctness validated** against the same build running on CPU: 8 of 10 battery
  tests token-identical; one diverges by a near-tie argmax flip from FP16 precision,
  and one matches on all 127 common tokens but the NPU stopped one token short of 128 (details and measured drift in
  [docs/correctness.md](docs/correctness.md)).
- **Serving prototype**: single persistent pipeline + FIFO queue with request
  isolation, explicit sessions, and graceful shutdown
  ([serving/](serving/), [docs/serving.md](docs/serving.md)); runs on NPU or
  the Arc iGPU (`--device GPU`, optional `--f32` exactness mode) and exposes a
  text-only Anthropic Messages-API shim at `POST /v1/messages`.
- **Local-model fleet (measured on the Arc iGPU)**: Qwen3-0.6B through
  Qwen3-30B-A3B — the 30B-A3B MoE runs at 28–30 tok/s (~4B-dense speed);
  full table in [docs/roadmap.md](docs/roadmap.md).
- **Manual static-blob fallback path** (stateful→stateless conversion + static shapes,
  compiled blob) — preserved on the engineering workstation, documented in
  [docs/environment.md](docs/environment.md); not required reading for reproduction.

## Tested hardware and software stack

| Layer | Verified version |
|---|---|
| Machine | Lenovo ThinkPad P1 Gen 8, Core Ultra 7 265H, NPU PCI `8086:7d1d`, platform 3720 |
| OS | Fedora 44, kernel `7.3.4-200.fc44.x86_64` |
| NPU driver / Level Zero | intel-npu-driver 1.32.0 / intel-level-zero 26.35.39758.21 (driver string 26.35.39758.17) |
| OpenVINO | 2026.5.0 source build, commit `88f8c6bf7c` (with Python bindings, `ENABLE_PYTHON=ON` + `ENABLE_WHEEL=ON`) |
| OpenVINO GenAI | `releases/2026/5` @ `2de7224b`, built against that OpenVINO tree |
| NPU compiler | plugin compiler via `IE_NPU_COMPILER_TYPE=PLUGIN` (mandatory — see below) |
| Model | Qwen3-0.6B INT4 OpenVINO IR (stateful, 56 KV-cache variables) |

Full details, install layout, and the two mandatory workarounds:
[docs/environment.md](docs/environment.md).

## Verified results (measured 2026-11-19)

### NPUW performance ([docs/npuw-tuning.md](docs/npuw-tuning.md))

TTFT and decode figures use the token-ID accounting (issue #1: throughput is computed
from generated token IDs, not streamer callback counts; re-measured 2026-11-19, rows
`re-*` in [`benchmarks/results/results.jsonl`](benchmarks/results/results.jsonl)). The
cold-compile column was not re-measured; it comes from the historical cold rows
(`baseline-cold`, `e1-bestperf-cold`, `e2-mp64-bestperf`), which token accounting does
not affect.

| Configuration | TTFT | Decode | Cold compile |
|---|---|---|---|
| Baseline (`mpl=256`, default hints) | 297 ms | 36.5 tok/s | 6.2 s |
| `GENERATE_HINT=BEST_PERF` | 317 ms | 42 tok/s | 19.8 s |
| `MAX_PROMPT_LEN=64` + BEST_PERF (recommended) | **189 ms** | **45 tok/s** | 19.8 s (then ~1.1 s warm load) |

- **Capacity rule:** the NPUW KV window is `MAX_PROMPT_LEN + MIN_RESPONSE_LEN` tokens,
  prompt and generated combined (128 for the recommended 64 + 64). If
  `prompt_tokens + max_new_tokens` exceeds it, the generation is **silently truncated**
  (prefix-exact, no error) — e.g. a 5-token prompt asking for 128 tokens gets 123.
  Raise `MIN_RESPONSE_LEN` for longer responses. Source-traced and confirmed on the NPU:
  [docs/npu-early-stop-investigation.md](docs/npu-early-stop-investigation.md).

- The previously circulated "13 tok/s" figure was a measurement artifact (TTFT folded
  into the decode denominator); a second correction (issue #1) replaced the streamer
  callback count with the generated-token count (31 callbacks for 32 tokens is typical).
- Chunked prefill was tested and was **slower** than static prefill for this
  workload (476 vs 691 tok/s prefill on a 289-token prompt; historical rows `e5*`) — not recommended here.

### Correctness ([docs/correctness.md](docs/correctness.md))

- 8/10 battery tests strictly token-identical CPU vs NPU (prompts 5–193 tokens,
  generations 32–128 tokens). The one token divergence: a near-tie argmax flip on a
  repeated-token prompt, semantically equivalent output — the expected FP16 signature,
  not cache corruption. The tenth (`near-limit-193t@128`) matches on all 127 common
  tokens, but the NPU generated 127 tokens vs the CPU's 128 (193 + 127 = 320 =
  `MAX_PROMPT_LEN` + `MIN_RESPONSE_LEN`) — expected KV-window capacity behavior,
  source-traced and confirmed by NPU boundary probes (issue #13).
- NPUW KV cache is **FP16** — verified by source inspection
  (`choose_kv_cache_storage_type()` default; `NPUW_LLM_OPTIMIZE_FP8=false`), not
  inferred from the INT4 weights. Activations are converted bf16→f16 model-wide.
- Per-step top-1-logprob drift: mean |ΔlogP| ≤ 0.04, max ≤ 0.15 per step; cosine similarity over the 16-step top-1 logprob vectors (CPU vs NPU) ≥ 0.9992 (raw minimum 0.99929) — **not** full-vocabulary logit cosine (see [docs/correctness.md](docs/correctness.md)). The only logit-level comparison (cosine of the last-position logit vector, 0.9916, measured 2026-11-18) is on the separate manual static-blob fallback path, not NPUW.
- Prompts longer than `MAX_PROMPT_LEN` are cleanly rejected (no silent truncation).

### Serving ([docs/serving.md](docs/serving.md))

- Recommended architecture: **one persistent pipeline + FIFO queue** (simplest,
  deterministic error semantics). Concurrent requests on one pipeline serialize with
  no throughput gain.
- The earlier "two pipelines are 22% slower" claim is **withdrawn**: it compared
  single-pipeline runs that each had 4/20 failed requests (counted as completed)
  against clean two-pipeline runs. The clean re-runs (rows `re-w*` in
  [`serving/results/serving-results.jsonl`](serving/results/serving-results.jsonl),
  20/20 successful in every run) replace that comparison: sequential throughput is a
  wash (1.93 vs 1.88 successful req/s for one vs two pipelines), and under 4 concurrent
  clients two pipelines are ~16% faster (2.17 vs 1.86 req/s). See
  [docs/serving.md](docs/serving.md).

### Driver compiler ([docs/driver-compiler.md](docs/driver-compiler.md))

- Platform 3720 defaults to the driver NPU compiler on Linux and fails
  model-independently; documented in
  [openvinotoolkit/openvino#38738](https://github.com/openvinotoolkit/openvino/issues/38738)
  (closed "not planned", no fix). No Intel driver release through v1.38.0 documents a
  fix. `IE_NPU_COMPILER_TYPE=PLUGIN` is the standing, verified workaround.

## Quick start

Requirements: a working OpenVINO 2026.5 / GenAI `releases/2026/5` installation and the
Qwen3-0.6B INT4 OpenVINO IR model (see
[docs/environment.md](docs/environment.md) for the exact layout and build flags).
Set `FLOSSWARE_AI_ROOT` if your installation root is not `~/AI`.

```bash
source scripts/env.sh        # sets PYTHONPATH, LD_LIBRARY_PATH, PLUGIN compiler
scripts/run-npuw.sh          # one-shot generation with the best config + NPU evidence
scripts/run-npuw.sh --benchmark --repeats 5   # performance benchmark
python3 validation/validate.py                # CPU-vs-NPU correctness battery
python3 serving/server.py --port 8931 &       # multi-session serving prototype
```

With the default `MAX_PROMPT_LEN=64`/`MIN_RESPONSE_LEN=64`, keep
`prompt_tokens + max_new_tokens ≤ 128`: beyond that NPUW truncates silently, and the
serving prototype rejects such requests with HTTP 400 `exceeds_kv_capacity`.

Two runtime gotchas discovered during the project (both bite silently):

- The GenAI shared library lives under `runtime/lib/intel64/`, **not**
  `runtime/lib/` — a wrong `LD_LIBRARY_PATH` surfaces as a confusing
  `Unsupported configuration key: NPU_COMPILER_VERSION` error.
- `python3 -I` (isolated mode) ignores `PYTHONPATH` — scripts that import
  `openvino_genai` must not be run with `-I`.

## Repository layout

| Path | Contents |
|---|---|
| [docs/](docs/) | engineering documentation (environment, tuning, correctness, serving, driver research) |
| [scripts/](scripts/) | `env.sh` (environment) and `run-npuw.sh` (best-config launch) |
| [benchmarks/](benchmarks/) | `benchmark.py` + raw results (`results/results.jsonl`) |
| [validation/](validation/) | `validate.py` + raw results (`results/validation-results.json`) |
| [serving/](serving/) | multi-session serving prototype + benchmark client + raw results |
| [eval/](eval/) | model measurement campaign: fixtures, eval driver, sweeps, fixture miners, outcome ledger |
| [hardware/](hardware/) | platform notes for the ThinkPad P1 Gen 8 |

## Tests and CI

Non-NPU unit tests run without OpenVINO, model weights, or NPU hardware:

```bash
python -m pytest serving/ benchmarks/ -q
```

This covers serving concurrency/queue/validation and benchmark token-accounting
regression tests (stdlib only plus `pytest`). A GitHub Actions workflow
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the same
tests on pushes to `main` and on pull requests (Python 3.14, `pytest` only, verbose
output so a failing test is named in the log; a failing test fails the check).
Hardware-dependent scripts (`benchmarks/benchmark.py`, `validation/validate.py`,
`validation/boundary_probe.py`, `scripts/run-npuw.sh`, a live `serving/server.py`) need
the NPU machine and are not part of CI.

## Evidence classification

- **Verified**: everything in the results tables above, with raw JSON in the `results/` directories. Device
  evidence per run is the requested device plus the `/proc/self/maps` NPU-plugin
  check — this shows the NPU plugin was loaded, not that every measured inference
  executed on the NPU; no per-inference execution telemetry is available in this
  build.
- **Observations** (single-machine, single-session): RSS ranges, TTFT variance
  between runs, NPU clock warm-up effects on first generation.
- **Hypotheses, not tested**: behavior on other NPU platforms (Meteor Lake, Lunar
  Lake), other driver versions, larger models. Explicitly untested — do not treat
  as capabilities.
- **Tested since M3 (2026-11-20)**: GPU (Arc iGPU) inference — the practical
  host for local models on this machine; see the fleet table in
  [docs/roadmap.md](docs/roadmap.md) and GPU serving in
  [docs/serving.md](docs/serving.md).
- **Unsupported/untested configurations**: batched generation,
  FP8 KV-cache optimization (`NPUW_LLM_OPTIMIZE_FP8`), speculative decoding,
  multi-model single-port serving.

## Known limitations and future work

- One shape bucket per compiled NPUW config; `MAX_PROMPT_LEN` must be sized to the
  workload (aligned up to multiples of 64 by the plugin).
- No per-step logits through the GenAI Python API in this build; correctness
  comparisons use greedy token IDs and top-1 logprobs (see
  [docs/correctness.md](docs/correctness.md) for methodology and limits).
- Future work: embedding/reranker benchmarks, untested fleet models, ensemble
  additional runtimes, re-testing against OpenVINO/driver releases that address
  the issues linked above.
- Prioritized post-Project-1 milestones, evidence plans, and backlog:
  [docs/roadmap.md](docs/roadmap.md).

## Third-party components

This repository contains only original FlossWare code and documentation. It does not
redistribute any third-party software, models, or binaries:

- [OpenVINO](https://github.com/openvinotoolkit/openvino) and
  [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) — Apache-2.0
  (source-built locally; not included here)
- [Qwen3-0.6B](https://huggingface.co/Qwen) — model weights under Alibaba's Qwen
  license (Apache-2.0 family); not included here
- Intel NPU driver, Level Zero, firmware — Intel system packages (Fedora)

## License

[MIT](LICENSE) — applies to the original code and documentation in this repository
only. Third-party components remain under their respective licenses.
