# thinkpad-p1-gen8

Reproducible AI engineering experiments and measured capabilities on the
**Lenovo ThinkPad P1 Gen 8** (Intel Core Ultra 7 265H, Intel NPU).

This is a hardware-platform engineering repository, not a model benchmark collection.
The machine is the stable reference platform; each project documents what was built,
how it was measured, and what the hardware and software stack actually delivered —
including the failures and workarounds.

**Project 1: Intel NPU inference with OpenVINO GenAI NPUW and Qwen3-0.6B INT4.**
Status: complete (2026-11-19). Everything below refers to that project unless noted.

> All numbers on this page are measurements from **one specific machine** (see
> [hardware/thinkpad-p1-gen8.md](hardware/thinkpad-p1-gen8.md)). They are not claims
> about Intel NPU performance in general.

## What works today

- **NPUW path (primary)**: OpenVINO GenAI `LLMPipeline` on `device='NPU'`, running the
  original stateful Qwen3-0.6B INT4 OpenVINO IR end-to-end on the NPU, tuned to
  **41 tokens/s decode** and **~170 ms time-to-first-token** for short prompts.
- **Correctness validated** against the same build running on CPU: 9 of 10 battery
  tests token-identical; the one divergence is a near-tie argmax flip from FP16
  precision (details and measured drift in
  [docs/correctness.md](docs/correctness.md)).
- **Serving prototype**: single persistent pipeline + FIFO queue with request
  isolation, explicit sessions, and graceful shutdown
  ([serving/](serving/), [docs/serving.md](docs/serving.md)).
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

> **Pending re-measurement (issue #1).** The decode tok/s figures below were
> measured with the streamer callback count as the numerator, which is not a
> generated-token count. `benchmarks/benchmark.py` now counts generated token IDs;
> these figures will be replaced after the re-runs listed in
> [docs/npuw-tuning.md](docs/npuw-tuning.md). TTFT and compile times are unaffected.

| Configuration | TTFT | Decode | Cold compile |
|---|---|---|---|
| Baseline (`mpl=256`, default hints) | 266 ms | 33 tok/s | 6.2 s |
| `GENERATE_HINT=BEST_PERF` | 262 ms | 41 tok/s | 19.8 s |
| `MAX_PROMPT_LEN=64` + BEST_PERF (recommended) | **164–170 ms** | **41 tok/s** | 19.8 s (then ~1.1 s warm load) |

- The previously circulated "13 tok/s" figure was a measurement artifact (TTFT folded
  into the decode denominator); the corrected baseline is 33 tok/s.
- Chunked prefill was tested and was **slower** than static prefill for this
  workload (476 vs 691 tok/s prefill on a 349-token prompt) — not recommended here.

### Correctness ([docs/correctness.md](docs/correctness.md))

- 9/10 battery tests token-identical CPU vs NPU (prompts 5–193 tokens, generations
  32–128 tokens). The single divergence: a near-tie argmax flip on a repeated-token
  prompt, semantically equivalent output — the expected FP16 signature, not cache
  corruption.
- NPUW KV cache is **FP16** — verified by source inspection
  (`choose_kv_cache_storage_type()` default; `NPUW_LLM_OPTIMIZE_FP8=false`), not
  inferred from the INT4 weights. Activations are converted bf16→f16 model-wide.
- Per-step top-1-logprob drift: mean |ΔlogP| ≤ 0.04, max ≤ 0.15 per step; cosine similarity over the 16-step top-1 logprob vectors (CPU vs NPU) ≥ 0.9993 — **not** full-vocabulary logit cosine (see [docs/correctness.md](docs/correctness.md)). The only logit-level comparison (cosine of the last-position logit vector, 0.9916, measured 2026-11-18) is on the separate manual static-blob fallback path, not NPUW.
- Prompts longer than `MAX_PROMPT_LEN` are cleanly rejected (no silent truncation).

### Serving ([docs/serving.md](docs/serving.md))

- Recommended architecture: **one persistent pipeline + FIFO queue** (simplest,
  deterministic error semantics). Concurrent requests on one pipeline serialize with
  no throughput gain.
- The earlier "two pipelines are 22% slower" claim is **withdrawn**: it compared
  single-pipeline runs that each had 4/20 failed requests (counted as completed)
  against clean two-pipeline runs. Counting successful requests only, the measured
  arrangements are within ~3% of each other; a clean, comparable re-run is pending
  (issue #2). See [docs/serving.md](docs/serving.md).

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
| [hardware/](hardware/) | platform notes for the ThinkPad P1 Gen 8 |

## Evidence classification

- **Verified**: everything in the results tables above (subject to the pending
  re-measurements flagged there), with raw JSON in the `results/` directories. Device
  evidence per run is the requested device plus the `/proc/self/maps` NPU-plugin
  check — this shows the NPU plugin was loaded, not that every measured inference
  executed on the NPU; no per-inference execution telemetry is available in this
  build.
- **Observations** (single-machine, single-session): RSS ranges, TTFT variance
  between runs, NPU clock warm-up effects on first generation.
- **Hypotheses, not tested**: behavior on other NPU platforms (Meteor Lake, Lunar
  Lake), other driver versions, larger models. Explicitly untested — do not treat
  as capabilities.
- **Unsupported/untested configurations**: GPU inference, batched generation,
  FP8 KV-cache optimization (`NPUW_LLM_OPTIMIZE_FP8`), speculative decoding.

## Known limitations and future work

- One shape bucket per compiled NPUW config; `MAX_PROMPT_LEN` must be sized to the
  workload (aligned up to multiples of 64 by the plugin).
- No per-step logits through the GenAI Python API in this build; correctness
  comparisons use greedy token IDs and top-1 logprobs (see
  [docs/correctness.md](docs/correctness.md) for methodology and limits).
- Future work: GPU (Arc iGPU) inference characterization, embedding models on NPU,
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
