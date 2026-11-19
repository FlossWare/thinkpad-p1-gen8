# NPU Inference Environment — Verified State (2026-11-19)

All facts below were re-verified on the machine on 2026-11-19 after reboot.

## Hardware

| Item | Value |
|---|---|
| Machine | Lenovo ThinkPad P1 Gen 8 |
| CPU | Intel Core Ultra 7 265H (Arrow Lake-H) |
| NPU | PCI `8086:7d1d` at `00:0b.0`, rev 05, "Core Ultra 200H/200V Series Processors NPU" |
| NPU device node | `/dev/accel0` |
| NPU platform | 3720 (Arrow Lake) |

## OS / driver stack (rpm-verified)

| Package | Version |
|---|---|
| OS | Fedora 44, kernel `7.3.4-200.fc44.x86_64` |
| Python | 3.14.10 |
| intel-npu-driver | 1.32.0-1.fc44 |
| oneapi-level-zero | 1.33.1-3.fc44 |
| intel-level-zero | 26.35.39758.21-2.fc44 |
| intel-npu-compiler (system) | 2025.1.0-14.fc44 |
| intel-npu-firmware | 20261019-1.fc44 |

## Software stack

| Component | Identity | Location |
|---|---|---|
| OpenVINO | source build 2026.5.0, commit `88f8c6bf7c` | `~/AI/src/openvino` |
| OpenVINO Python wheel (built from same tree) | 2026.5.0 | `~/AI/install/openvino-2026.5-python` |
| OpenVINO GenAI | `releases/2026/5` @ `2de7224b`, built against the source OV tree | source `~/AI/src/openvino.genai`, install `~/AI/install/openvino-genai-2026.5` |
| Model | Qwen3-0.6B INT4, OpenVINO IR, stateful (56 KV-cache variables) | `~/AI/models/openvino/qwen3-0.6b-int4` |
| NPUW compiled-model cache | | `~/AI/npuw-cache` |

Pip openvino 2026.2 (`~/.local`) and system openvino 2025.1 (`/usr/lib64`) exist but are **not** used by the NPU stack.

## Mandatory workarounds (do not remove)

1. `IE_NPU_COMPILER_TYPE=PLUGIN` — the **driver** NPU compiler fails model-independently
   on this Arrow Lake machine (`ZE_RESULT_ERROR_UNKNOWN 0x7ffffffe`, even on trivial Relu).
   Platform 3720 defaults to the driver compiler; PLUGIN is only the default for
   NPU4000/5010/5020/6010 (`compiler_adapter_factory.cpp`).
2. The OV wheel omits `libopenvino_intel_npu_compiler.so` /
   `libopenvino_intel_npu_compiler_loader.so`. Both are symlinked from the source build
   into `~/AI/install/openvino-2026.5-python/openvino/libs/`. Without them the NPUW LLM
   path dies with `Unsupported configuration key: NPU_COMPILER_VERSION`.

## Exact working launch environment

See `scripts/env.sh` (sourceable) and `scripts/run-npuw.sh`. Effective values:

```bash
export PYTHONPATH=$HOME/AI/install/openvino-genai-2026.5/python:$HOME/AI/install/openvino-2026.5-python
export LD_LIBRARY_PATH=$HOME/AI/install/openvino-2026.5-python/openvino/libs:$HOME/AI/install/openvino-genai-2026.5/runtime/lib/intel64
export IE_NPU_COMPILER_TYPE=PLUGIN
```

Note: the GenAI shared library lives under `runtime/lib/intel64/`, **not** `runtime/lib/`.
tbb and both NPU compiler libs are already inside the wheel's `openvino/libs/`.

## Working paths and baselines

### Path A (primary): OpenVINO GenAI NPUW

Launch script: `scripts/run-npuw.sh` (the original acceptance script lives outside this repo). Baseline config:
`MAX_PROMPT_LEN=256, MIN_RESPONSE_LEN=64, CACHE_DIR=~/AI/npuw-cache`.

Measured 2026-11-19 (post-reboot re-verification):
- cold load+compile: 6.9 s (2026-11-18); warm load (cache hit): **1.2 s**
- decode: **11.4–15.4 tok/s** (32-token greedy runs), original baseline 13 tok/s
- peak RSS: 1216 MB (1216–1650 MB range across sessions)
- NPU execution confirmed via `/proc/self/maps`: `libopenvino_intel_npu_plugin.so`,
  `libopenvino_intel_npu_compiler_loader.so`, `libze_intel_npu.so.1.32.0`

Caveats learned:
- GenAI python scripts **must** use a `if __name__ == '__main__':` guard — GenAI's
  forkserver re-imports the main module, duplicating all top-level work.
- `LLMPipeline.get_perf_metrics()` does **not exist** in this build — TTFT/perf must be
  measured via a streamer callback.
- The `config=` dict parameter is deprecated; pass properties as kwargs.

### Path B (fallback): manual static blob

Artifacts in `~/AI/ovfix-static-blob/` (copied from ephemeral `/tmp/ovfix`) and
`~/AI/models/openvino/` (`qwen3-static-p0/`, `qwen3-static-p128/`, `ov_static_gen.py`).
40 s compile, 1.19 GB blob, runs on NPU.0, last-token argmax matches CPU, logit cosine
0.9916 vs CPU. One shape bucket per blob; `beam_idx` removed by the transformation.
**Do not modify.**

## NPUW LLM options actually supported by this build (source-verified)

From `openvino/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp` and
`al/include/intel_npu/config/`:

- `NPUW_LLM_PREFILL_HINT`: `DYNAMIC` (default) | `STATIC`. Chunked prefill activates
  only when `DYNAMIC` **and** `NPUW_LLM_PREFILL_CHUNK_SIZE < MAX_PROMPT_LEN`
  (chunk size must be a power of 2, default 1024). With `MAX_PROMPT_LEN=256`,
  chunking is **inactive** — prefill is static either way.
- `NPUW_LLM_GENERATE_HINT`: `FAST_COMPILE` (default, sets `NPUW_UNFOLD_IREQS=YES`) |
  `BEST_PERF` (sets `NPUW_ONLINE_PIPELINE=NONE`). Only valid with the default
  generate config.
- `MAX_PROMPT_LEN` / `MIN_RESPONSE_LEN` (GenAI level) map to `NPUW_LLM_MAX_PROMPT_LEN`
  / `NPUW_LLM_MIN_RESPONSE_LEN`; both are aligned **up to multiples of 64** by the plugin.
- `NPUW_LLM_PREFILL_ATTENTION_HINT`: auto-selected to HFA on this stack;
  `NPUW_ATTN_HFA_FUSED=YES` auto-set.
- Prefix caching exists (`NPUW_LLM_ENABLE_PREFIX_CACHING`,
  `NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE`) but only under chunked prefill.

## Proven root cause of the original compile failure (do not revisit)

The stateful IR has dynamic KV-cache shapes (`[?,8,?,128]`, 56 variables). Input shape
specialization does not specialize `ReadValue` variable shapes; the NPU compiler's
ReadValue parser (`npu_compiler ... /IE.cpp:4645`, commit `56e4d446c`) calls
`get_shape()` unguarded → `PartialShape::to_shape` throws. Not a version mismatch.
`NPU_DYNAMIC_SHAPE_TO_STATIC` does not fix it.
