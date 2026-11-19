# Lenovo ThinkPad P1 Gen 8 — Hardware Platform Notes

This is a hardware-platform engineering repository: the machine is the stable
reference platform, and projects measure what its accelerators can actually do.
This page records the verified platform facts. All measurements elsewhere in this
repository were taken on this exact machine.

## Verified platform (2026-11-19)

| Component | Detail |
|---|---|
| Machine | Lenovo ThinkPad P1 Gen 8 |
| CPU | Intel Core Ultra 7 265H (Arrow Lake-H, 16 cores) |
| NPU | Intel AI Boost, integrated; PCI `8086:7d1d` at `00:0b.0` (rev 05); device node `/dev/accel0`; NPU platform **3720** |
| iGPU | Intel Arc Graphics (reports as "Intel(R) Arc(TM) Graphics (iGPU)"); PCI `8086:7d51` at `00:02.0` (rev 03, Arrow Lake-P Arc Pro 130T/140T class); media stack libva-intel-media-driver 26.3.5, Level Zero 1.33.1 |
| OS | Fedora 44, kernel `7.3.4-200.fc44.x86_64` |
| NPU driver | intel-npu-driver 1.32.0 (Fedora package) |
| Level Zero | oneapi-level-zero 1.33.1 / intel-level-zero 26.35.39758.21 (Fedora packages; driver reports 26.35.39758.17) |
| NPU firmware | intel-npu-firmware 20261019-1.fc44 |

## Platform quirks relevant to AI engineering (verified on this machine)

1. **NPU compiler selection**: platform 3720 defaults to the *driver* NPU compiler on
   Linux, which fails model-independently (`ZE_RESULT_ERROR_UNKNOWN 0x7ffffffe`, even
   on a trivial Relu graph). `IE_NPU_COMPILER_TYPE=PLUGIN` is mandatory. See
   [docs/driver-compiler.md](../docs/driver-compiler.md) and
   [openvinotoolkit/openvino#38738](https://github.com/openvinotoolkit/openvino/issues/38738).
2. **OpenVINO wheel omissions**: the 2026.5 wheel does not ship
   `libopenvino_intel_npu_compiler.so` / `..._loader.so`; they must be symlinked from
   the source build into the wheel's `libs/` directory. See
   [docs/environment.md](../docs/environment.md).
3. **Stateful IR with dynamic KV-cache shapes** fails to compile on the NPU compiler's
   ReadValue parser; NPUW (GenAI) handles it, the manual path needs a
   stateful→stateless + static-shape conversion. See
   [docs/environment.md](../docs/environment.md) and
   [openvinotoolkit/openvino#39184](https://github.com/openvinotoolkit/openvino/issues/39184).

## Tested on this machine

- Intel NPU via OpenVINO 2026.5 / GenAI `releases/2026/5` NPUW (this repository's
  first project: Qwen3-0.6B INT4).

## Not yet tested / future work

- Intel GPU (Arc iGPU) inference — untested.
- CPU-plugin throughput characterization beyond the correctness reference role.
- Other models (larger LLMs, embedding models) and other runtimes
  (llama.cpp/GGUF, IPEX-LLM) — unsupported claims are intentionally absent.
