# Sourceable environment for Intel NPU inference with OpenVINO GenAI NPUW
# on the ThinkPad P1 Gen 8 (Intel Core Ultra 7 265H, NPU platform 3720).
#
# Configuration:
#   FLOSSWARE_AI_ROOT  root of your local OpenVINO/GenAI installation
#                      (default: ~/AI). See docs/environment.md for the
#                      expected layout underneath it.
#   NPU_MODEL_DIR      model directory              (default: $FLOSSWARE_AI_ROOT/models/openvino/qwen3-0.6b-int4)
#   NPUW_CACHE_DIR     NPUW compiled-model cache    (default: $FLOSSWARE_AI_ROOT/npuw-cache)
#
# All three can be overridden before or after sourcing.
# shellcheck shell=bash

export FLOSSWARE_AI_ROOT="${FLOSSWARE_AI_ROOT:-$HOME/AI}"

# GenAI python package + OpenVINO wheel (must be built from the same source tree;
# see docs/environment.md for the exact revisions used for the published results).
export PYTHONPATH="${FLOSSWARE_AI_ROOT}/install/openvino-genai-2026.5/python:${FLOSSWARE_AI_ROOT}/install/openvino-2026.5-python${PYTHONPATH:+:${PYTHONPATH}}"

# NOTE: the GenAI shared library lives under runtime/lib/intel64/ -- NOT runtime/lib/.
# tbb and the NPU compiler libs are expected inside the wheel's openvino/libs/
# (the wheel omits libopenvino_intel_npu_compiler*.so; symlink them from the
# source build -- see docs/environment.md, "Mandatory workarounds").
export LD_LIBRARY_PATH="${FLOSSWARE_AI_ROOT}/install/openvino-2026.5-python/openvino/libs:${FLOSSWARE_AI_ROOT}/install/openvino-genai-2026.5/runtime/lib/intel64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# MANDATORY on NPU platform 3720 (Arrow Lake-H) on Linux: OpenVINO defaults this
# platform to the driver NPU compiler, which fails model-independently
# (ZE_RESULT_ERROR_UNKNOWN 0x7ffffffe; see docs/driver-compiler.md and
# openvinotoolkit/openvino#38738). The plugin compiler works.
export IE_NPU_COMPILER_TYPE=PLUGIN

export NPU_MODEL_DIR="${NPU_MODEL_DIR:-${FLOSSWARE_AI_ROOT}/models/openvino/qwen3-0.6b-int4}"
export NPUW_CACHE_DIR="${NPUW_CACHE_DIR:-${FLOSSWARE_AI_ROOT}/npuw-cache}"
