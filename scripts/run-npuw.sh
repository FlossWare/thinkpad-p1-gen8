#!/usr/bin/env bash
# Reproducible launch of the best measured NPUW configuration for Qwen3-0.6B INT4
# on the ThinkPad P1 Gen 8. See docs/npuw-tuning.md for the experiment table
# behind these values.
#
# Usage (from the repository root or from scripts/):
#   scripts/run-npuw.sh                  # one-shot generation demo + evidence
#   scripts/run-npuw.sh "your prompt"    # custom prompt
#   scripts/run-npuw.sh --benchmark      # run benchmarks/benchmark.py with the best config
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/env.sh

# Best measured configuration (2026-11-19):
#   GENERATE_HINT=BEST_PERF  -> decode 41 tok/s (+25% vs FAST_COMPILE)
#   MAX_PROMPT_LEN=64        -> TTFT 164 ms for short prompts (-38% vs 256)
#   MIN_RESPONSE_LEN=64      -> plugin-aligned floor; sustains 128-token gens
# First cold compile ~20 s; subsequent loads ~1.1 s from $NPUW_CACHE_DIR.
export NPUW_BEST_CONFIG="MAX_PROMPT_LEN=64 MIN_RESPONSE_LEN=64 GENERATE_HINT=BEST_PERF"

if [[ "${1:-}" == "--benchmark" ]]; then
    shift
    exec python3 benchmarks/benchmark.py --label best-config "$@" \
        --max-prompt-len 64 --set GENERATE_HINT=BEST_PERF
fi

PROMPT="${1:-The capital of France is}"

# NOTE: the Python payload runs from a temp file, not a heredoc pipe: GenAI's
# forkserver re-imports the main module, and `python3 -` with a heredoc makes
# that re-import fail noisily on '<stdin>' (harmless, but confusing).
TMP_PY="$(mktemp /tmp/run-npuw-XXXXXX.py)"
trap 'rm -f "$TMP_PY"' EXIT
cat > "$TMP_PY" <<'PY'
import sys, time

if __name__ != '__main__':
    raise SystemExit(0)  # GenAI forkserver re-import guard

import openvino_genai as genai

prompt = sys.argv[1]
t0 = time.perf_counter()
pipe = genai.LLMPipeline(
    __import__('os').environ['NPU_MODEL_DIR'], device='NPU',
    MAX_PROMPT_LEN=64, MIN_RESPONSE_LEN=64,
    GENERATE_HINT='BEST_PERF',
    CACHE_DIR=__import__('os').environ['NPUW_CACHE_DIR'],
)
print(f'[load] {time.perf_counter() - t0:.1f}s')

times = []
def streamer(chunk):
    times.append(time.perf_counter())
    return None

t0 = time.perf_counter()
out = pipe.generate(prompt, max_new_tokens=32, streamer=streamer, do_sample=False)
wall = time.perf_counter() - t0
if times:
    ttft = times[0] - t0
    dec = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else float('nan')
    print(f'[perf] ttft={ttft*1000:.0f}ms decode={dec:.1f} tok/s wall={wall:.2f}s')
print(f'[output] {out}')

with open('/proc/self/maps') as f:
    libs = {l.split()[-1] for l in f if 'npu' in l.lower()}
ok = any('libopenvino_intel_npu_plugin' in x for x in libs)
print(f'[evidence] NPU plugin mapped: {ok}')
sys.exit(0 if ok else 1)
PY
python3 "$TMP_PY" "$PROMPT"
