#!/usr/bin/env bash
# 100% automated Thompson eval over the local fleet + free gateway models.
#
# Local models are served ONE AT A TIME (the iGPU cannot hold 14 models);
# the loop starts each server, waits for health, runs the fixtures, stops it.
#
# Usage (repo root, scripts/env.sh sourced):
#   bash eval/run_local_fleet.sh [--gateway-too]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

PORT=${EVAL_PORT:-8991}
OUT=eval/results/outcomes.jsonl
mkdir -p eval/results
FAILED=0

# cleanup + signal handling: bash resumes the script after a trap handler
# unless the handler exits, so INT/TERM must exit explicitly (otherwise
# Ctrl-C records a bogus failure and starts the NEXT model's server).
cleanup() { [ -n "$SRV" ] && kill -TERM "$SRV" 2>/dev/null; }
SRV=""
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

MODELS=(
  qwen3-0.6b-int4-official
  qwen3-1.7b-int4-official
  qwen2.5-coder-3b-int4-ov
  qwen3-4b-int4-official
  qwen2.5-coder-7b-instruct-int4-ov
  mistral-7b-instruct-v0.3-int4-ov
  qwen3-8b-int4-official
  llama-3.1-8b-instruct-int4-ov
  starcoder2-7b-int4-ov
  deepseek-r1-distill-qwen-14b-int4-ov
  qwen3-14b-int4-official
  qwen3-30b-a3b-int4-official
  qwen3-coder-30b-a3b-instruct-int4-ov
)

for M in "${MODELS[@]}"; do
  echo "=== serving $M on :$PORT ==="
  NPU_MODEL_DIR="$HOME/.FlossWare/models/llm/$M" \
    python3 serving/server.py --device GPU --port "$PORT" > "eval/results/server-$M.log" 2>&1 &
  SRV=$!

  # wait for health (up to 120s — big models load slowly), bailing out
  # early if the server process dies (crash at startup, port already held)
  READY=0
  for _ in $(seq 1 120); do
    kill -0 $SRV 2>/dev/null || break
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok": *true'; then
      READY=1; break
    fi
    sleep 1
  done

  if [ "$READY" = 1 ]; then
    python3 eval/run_eval.py --feed --local-only --out "$OUT" \
      --models <(echo "{\"local\": [{\"name\": \"$M\", \"kind\": \"local\", \"endpoint\": \"http://127.0.0.1:$PORT\"}]}") \
      || { echo "EVAL FAILED for $M"
           echo "{\"ts\": \"$(date -Is)\", \"model\": \"$M\", \"status\": \"eval_failed\"}" >> "$OUT"
           FAILED=1; }
  else
    echo "SERVER FAILED to become healthy: $M (see eval/results/server-$M.log)"
    echo "{\"ts\": \"$(date -Is)\", \"model\": \"$M\", \"status\": \"server_failed\"}" >> "$OUT"
    FAILED=1
  fi

  kill -TERM $SRV 2>/dev/null
  # escalate to SIGKILL if the server stalls on TERM (a hung wait would
  # block the whole fleet run)
  for _ in $(seq 1 10); do
    kill -0 $SRV 2>/dev/null || break
    sleep 1
  done
  kill -9 $SRV 2>/dev/null
  wait $SRV 2>/dev/null
  SRV=""
done

if [ "${1:-}" = "--gateway-too" ]; then
  echo "=== gateway (free) models ==="
  python3 eval/run_eval.py --feed --gateway-only --out "$OUT" \
    || { echo "GATEWAY EVAL FAILED" >&2; FAILED=1; }
fi

echo "=== ALL DONE — outcomes in $OUT ==="
exit $FAILED
