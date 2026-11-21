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

  # wait for health (up to 120s — big models load slowly)
  READY=0
  for _ in $(seq 1 120); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok": *true'; then
      READY=1; break
    fi
    sleep 1
  done

  if [ "$READY" = 1 ]; then
    python3 eval/run_eval.py --local-only --out "$OUT" \
      --models <(echo "{\"local\": [{\"name\": \"$M\", \"kind\": \"local\", \"endpoint\": \"http://127.0.0.1:$PORT\"}]}") \
      || echo "EVAL FAILED for $M"
  else
    echo "SERVER FAILED to become healthy: $M (see eval/results/server-$M.log)"
    echo "{\"ts\": \"$(date -Is)\", \"model\": \"$M\", \"status\": \"server_failed\"}" >> "$OUT"
  fi

  kill -TERM $SRV 2>/dev/null
  wait $SRV 2>/dev/null
done

if [ "${1:-}" = "--gateway-too" ]; then
  echo "=== gateway (free) models ==="
  python3 eval/run_eval.py --gateway-only --out "$OUT"
fi

echo "=== ALL DONE — outcomes in $OUT ==="
