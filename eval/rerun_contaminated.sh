#!/usr/bin/env bash
# Rerun the 4 GPU-contaminated models under clean conditions, feeding Thompson.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
OUT=eval/results/outcomes.jsonl

for M in llama-3.1-8b-instruct-int4-ov mistral-7b-instruct-v0.3-int4-ov qwen3-8b-int4-official qwen3-4b-int4-official; do
  echo "=== RERUN $M ==="
  NPU_MODEL_DIR="$HOME/.FlossWare/models/llm/$M" python3 serving/server.py --device GPU --port 8991 > "eval/results/server-rerun-$M.log" 2>&1 &
  SRV=$!
  READY=0
  for _ in $(seq 1 90); do
    curl -s -m 2 "http://127.0.0.1:8991/health" | grep -q '"ok": *true' && READY=1 && break
    sleep 1
  done
  if [ "$READY" = 1 ]; then
    printf '{"local": [{"name": "%s-rerun", "kind": "local", "endpoint": "http://127.0.0.1:8991"}]}' "$M" > /tmp/rerun-pool.json
    python3 eval/run_eval.py --feed --local-only --out "$OUT" --models /tmp/rerun-pool.json 2>&1 | tail -3
  else
    echo "SERVER FAILED: $M"
  fi
  kill -TERM $SRV 2>/dev/null
  sleep 8
  kill -9 $SRV 2>/dev/null
  wait $SRV 2>/dev/null
done
echo "RERUNS DONE"
