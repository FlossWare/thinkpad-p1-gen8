#!/usr/bin/env bash
# Rerun ollama models that failed under the CPU-bound binary, now on the
# Vulkan-enabled v0.40.17 build. Run AFTER finish_ollama.sh has completed.
#
# Covers:
#   - qwen3:1.7b, qwen3:4b  — CPU timeouts (thinking models, 300s limit)
#   - gemma3:4b             — 2 fixture failures on CPU
#   - qwen3.5:2b, gemma4:e2b — arch-fail models, IF finish chain tests passed
#
# Records go to the ledger under CANONICAL names (no -rerun suffix): the CPU
# failures were never fed to Thompson (only status=ok records feed), so the
# canonical arm simply receives its missing outcomes.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=eval/results/outcomes.jsonl

# 0. refuse to run unless ollama is actually offloading to the GPU now.
# Bound the journal query to the current service instance: a stale
# "offloaded N/M" line from a previous (CPU-only) run must not pass the gate.
SINCE=$(systemctl --user show -p ActiveEnterTimestamp --value ollama.service)
GPU_LAYERS=$(journalctl --user -u ollama.service --since "$SINCE" --no-pager 2>/dev/null \
  | grep -oE "offloaded [0-9]+/[0-9]+ layers to GPU" | tail -1)
echo "latest offload line: ${GPU_LAYERS:-none}"
if [ -z "$GPU_LAYERS" ] || echo "$GPU_LAYERS" | grep -q "offloaded 0/"; then
  echo "REFUSING: ollama still not using GPU — fix Vulkan before rerunning"
  exit 1
fi

MODELS='["qwen3:1.7b", "qwen3:4b", "gemma3:4b"]'

# add arch-fail models only if they passed the finish-chain smoke test
# (-xF: exact full-line match; "." in model names is not a wildcard here)
for M in "qwen3.5:2b" "gemma4:e2b"; do
  if grep -qxF "$M: OK" /tmp/ollama-finish.log 2>/dev/null; then
    MODELS=$(echo "$MODELS" | python3 -c "import json,sys; l=json.load(sys.stdin); print(json.dumps(l + ['$M']))")
  fi
done
echo "rerunning: $MODELS"

echo "$MODELS" | python3 -c "
import json, sys
pool = {'ollama': [{'name': 'ollama/' + m, 'kind': 'ollama', 'model_id': m}
                   for m in json.load(sys.stdin)]}
print(json.dumps(pool))" > /tmp/ollama-rerun-pool.json

python3 eval/run_eval.py --feed --models /tmp/ollama-rerun-pool.json --out "$OUT" \
  || { echo "RERUN FAILED: eval driver error" >&2; exit 1; }
echo "RERUN DONE"
