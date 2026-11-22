#!/usr/bin/env bash
# After the GPU sweep completes: upgrade ollama binary, test the 3 arch-fail
# models (qwen3.5:2b, gemma4:e2b, qwen3:4b), and if they work, sweep them too.
set -u
cd "$(dirname "$0")/.."

# 1. wait for the running sweep to finish
while pgrep -f "ollama_pool" > /dev/null; do sleep 20; done
echo "=== sweep finished; upgrading ollama binary ==="

# 2. swap the binary (unit points at /usr/bin/ollama; override via PATH env or direct)
cp ~/.local/bin/ollama-new ~/.local/bin/ollama
chmod +x ~/.local/bin/ollama
systemctl --user stop ollama.service
sed -i 's|ExecStart=/usr/bin/ollama serve|ExecStart=/home/sfloess/.local/bin/ollama serve|' ~/.config/systemd/user/ollama.service
systemctl --user daemon-reload
systemctl --user start ollama.service
sleep 8
systemctl --user is-active ollama.service || { echo "SERVICE FAILED after upgrade"; exit 1; }
~/.local/bin/ollama --version 2>&1 | head -2

# 3. test the 3 previously-failing models
for M in "qwen3.5:2b" "gemma4:e2b" "qwen3:4b"; do
  echo "=== testing $M ==="
  R=$(curl -s -m 180 http://localhost:11434/api/chat \
    -d "{\"model\": \"$M\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK\"}], \"stream\": false}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('message',{}).get('content',''); print('OK' if c.strip() else 'FAIL:'+d.get('error','?')[:60])" 2>/dev/null)
  echo "$M: $R"
done

# 4. if all three pass, sweep them
echo "=== arch-failure models tested; check results above for sweep decision ==="
echo "OLLAMA UPGRADE DONE"
