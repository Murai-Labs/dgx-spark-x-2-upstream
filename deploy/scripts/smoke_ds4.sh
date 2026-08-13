#!/bin/bash
# Post-boot smoke test for DeepSeek-V4-Flash on the 2-node Spark cluster.
# Run on the HEAD node once vLLM reports ready.
BASE="http://127.0.0.1:${VLLM_PORT:-8888}/v1"
echo "=== 1. /v1/models ==="
curl -sf --max-time 20 "$BASE/models" | python3 -c 'import json,sys; d=json.load(sys.stdin); [print("  id:",m["id"]) for m in d.get("data",[])]' \
  || { echo "  FAILED - server not answering on $BASE"; exit 1; }

echo
echo "=== 2. short completion (latency + token accounting) ==="
START=$(date +%s.%N)
RESP=$(curl -sf --max-time 180 "$BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":24,"temperature":0,"stream":false}')
RC=$?
END=$(date +%s.%N)
if [ $RC -ne 0 ] || [ -z "$RESP" ]; then echo "  FAILED - no completion response"; exit 1; fi
echo "$RESP" | python3 -c '
import json,sys
d=json.load(sys.stdin)
ch=d.get("choices",[{}])[0].get("message",{})
print("  content:", repr((ch.get("content") or "")[:120]))
if ch.get("reasoning_content"): print("  reasoning present:", len(ch["reasoning_content"]), "chars")
u=d.get("usage",{})
print("  usage:", {k:u.get(k) for k in ("prompt_tokens","completion_tokens","total_tokens")})
'
echo "  wall: $(awk -v a=$START -v b=$END 'BEGIN{printf "%.2fs", b-a}')"
echo
echo "=== 3. KV cache / capacity lines from the server log ==="
sudo docker logs vllm-ds4 2>&1 | grep -iE 'kv cache|gpu blocks|maximum concurrency|graph capturing|available kv' | tail -8
