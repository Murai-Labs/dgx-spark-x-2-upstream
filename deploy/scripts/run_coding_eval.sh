#!/bin/bash
# Coding-quality evaluation for the live DeepSeek-V4-Flash endpoint.
#
# Runs EvalPlus HumanEval+ (164 problems, each with ~80x more tests than the
# original HumanEval) against the OpenAI-compatible API.
#
# Contamination caveat: HumanEval (2021) is almost certainly in this model's
# training data. HumanEval+ adds harder generated tests, which catches some
# memorisation, but a high score here is NOT evidence of general coding skill.
# It is a sanity check that the serving stack produces valid, runnable code.
# LiveCodeBench (post-cutoff problems) is the contamination-resistant follow-up.
set -euo pipefail

OUT=<HOME>/deepseek-v4/codeeval
mkdir -p "$OUT"

sudo docker run --rm --network host \
  -v "$OUT":/out \
  -e OPENAI_API_KEY=dummy \
  -e OPENAI_BASE_URL=http://127.0.0.1:8888/v1 \
  -e HF_HOME=/cache/huggingface \
  -v <HOME>/.cache/huggingface:/cache/huggingface \
  --entrypoint bash vllm-node-pr51959 -c '
set -euo pipefail
python3 -m pip install --quiet --no-input evalplus 2>&1 | tail -2
echo "=== evalplus installed ==="
cd /out
# greedy decoding, one sample per problem -> deterministic, comparable
evalplus.codegen \
  --model deepseek-v4-flash \
  --dataset humaneval \
  --backend openai \
  --base-url http://127.0.0.1:8888/v1 \
  --greedy \
  --root /out 2>&1 | tail -20
echo "=== generation done, evaluating ==="
SAMPLES=$(find /out -name "*.jsonl" | head -1)
echo "samples: $SAMPLES"
evalplus.evaluate --dataset humaneval --samples "$SAMPLES" 2>&1 | tail -25
'
