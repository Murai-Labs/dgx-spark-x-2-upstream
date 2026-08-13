#!/bin/bash
# Replicate the DeepSeek-V4-Flash cache from <hostname> to <hostname> over the
# 200G cluster link. Run ON NODE 1 after the pull completes.
set -euo pipefail
M=models--deepseek-ai--DeepSeek-V4-Flash-0731
HUB=<HOME>/.cache/huggingface/hub
KEY=<HOME>/.ssh/<cluster-ssh-key>
PEER=<user>@<WORKER_ROCE_IP>

# Guard: refuse to replicate a partial download.
if [ -n "$(find "$HUB/$M/blobs" -name '*.incomplete' 2>/dev/null | head -1)" ]; then
  echo "ABORT: incomplete blobs present - download not finished."; exit 1
fi
SRC_BYTES=$(sudo du -sb "$HUB/$M" | cut -f1)
echo "source: $(awk -v s=$SRC_BYTES 'BEGIN{printf "%.1f GiB", s/1073741824}')"

ssh -i "$KEY" -o StrictHostKeyChecking=no "$PEER" "sudo mkdir -p $HUB"

echo "=== rsync over cluster link (<WORKER_ROCE_IP>) ==="
sudo rsync -a --info=progress2 --no-inc-recursive \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --rsync-path="sudo rsync" \
  "$HUB/$M" "$PEER:$HUB/"

echo
echo "=== verify sizes match ==="
DST_BYTES=$(ssh -i "$KEY" -o StrictHostKeyChecking=no "$PEER" "sudo du -sb $HUB/$M | cut -f1")
echo "  node1: $SRC_BYTES"
echo "  node2: $DST_BYTES"
if [ "$SRC_BYTES" = "$DST_BYTES" ]; then echo "  SIZES MATCH"; else echo "  *** MISMATCH ***"; exit 1; fi
