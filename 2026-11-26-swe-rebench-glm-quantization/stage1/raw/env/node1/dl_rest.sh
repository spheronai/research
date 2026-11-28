#!/bin/bash
# Download arms B, A, D at pinned commits, one after another, while arm C runs.
export PATH=/root/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/s3
for spec in "fp8 zai-org/GLM-5.3-Flash 46a050bec13002ad29ee795c16e61cf4fa3c4a9c" \
            "bf16 zai-org/GLM-5.3-Flash-BF16 f9fed3bbc82291073373d67452e9f25317ef7577" \
            "glm52fp8 zai-org/GLM-5.2-FP8 f33c6dc501ee5a2c7e35155653b1b1abbc320951"; do
  set -- $spec
  echo "dl $1 start $(date -u +%T)" >> logs/dl.log
  for try in 1 2 3; do
    hf download $2 --revision $3 --local-dir /mnt/vol/w/$1 >> logs/dl-$1.log 2>&1 && break
  done
  echo "DL_DONE $1 $? $(date -u +%T)" >> logs/dl.log
done
echo ALL_DL_DONE >> logs/dl.log
