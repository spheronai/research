#!/bin/bash
# Stage 1: serve one arm. Same image, flags and KV dtype (bfloat16, forced: auto picks fp8 for the NVFP4 checkpoint) for every arm; only the checkpoint changes.
# Usage: ./serve.sh <arm>   arm = C (NVFP4) | B (FP8) | A (BF16) | D (GLM-5.2 FP8)
case "$1" in
  C) W=nvfp4 ;; B) W=fp8 ;; A) W=bf16 ;; D) W=glm52fp8 ;; *) echo "unknown arm"; exit 1 ;;
esac
docker rm -f vllm >/dev/null 2>&1
docker run -d --name vllm --gpus all --ipc=host --network host \
  -v /mnt/vol/w/$W:/checkpoint:ro \
  --entrypoint bash vllm/vllm-openai@sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86 -c '
pip install -q -U "transformers>=5.16.1"; vllm --version;
exec vllm serve /checkpoint --served-model-name glm \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --enable-expert-parallel \
  --reasoning-parser glm45 --enable-auto-tool-choice --tool-call-parser glm47 \
  --kv-cache-dtype bfloat16 --max-model-len 131072 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
  --gpu-memory-utilization 0.85 --seed 0 \
  --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 64}"'
