#!/usr/bin/env bash
# Retries the arms that failed at startup on 23 Nov 2026 (see the report's deviations).
# C and F: out of memory in FlashInfer's fp4_gemm autotune warmup, after vLLM had already allocated
#   the KV cache at --gpu-memory-utilization 0.90. Retried at 0.85, then 0.80 if that also fails.
# E: the FP8 KV cache JIT build could not find ninja because /opt/vllm/bin was not on PATH.
# The first failed starts are kept in raw/failed-attempts/<arm>-first-start.
R=${STUDY_DIR:-$HOME/study}  # study folder on the GPU machine
FA=$R/raw/failed-attempts
export PATH=/opt/vllm/bin:$PATH
for a in C E F; do [ -d $R/raw/$a ] && mv $R/raw/$a $FA/$a-first-start; done
try() {  # "util list" arm model kv steps
  local utils=${1}; shift
  for u in $utils; do
    GPU_UTIL=$u bash $R/harness/run_arm.sh "$@" && return 0
    mv $R/raw/${1} $FA/${1}-util$u
  done
}
try "0.85 0.80" C /models/C bfloat16 full
try "0.90 0.85" E /models/B fp8 kvonly
try "0.85 0.80" F /models/C auto kvonly
echo RETRY_DONE
