#!/bin/bash
# Read one arm's shards into page cache with 8 parallel sequential reads (vLLM's loader is slow on network storage).
case "$1" in C) W=nvfp4 ;; B) W=fp8 ;; A) W=bf16 ;; D) W=glm52fp8 ;; *) exit 1 ;; esac
echo "warm $1 start $(date -u +%T)" >> /root/s3/logs/warm.log
ls /mnt/vol/w/$W/*.safetensors | xargs -P 8 -I{} dd if={} of=/dev/null bs=16M status=none
echo "warm $1 done $(date -u +%T)" >> /root/s3/logs/warm.log
