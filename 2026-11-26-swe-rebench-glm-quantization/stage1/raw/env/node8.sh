#!/bin/bash
# Node 8 bring-up. Node 7 was reclaimed (spot) at 01:11:44 UTC on 28 Nov, during arm D.
# Arm D (GLM-5.2 FP8) could not finish tasks at 48 concurrent: 48 of its first 50 trials hit each task's 50-minute
# agent limit after a median of 44 agent steps (arm A's trials finish in a median of about 68), with vLLM's KV cache
# at 95-99% and requests queueing. On 28 Nov arm D restarts at 16 concurrent for 2 attempts
# per task (222 trials); arm.sh carries the change. Every arm D trial from the 48-concurrent run, finished or not,
# moves to aborted/stage1-D-c48/ with that run's raw files and logs. None of it is used in the analysis.
# Then the same setup as nodes 3, 5, 6 and 7, and the chain restarts (A, B and C are done, so it runs D only).
set -x
mkdir -p /mnt/vol
mountpoint -q /mnt/vol || mount -t virtiofs study3stage1vol /mnt/vol
cd /mnt/vol/s3
Q=aborted/stage1-D-c48
mkdir -p $Q
[ -d jobs/stage1-D ] && mv jobs/stage1-D $Q/jobs
[ -d raw/D ] && mv raw/D $Q/raw
for f in logs/harbor-D.log logs/vllm-D-start.log logs/vllm-D.log logs/arm-D.out logs/summary-D.txt; do
  [ -f "$f" ] && mv "$f" $Q/
done
echo "moved $(ls -d $Q/jobs/*__* 2>/dev/null | wc -l) trial dirs"
cp raw/env/arm.sh arm.sh
grep -n "N=16" arm.sh
echo "QUARANTINE_NODE7_D_C48 $(date -u +%T)" >> logs/chain.log
bash ./setup2.sh > logs/setup-node8.log 2>&1
echo "SETUP_NODE8_EXIT $? $(date -u +%T)" >> logs/chain.log
cd /root/s3
echo "CHAIN_RESTART node8 $(date -u +%T)" >> logs/chain.log
setsid nohup ./chain.sh > logs/chain-node8.out 2>&1 < /dev/null &
sleep 2
pgrep -af "chain.sh|arm.sh" | grep -v pgrep
