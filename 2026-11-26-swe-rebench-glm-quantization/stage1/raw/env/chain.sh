#!/bin/bash
# Run the remaining arms back to back once arm C finishes, in the plan's order (B, A, D).
# An arm already marked done is skipped, so this can be restarted after a reclaim.
# Stops at the first arm whose vLLM fails to come up (arm.sh exits 1), so a broken arm is never skipped silently.
cd /root/s3
while pgrep -f "^/bin/bash ./arm.sh C" >/dev/null; do sleep 30; done
echo "CHAIN_START $(date -u +%T)" >> logs/chain.log
for a in B A D; do
  grep -q "ARM_DONE $a" logs/arm-$a.out 2>/dev/null && continue
  ./arm.sh $a >> logs/arm-$a.out 2>&1 || { echo "CHAIN_STOP $a $(date -u +%T)" >> logs/chain.log; exit 1; }
  echo "CHAIN_DONE $a $(date -u +%T)" >> logs/chain.log
done
echo "CHAIN_ALL_DONE $(date -u +%T)" >> logs/chain.log
