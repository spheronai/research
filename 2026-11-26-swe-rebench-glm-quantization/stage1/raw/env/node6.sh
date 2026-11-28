#!/bin/bash
# Node 6 bring-up. Node 5 was reclaimed (spot) at 11:12:49 UTC on 27 Nov, about 90% through arm B.
# As after node 3, trials still running when vLLM died were written up as failures with no real outcome.
# CUTOFF is the finished_at time from which arm B results are reclaim damage, chosen by reading the
# last results on the volume before this runs (see RUNLOG.md). Those trials move to aborted/ so
# `harbor jobs resume` reruns them. Then the same setup as nodes 3 and 5, and the chain restarts.
set -x
: "${CUTOFF:?set CUTOFF, e.g. 2026-11-27T11:12:00}"
mkdir -p /mnt/vol
mountpoint -q /mnt/vol || mount -t virtiofs study3stage1vol /mnt/vol
cd /mnt/vol/s3
mkdir -p aborted/stage1-B-node5-reclaim
CUTOFF="$CUTOFF" python3 - <<'PY'
import json, glob, os, shutil
cut = os.environ['CUTOFF']
moved = 0
for f in glob.glob('/mnt/vol/s3/jobs/stage1-B/*__*/result.json'):
    d = json.load(open(f))
    if (d.get('finished_at') or '') >= cut:
        t = os.path.dirname(f)
        shutil.move(t, '/mnt/vol/s3/aborted/stage1-B-node5-reclaim/' + os.path.basename(t))
        moved += 1
print('moved', moved)
PY
echo "QUARANTINE_NODE5_B cutoff=$CUTOFF $(date -u +%T)" >> logs/chain.log
bash ./setup2.sh > logs/setup-node6.log 2>&1
echo "SETUP_NODE6_EXIT $? $(date -u +%T)" >> logs/chain.log
cd /root/s3
echo "CHAIN_RESTART node6 $(date -u +%T)" >> logs/chain.log
echo "--- restart on node 6 $(date -u +%T)" >> logs/arm-B.out
setsid nohup ./chain.sh > logs/chain-node6.out 2>&1 < /dev/null &
sleep 2
pgrep -af "chain.sh|arm.sh" | grep -v pgrep
