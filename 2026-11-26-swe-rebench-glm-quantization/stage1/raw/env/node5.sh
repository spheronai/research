#!/bin/bash
# Node 5 bring-up. Node 3 was reclaimed (spot) at 06:24:50 UTC on 27 Nov, about 3 h into arm B; node 4 was
# reclaimed before it finished booting and was never charged.
# In node 3's last 36 seconds (06:23:11 to 06:23:47) every arm B trial still running was written up as a
# failure with no verifier reward (agent exit 255, rate-limit and network errors from the dying vLLM, then
# docker compose errors). Every trial that finished at or before 06:22:20 has a normal verifier result.
# `harbor jobs resume` would treat the cut trials as finished, so they move to aborted/ and harbor reruns them.
# Then the same setup as node 3 (setup2.sh) and the chain restarts: arm B resumes, then A and D.
set -x
mkdir -p /mnt/vol
mountpoint -q /mnt/vol || mount -t virtiofs study3stage1vol /mnt/vol
cd /mnt/vol/s3
mkdir -p aborted/stage1-B-node3-reclaim
python3 - <<'PY'
import json, glob, os, shutil
moved = 0
for f in glob.glob('/mnt/vol/s3/jobs/stage1-B/*__*/result.json'):
    d = json.load(open(f))
    if (d.get('finished_at') or '') >= '2026-11-27T06:23:00':
        t = os.path.dirname(f)
        shutil.move(t, '/mnt/vol/s3/aborted/stage1-B-node3-reclaim/' + os.path.basename(t))
        moved += 1
print('moved', moved)
PY
echo "QUARANTINE_NODE3_B $(date -u +%T)" >> logs/chain.log
bash ./setup2.sh > logs/setup-node5.log 2>&1
echo "SETUP_NODE5_EXIT $? $(date -u +%T)" >> logs/chain.log
cd /root/s3
echo "CHAIN_RESTART node5 $(date -u +%T)" >> logs/chain.log
echo "--- restart on node 5 $(date -u +%T)" >> logs/arm-B.out
setsid nohup ./chain.sh > logs/chain-node5.out 2>&1 < /dev/null &
sleep 2
pgrep -af "chain.sh|arm.sh" | grep -v pgrep
