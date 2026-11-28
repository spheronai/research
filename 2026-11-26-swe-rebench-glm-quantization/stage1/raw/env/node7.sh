#!/bin/bash
# Node 7 bring-up. Node 6 was reclaimed (spot) at 18:42:51 UTC on 27 Nov, during arm A.
# Same recovery as nodes 5 and 6, generalised to the arm that was running (ARM):
#  - trials of that arm with finished_at >= CUTOFF are reclaim damage (no real outcome) and move to
#    aborted/stage1-$ARM-node6-reclaim/ so `harbor jobs resume` reruns them. CUTOFF is chosen by reading
#    the arm's last results on the volume first (see RUNLOG.md).
#  - if the dying node's chain already logged ARM_DONE for that arm (harbor counts the cut-off trials as
#    finished), the line is renamed ARM_VOID so the restarted chain resumes the arm instead of skipping it.
# Then the same setup as nodes 3, 5 and 6, and the chain restarts.
set -x
: "${ARM:?set ARM, e.g. A}"
: "${CUTOFF:?set CUTOFF, e.g. 2026-11-27T18:42:00}"
mkdir -p /mnt/vol
mountpoint -q /mnt/vol || mount -t virtiofs study3stage1vol /mnt/vol
cd /mnt/vol/s3
Q=aborted/stage1-$ARM-node6-reclaim
mkdir -p $Q
ARM="$ARM" CUTOFF="$CUTOFF" Q="$Q" python3 - <<'PY'
import json, glob, os, shutil
arm, cut, q = os.environ['ARM'], os.environ['CUTOFF'], os.environ['Q']
moved = 0
for f in glob.glob(f'/mnt/vol/s3/jobs/stage1-{arm}/*__*/result.json'):
    d = json.load(open(f))
    if (d.get('finished_at') or '') >= cut:
        t = os.path.dirname(f)
        shutil.move(t, f'/mnt/vol/s3/{q}/' + os.path.basename(t))
        moved += 1
print('moved', moved)
PY
sed -i "s/^ARM_DONE $ARM \(.*\)$/ARM_VOID_$ARM \1 (node 6 reclaim; cut-off trials quarantined, rerun on node 7)/" logs/arm-$ARM.out
echo "QUARANTINE_NODE6_$ARM cutoff=$CUTOFF $(date -u +%T)" >> logs/chain.log
bash ./setup2.sh > logs/setup-node7.log 2>&1
echo "SETUP_NODE7_EXIT $? $(date -u +%T)" >> logs/chain.log
cd /root/s3
echo "CHAIN_RESTART node7 $(date -u +%T)" >> logs/chain.log
echo "--- restart on node 7 $(date -u +%T)" >> logs/arm-$ARM.out
setsid nohup ./chain.sh > logs/chain-node7.out 2>&1 < /dev/null &
sleep 2
pgrep -af "chain.sh|arm.sh" | grep -v pgrep
