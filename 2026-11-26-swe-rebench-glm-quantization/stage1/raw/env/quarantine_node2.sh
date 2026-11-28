#!/bin/bash
# Node 2 was reclaimed at 22:04 UTC on 26 Nov, 4 minutes into arm B. In its last seconds it wrote result.json
# for 77 arm B trials as infrastructure failures (agent exit 255, no reward). `harbor jobs resume` treats those
# as finished, which would count them as unresolved. This stops arm B, moves every trial that has a result.json
# with started_at before node 3 existed into aborted/, and restarts the chain so harbor reruns them.
# Trials without a result.json are left for harbor's resume to rerun.
cd /root/s3
for p in $(ps -eo pid,args | awk '$0 ~ /\.\/chain\.sh/ || $0 ~ /\.\/arm\.sh / || $0 ~ /bin\/harbor (run|jobs)/ || $0 ~ /nvidia-smi --query-gpu/ {print $1}'); do
  [ "$p" != "$$" ] && kill $p 2>/dev/null
done
sleep 5
docker ps -aq | xargs -r docker rm -f > /dev/null
docker network prune -f > /dev/null
mkdir -p aborted/stage1-B-node2-reclaim
python3 - <<'PY'
import json, glob, os, shutil
moved = 0
for f in glob.glob('/root/s3/jobs/stage1-B/*__*/result.json'):
    d = json.load(open(f))
    if (d.get('started_at') or '') < '2026-11-27T02:38':
        t = os.path.dirname(f)
        shutil.move(t, '/root/s3/aborted/stage1-B-node2-reclaim/' + os.path.basename(t))
        moved += 1
print('moved', moved)
PY
echo "QUARANTINE_NODE2_B $(date -u +%T)" >> logs/chain.log
echo "--- restart after quarantine $(date -u +%T)" >> logs/arm-B.out
setsid nohup ./chain.sh > logs/chain-node3b.out 2>&1 < /dev/null &
sleep 2
pgrep -af "chain.sh|arm.sh" | grep -v pgrep
