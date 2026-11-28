#!/bin/bash
# Stop a running arm: arm.sh, harbor, samplers, trial containers and vLLM. Usage: ./stop_arm.sh <arm> <reason-tag>
ARM=$1; TAG=$2
cd /root/s3
for p in $(ps -eo pid,args | awk '$0 ~ /\.\/arm\.sh / || $0 ~ /bin\/harbor run/ || $0 ~ /nvidia-smi --query-gpu/ || ($0 ~ /sleep (5|30)$/) {print $1}'); do
  [ "$p" != "$$" ] && kill $p 2>/dev/null
done
sleep 3
docker ps -aq | xargs -r docker rm -f > /dev/null
docker network prune -f > /dev/null
mkdir -p aborted
[ -d jobs/stage1-$ARM ] && mv jobs/stage1-$ARM aborted/stage1-$ARM-$TAG
[ -f logs/harbor-$ARM.log ] && mv logs/harbor-$ARM.log aborted/harbor-$ARM-$TAG.log
[ -f logs/arm-$ARM.out ] && mv logs/arm-$ARM.out aborted/arm-$ARM-$TAG.out
[ -d raw/$ARM ] && mv raw/$ARM aborted/raw-$ARM-$TAG
echo "STOPPED $ARM $TAG $(date -u +%T)"
