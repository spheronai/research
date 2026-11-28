#!/bin/bash
# Each Harbor trial gets its own compose network. Docker's default pools hold about 31 networks, so at 48
# concurrent trials the rest failed with "all predefined address pools have been fully subnetted".
# Give Docker 1,024 /24 networks inside 172.20.0.0/14 (still inside 172.16.0.0/12, which the port 8000
# iptables rule accepts), and pin docker0 at 172.17.0.1, which the agent config uses as api_base.
set -x
cd /root/s3
mkdir -p aborted
docker rm -f vllm
docker ps -aq | xargs -r docker rm -f
docker network prune -f
mv jobs/stage1-C aborted/stage1-C-address-pools
mv logs/harbor-C.log aborted/harbor-C-address-pools.log
mv logs/arm-C.out aborted/arm-C-address-pools.out
rm -rf raw/C
cat > /etc/docker/daemon.json <<'EOF'
{"data-root": "/mnt/dockertmp/docker",
 "bip": "172.17.0.1/16",
 "default-address-pools": [{"base": "172.20.0.0/14", "size": 24}],
 "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}}
EOF
cp /etc/docker/daemon.json raw/env/daemon.json
systemctl restart docker
ip -4 addr show docker0 | grep inet
docker images -q | wc -l
echo "address pools fixed $(date -u +%T): 525 of 555 arm C trials failed at network creation and were discarded" >> raw/env/docker-root.txt
nohup ./arm.sh C > logs/arm-C.out 2>&1 &
echo FIX_DONE
