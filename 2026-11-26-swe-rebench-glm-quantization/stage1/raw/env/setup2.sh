#!/bin/bash
# Stage 1 setup on the second 8x B200 node, after the first was reclaimed (spot) at 14:47 UTC on 26 Nov 2026.
# Same steps as node1/setup.sh plus the fixes made on node 1 (docker_tmpfs.sh, fix_pools.sh), and two changes:
#   1. The work directory lives on the volume (/mnt/vol/s3, linked from /root/s3), so Harbor's per-trial
#      results survive a reclaim. On node 1 it was on local disk and arm C's 162 finished trials were lost.
#   2. Once pulled, the task images are saved to the volume, so a third node loads them from there instead
#      of pulling 111 images through Docker Hub's 100-pulls-per-hour limit.
set -x
mkdir -p /mnt/vol
mountpoint -q /mnt/vol || mount -t virtiofs study3stage1vol /mnt/vol
mkdir -p /mnt/vol/s3
if [ ! -L /root/s3 ]; then
  [ -d /root/s3 ] && cp -a /root/s3/. /mnt/vol/s3/ && rm -rf /root/s3
  ln -s /mnt/vol/s3 /root/s3
fi
cd /root/s3
mkdir -p logs raw/env jobs
echo "node $(hostname) up $(date -u +%FT%TZ)" >> raw/env/nodes.txt
# vLLM only on loopback and the Docker bridge.
iptables -I INPUT -p tcp --dport 8000 -j DROP
iptables -I INPUT -p tcp --dport 8000 -s 172.16.0.0/12 -j ACCEPT
iptables -I INPUT -i lo -p tcp --dport 8000 -j ACCEPT
# Docker on RAM-backed tmpfs (overlay cannot sit on virtiofs, and a loop file on it was too slow), with
# 1,024 /24 networks for 48 concurrent compose projects and docker0 pinned at 172.17.0.1 (the agent's api_base).
systemctl stop docker docker.socket
mkdir -p /mnt/dockertmp
mountpoint -q /mnt/dockertmp || mount -t tmpfs -o size=800G tmpfs /mnt/dockertmp
cat > /etc/docker/daemon.json <<'EOF'
{"data-root": "/mnt/dockertmp/docker",
 "bip": "172.17.0.1/16",
 "default-address-pools": [{"base": "172.20.0.0/14", "size": 24}],
 "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}}
EOF
cp /etc/docker/daemon.json raw/env/daemon.json
systemctl start docker
echo "node2 $(date -u +%T): $(docker info --format '{{.DockerRootDir}} {{.Driver}}')" >> raw/env/docker-root.txt
nvidia-smi -q > raw/env/nvidia-smi-node2.txt
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv tool install "huggingface_hub[hf_transfer]"
uv tool install "harbor==0.27.0"
harbor --version > raw/env/harbor-version.txt 2>&1
P=/root/.local/share/uv/tools/harbor/lib/python3.12/site-packages/harbor/agents/installed/mini_swe_agent.py; cp $P raw/env/mini_swe_agent.py.orig
python3 - "$P" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
i=s.index("async def install(self, environment")
old='"set -euo pipefail; "'
j=s.index(old,i)
new='"set -euo pipefail; export UV_TOOL_BIN_DIR=\\"$HOME/.local/bin\\" UV_OFFLINE=0; "'
s=s[:j]+new+s[j+len(old):]
open(p,'w').write(s)
PY
diff -u raw/env/mini_swe_agent.py.orig $P > raw/env/harbor-mini-swe-agent.patch
s=$(date +%s)
docker pull vllm/vllm-openai@sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86 > logs/pull.log 2>&1
echo "PULL_DONE $? in $(( $(date +%s)-s )) s" >> logs/pull.log
[ -d tasks/swe-rebench-07-2026 ] || harbor download ibragim-badertdinov/swe-rebench-07-2026@2026-07-r4 -o /root/s3/tasks > logs/task-dl.log 2>&1
ls tasks/swe-rebench-07-2026 > raw/env/tasks_all.txt
for t in $(cat raw/env/tasks_all.txt); do grep -h '^FROM' tasks/swe-rebench-07-2026/$t/environment/Dockerfile | awk '{print $2}'; done | sort -u > raw/env/images_all.txt
echo "SETUP_BASE_DONE $(date -u +%T) tasks=$(wc -l < raw/env/tasks_all.txt) images=$(wc -l < raw/env/images_all.txt)"
# Task images: load the saved copy if a previous node left one, else pull with retries.
if [ -f /mnt/vol/images/images.tar ]; then
  s=$(date +%s); docker load -i /mnt/vol/images/images.tar > logs/img-load.log 2>&1; echo "IMG_LOAD_DONE $? in $(( $(date +%s)-s )) s" >> logs/img-load.log
fi
./prefetch.sh > logs/prefetch.log 2>&1
echo "PREFETCH_DONE $(date -u +%T)"
