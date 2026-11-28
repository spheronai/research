#!/bin/bash
# Stage 1 setup on 8x B200. Volume is virtiofs at /mnt/vol.
set -x
cd /root/s3
mkdir -p logs raw/env jobs
echo "volume speed check $(date -u +%H:%M) UTC: write 7165 MB/s, read 11456 MB/s (8 streams, 8 GB, caches dropped)" > raw/env/volume-speed.txt
# vLLM only on loopback and the Docker bridge.
iptables -I INPUT -p tcp --dport 8000 -j DROP
iptables -I INPUT -p tcp --dport 8000 -s 172.16.0.0/12 -j ACCEPT
iptables -I INPUT -i lo -p tcp --dport 8000 -j ACCEPT
# Docker cannot keep overlay layers on virtiofs, so its data root is an ext4 loop file on the volume.
if ! mountpoint -q /mnt/dockerfs; then
  truncate -s 1200G /mnt/vol/docker.img
  mkfs.ext4 -q -F /mnt/vol/docker.img
  mkdir -p /mnt/dockerfs && mount -o loop /mnt/vol/docker.img /mnt/dockerfs
fi
systemctl stop docker docker.socket
echo '{"data-root": "/mnt/dockerfs/docker", "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}' > /etc/docker/daemon.json
systemctl start docker
docker info --format '{{.DockerRootDir}} {{.Driver}}' > raw/env/docker-root.txt
nvidia-smi -q > raw/env/nvidia-smi.txt
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
( docker pull vllm/vllm-openai@sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86 > logs/pull.log 2>&1; echo PULL_DONE $? >> logs/pull.log ) &
( harbor download ibragim-badertdinov/swe-rebench-07-2026@2026-07-r4 -o /root/s3/tasks > logs/task-dl.log 2>&1
  ls tasks/swe-rebench-07-2026 > raw/env/tasks_all.txt
  for t in $(cat raw/env/tasks_all.txt); do grep -h '^FROM' tasks/swe-rebench-07-2026/$t/environment/Dockerfile | awk '{print $2}'; done | sort -u > raw/env/images_all.txt
  cat raw/env/images_all.txt | xargs -P 16 -n 1 docker pull > logs/img-pull.log 2>&1; echo IMG_DONE >> logs/img-pull.log ) &
# Weights: C first (runs first), then B, A, D while C runs.
echo "dl nvfp4 start $(date -u +%T)" >> logs/dl.log
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/GLM-5.3-Flash-NVFP4 --revision 1d00738646cbb0c6d0a9eb403dac5f366f1473a7 --local-dir /mnt/vol/w/nvfp4 >> logs/dl-nvfp4.log 2>&1
echo "DL_DONE nvfp4 $? $(date -u +%T)" >> logs/dl.log
nohup ./dl_rest.sh > logs/dl-rest.log 2>&1 &
wait
echo SETUP_DONE $(date -u +%T)
