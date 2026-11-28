#!/bin/bash
# Move Docker's data root from an ext4 loop file on the virtiofs volume (layer extraction ran at 4 to 200 MB/s,
# because every small write round-trips through virtiofs) to a RAM-backed tmpfs. The node has 1.76 TB RAM.
set -x
cd /root/s3
systemctl stop docker docker.socket
umount /mnt/dockerfs && losetup -D && rm -f /mnt/vol/docker.img
mkdir -p /mnt/dockertmp && mount -t tmpfs -o size=800G tmpfs /mnt/dockertmp
echo '{"data-root": "/mnt/dockertmp/docker", "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}}' > /etc/docker/daemon.json
systemctl start docker
docker info --format '{{.DockerRootDir}} {{.Driver}}' > raw/env/docker-root.txt
echo "docker data root moved to tmpfs (800G) at $(date -u +%T); the loop file on virtiofs was too slow to unpack images" >> raw/env/docker-root.txt
s=$(date +%s)
docker pull vllm/vllm-openai@sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86 > logs/pull.log 2>&1
echo "PULL_DONE $? in $(( $(date +%s)-s )) s" >> logs/pull.log
nohup ./prefetch.sh > logs/prefetch.log 2>&1 &
echo TMPFS_DONE
