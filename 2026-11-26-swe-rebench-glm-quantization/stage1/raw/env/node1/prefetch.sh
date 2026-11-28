#!/bin/bash
# Pre-pull every task image once. Docker Hub allows 100 pulls per hour from this IP, so a failed pull
# waits 5 minutes and retries until it lands.
cd /root/s3
one() {
  until docker image inspect "$1" >/dev/null 2>&1; do
    docker pull -q "$1" >/dev/null 2>&1 && break
    echo "retry $1 $(date -u +%T)"; sleep 300
  done
  echo "ok $1"
}
export -f one
cat raw/env/images_all.txt | xargs -P 8 -I{} bash -c 'one {}'
echo "PREFETCH_DONE $(date -u +%T)"
