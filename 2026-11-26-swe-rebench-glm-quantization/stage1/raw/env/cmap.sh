#!/bin/bash
# Logs docker container id -> name every 20 s, so cgroup OOM kills in dmesg (which name the container id)
# can be tied to the Harbor trial that container belonged to.
while true; do docker ps --no-trunc --format "{{.ID}} {{.Names}}" >> /root/s3/logs/containers.raw; sleep 20; done
