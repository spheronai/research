#!/bin/bash
# Stage 1: one arm end to end. Warm cache, serve, smoke test, 111 tasks x 5 attempts at 48 concurrent, summarize.
ARM=$1
# Arm D (GLM-5.2 FP8) runs at 16 concurrent for 2 attempts: at 48 its trials hit the 50-minute agent limit
# (see RUNLOG.md, 28 Nov). Arms A to C keep 48 and 5.
N=48; K=5
[ "$ARM" = D ] && { N=16; K=2; }
export PATH=/root/.local/bin:$PATH VLLM_API_KEY=dummy
cd /root/s3
mkdir -p raw/$ARM
./warm.sh $ARM
./serve.sh $ARM
until curl -sf -m 3 localhost:8000/health >/dev/null; do sleep 10; if ! docker ps -q -f name=vllm | grep -q .; then echo "VLLM_DIED"; docker logs vllm > logs/vllm-$ARM.log 2>&1; exit 1; fi; done
echo "ready $ARM $(date -u +%T)"
docker logs vllm > logs/vllm-$ARM-start.log 2>&1
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm","messages":[{"role":"user","content":"List files in /tmp"}],"tools":[{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],"max_tokens":512}' > raw/$ARM/smoke.json
( while true; do nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used --format=csv,noheader >> raw/$ARM/power.csv; sleep 5; done ) &
P1=$!
( while true; do echo "# $(date -u +%s)" >> raw/$ARM/vllm_metrics.txt; curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:(prompt_tokens_total|generation_tokens_total|num_requests_running|num_requests_waiting|kv_cache_usage_perc|prefix_cache_hits_total|prefix_cache_queries_total|num_preemptions_total)" >> raw/$ARM/vllm_metrics.txt; sleep 30; done ) &
P2=$!
date -u +%s > raw/$ARM/run_start
if [ -d jobs/stage1-$ARM ]; then
  harbor jobs resume -p jobs/stage1-$ARM >> logs/harbor-$ARM.log 2>&1
else
  harbor run -p tasks/swe-rebench-07-2026 -a mini-swe-agent -m hosted_vllm/glm \
    --skip-model-parsing-and-provider-mapping --api-key-name VLLM_API_KEY \
    --ak config_file=/root/s3/swerebench_mini.yaml \
    --ae MSWEA_COST_TRACKING=ignore_errors --override-memory-mb 16384 -n $N -k $K -y -q -o /root/s3/jobs --job-name stage1-$ARM >> logs/harbor-$ARM.log 2>&1
fi
echo "HARBOR_EXIT $? $(date -u +%T)" >> logs/harbor-$ARM.log
date -u +%s > raw/$ARM/run_end
kill $P1 $P2
docker logs vllm > logs/vllm-$ARM.log 2>&1
python3 raw/env/summarize.py $ARM > logs/summary-$ARM.txt 2>&1
echo "ARM_DONE $ARM $(date -u +%T)"
