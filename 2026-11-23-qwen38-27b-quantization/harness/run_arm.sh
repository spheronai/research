#!/usr/bin/env bash
# One arm: start vLLM, time load to healthy, run evals, NIAH, throughput sweep, sample power, stop.
# Usage: run_arm.sh <arm> <model_dir> <kv_cache_dtype> <steps>
# steps: comma list of gsm8k,mmlu_pro,math500,ifeval,niah,bench, or a shorthand:
#   full = gsm8k,mmlu_pro,math500,ifeval,niah,bench   kvonly = gsm8k,niah
set -uo pipefail
ARM=${1}; MODEL=${2}; KV=${3}; MODE=${4:-full}
case $MODE in full) STEPS=gsm8k,mmlu_pro,math500,ifeval,niah,bench;; kvonly) STEPS=gsm8k,niah;; *) STEPS=$MODE;; esac
has() { [[ ",$STEPS," == *",${1},"* ]]; }
R=${STUDY_DIR:-$HOME/study}  # study folder on the GPU machine
O=$R/raw/$ARM; mkdir -p $O/evals $O/bench $O/logs
PORT=8000; NAME=qwen38-27b
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $O/logs/timeline.log; }

nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm \
  --format=csv,nounits -lms 1000 > $O/logs/power.csv &
SMI=$!

log "serve start arm=$ARM model=$MODEL kv=$KV mode=$MODE gpu_util=${GPU_UTIL:-0.90}"
T0=$(date +%s.%N)
/opt/vllm/bin/vllm serve $MODEL --served-model-name $NAME --port $PORT \
  --max-model-len 40960 --gpu-memory-utilization ${GPU_UTIL:-0.90} --max-num-seqs 256 --seed 0 \
  --kv-cache-dtype $KV --disable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > $O/logs/server.log 2>&1 &
SRV=$!
until curl -sf localhost:$PORT/health >/dev/null; do
  if ! kill -0 $SRV 2>/dev/null; then log "SERVER DIED during load"; kill $SMI; exit 2; fi
  sleep 1
done
T1=$(date +%s.%N)
LOAD=$(echo "$T1 - $T0" | bc); log "healthy load_s=$LOAD"; echo "$LOAD" > $O/load_seconds.txt

# Thinking-off check: record one raw answer
curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer briefly.\"}],\"max_tokens\":200,\"temperature\":0}" \
  > $O/logs/sanity.json
log "sanity: $(jq -c '.choices[0].message' $O/logs/sanity.json | head -c 400)"

LM="model=$NAME,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,timeout=1200"
ev() {  # name, lm-eval args...
  local n=${1}; shift
  log "eval $n start"
  /opt/lmeval/bin/lm_eval --model local-chat-completions --model_args "$LM" --apply_chat_template \
    --output_path $O/evals/$n --log_samples --seed 0 "$@" > $O/logs/eval-$n.log 2>&1
  log "eval $n done rc=$? $(grep -E '^\|' $O/logs/eval-$n.log | tail -n +3 | head -20 | tr -s ' ' | tr '\n' ';')"
}

has gsm8k && ev gsm8k --tasks gsm8k --num_fewshot 5 --fewshot_as_multiturn --gen_kwargs temperature=0,max_gen_toks=512
has mmlu_pro && ev mmlu_pro --tasks mmlu_pro --limit 140 --fewshot_as_multiturn --gen_kwargs temperature=0,max_gen_toks=1024
has math500 && ev math500 --tasks minerva_math500 --fewshot_as_multiturn --gen_kwargs temperature=0,max_gen_toks=1024
has ifeval && ev ifeval --tasks ifeval --gen_kwargs temperature=0

if has niah; then
log "niah start"
/opt/vllm/bin/python $R/harness/niah.py $PORT $NAME $MODEL $O/evals/niah.jsonl --lengths 8192,32768 --n 100 \
  > $O/evals/niah_summary.json 2> $O/logs/niah.err
log "niah done $(cat $O/evals/niah_summary.json)"
fi

if has bench; then
  for C in 1 4 16 64 128; do
    N=$(( C*4 > 8 ? C*4 : 8 ))  # 4 rounds per level, 8 prompts at concurrency 1
    for REP in 1 2 3; do
      log "bench c=$C rep=$REP start"
      echo "$C,$REP,start,$(date +%s.%N)" >> $O/bench/windows.csv
      /opt/vllm/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model $NAME \
        --tokenizer $MODEL --dataset-name random --random-input-len 1024 --random-output-len 512 \
        --ignore-eos --num-prompts $N --max-concurrency $C --seed $REP --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 --save-result --result-dir $O/bench --result-filename c${C}_r${REP}.json \
        > $O/logs/bench-c${C}-r${REP}.log 2>&1
      echo "$C,$REP,end,$(date +%s.%N)" >> $O/bench/windows.csv
      log "bench c=$C rep=$REP done $(jq -c '{o:.output_throughput,ttft_p95:.p95_ttft_ms,tpot50:.median_tpot_ms}' $O/bench/c${C}_r${REP}.json 2>/dev/null)"
    done
  done
fi

kill $SRV; wait $SRV 2>/dev/null; kill $SMI
log "ARM_DONE $ARM"
