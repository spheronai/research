#!/usr/bin/env bash
# After rendering and baselines: train 4 mixes x 2 seeds, predict, leak check, score, re-shoot.
# TRAIN_N (default 6000) is cut to 4000 by the plan's render cut rule if it was triggered.
cd /ephemeral/study
N=${TRAIN_N:-6000}; T=$((N / 3))
until grep -q DA3_ALL_DONE logs/da3_loop.log 2>/dev/null && grep -q CPU_CHAIN_DONE logs/cpu_chain.log 2>/dev/null \
      && grep -q VLM_DONE logs/vlm_chain.log 2>/dev/null; do sleep 30; done
echo "START_TRAIN N=$N $(date -u +%H:%M:%S)"
declare -A MIX=( [DG]="train_G:$N" [DF]="train_F:$N" [DH]="train_H:$N" [DGFH]="train_G:$T train_F:$T train_H:$T" )
PAR=${TRAIN_PAR:-4}
i=0
for arm in DG DF DH DGFH; do
  for s in 0 1; do
    venv/bin/python harness/train.py --out runs/${arm}_s$s --seed $s --train-split ${MIX[$arm]} > logs/train_${arm}_s$s.log 2>&1 &
    i=$((i + 1)); if [ $((i % PAR)) -eq 0 ]; then wait; fi
  done
done
wait
echo "TRAINED $(date -u +%H:%M:%S)"
for r in runs/D*_s*; do
  venv/bin/python harness/train.py --predict $r --splits val test_scenes test_combos test_human test_moving > logs/predict_$(basename $r).log 2>&1 &
  [ $(jobs -r | wc -l) -ge 4 ] && wait -n
done
wait
venv/bin/python harness/leak_check.py > results/leak_check.txt 2>&1
for split in val test_scenes test_combos test_human test_moving; do
  venv/bin/python harness/evaluate.py --split $split --out results > results/eval_$split.txt 2>&1
done
venv/bin/python harness/evaluate.py --split test_human --subset moving --out results > results/eval_test_human_moving.txt 2>&1
echo "SCORED $(date -u +%H:%M:%S)"
venv/bin/python harness/reshoot_prep.py --n 200 --out reshoot/jobs.jsonl > logs/reshoot_prep.log 2>&1
for s in 0 1 2 3; do
  ./blender -b --factory-startup -P harness/reshoot_render.py -- --jobs reshoot/jobs.jsonl --shard $s --nshards 4 \
    --assets /ephemeral/study/assets --out /ephemeral/study/reshoot > logs/reshoot_render_$s.log 2>&1 &
done
wait
venv/bin/python harness/reshoot_score.py --out results/reshoot.json > results/reshoot.txt 2>&1
echo "FINAL_DONE $(date -u +%H:%M:%S)"
