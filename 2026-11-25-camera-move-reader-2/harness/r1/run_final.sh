#!/usr/bin/env bash
# After train renders and DA3 finish: train both seeds, predict, re-shoot, score, leak check.
cd /ephemeral/study
until grep -q DA3_TRAIN_DONE logs/da3_train_loop.log 2>/dev/null; do sleep 30; done
until grep -q "COMBOS_DONE" logs/combos_driver.log && grep -q "CPU_CHAIN_DONE" logs/cpu_chain.log; do sleep 30; done
echo "START_TRAIN $(date -u +%H:%M:%S)"
venv/bin/python harness/train.py --out runs/seed0 --seed 0 > logs/train_seed0.log 2>&1 &
venv/bin/python harness/train.py --out runs/seed1 --seed 1 > logs/train_seed1.log 2>&1 &
wait
echo "TRAINED $(date -u +%H:%M:%S)"
for s in seed0 seed1; do
  venv/bin/python harness/train.py --predict runs/$s --splits val test_scenes test_combos real_tos >> logs/train_$s.log 2>&1
done
venv/bin/python harness/leak_check.py > results/leak_check.txt 2>&1
for split in test_scenes test_combos; do
  venv/bin/python harness/evaluate.py --split $split --out results > results/eval_$split.txt 2>&1
done
echo "SCORED $(date -u +%H:%M:%S)"
venv/bin/python harness/reshoot_prep.py --n 200 --out reshoot/jobs.jsonl > logs/reshoot_prep.log 2>&1
for s in 0 1 2 3; do
  ./blender -b --factory-startup -P harness/reshoot_render.py -- --jobs reshoot/jobs.jsonl --shard $s --nshards 4 \
    --assets /ephemeral/study/assets --out /ephemeral/study/reshoot > logs/reshoot_render_$s.log 2>&1 &
done
wait
venv/bin/python harness/reshoot_score.py --out results/reshoot.json > results/reshoot.txt 2>&1
echo "FINAL_DONE $(date -u +%H:%M:%S)"
