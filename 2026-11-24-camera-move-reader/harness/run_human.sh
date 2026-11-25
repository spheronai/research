#!/usr/bin/env bash
cd /ephemeral/study
for s in 0 1; do
  ./blender -b --factory-startup -P harness/reshoot_render.py -- --jobs reshoot/human_jobs.jsonl --shard $s --nshards 2 \
    --assets /ephemeral/study/assets --out /ephemeral/study/data > logs/human_render_$s.log 2>&1 &
done
wait
echo "HUMAN_RENDERED $(ls data/human_moves/*/f15.png | wc -l)"
COLMAP_MODE=per_image venv/bin/python harness/colmap_baseline.py data/human_moves --workers 12 > logs/human_colmap1.log 2>&1 &
COLMAP_MODE=single venv/bin/python harness/colmap_baseline.py data/human_moves --workers 12 > logs/human_colmap2.log 2>&1 &
venv-da3/bin/python harness/da3_infer.py data/human_moves > logs/human_da3.log 2>&1
venv/bin/python harness/vlm_baseline.py --splits human_moves --out runs/vlm > logs/human_vlm.log 2>&1
for s in seed0 seed1; do venv/bin/python harness/train.py --predict runs/$s --splits human_moves >> logs/human_pred.log 2>&1; done
wait
venv/bin/python harness/evaluate.py --split human_moves --out results > results/eval_human_moves.txt 2>&1
echo HUMAN_DONE
