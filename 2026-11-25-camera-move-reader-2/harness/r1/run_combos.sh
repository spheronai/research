#!/usr/bin/env bash
# Re-render test_combos after the held-out sampler fix, then run every baseline on it.
cd /ephemeral/study
for s in 0 1; do
  ./blender -b --factory-startup -P harness/render_clips.py -- --jobs jobs/test_combos.jsonl --shard $s --nshards 2 \
    --assets /ephemeral/study/assets --out /ephemeral/study/data > logs/render_test_combos_fix_$s.log 2>&1 &
done
wait
echo "COMBOS_RENDERED $(ls data/test_combos | wc -l) $(date -u +%H:%M:%S)"
COLMAP_MODE=per_image venv/bin/python harness/colmap_baseline.py data/test_combos --workers 8 &
venv-da3/bin/python harness/da3_infer.py data/test_combos
venv/bin/python harness/vlm_baseline.py --splits test_combos --out runs/vlm
wait
COLMAP_MODE=single venv/bin/python harness/colmap_baseline.py data/test_combos --workers 8
echo COMBOS_DONE
