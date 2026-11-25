#!/usr/bin/env bash
# Run baselines on each split as soon as its renders finish.
# CPU chain: COLMAP (per-image focal, then shared focal). GPU chain: DA3, then the VLM.
cd /ephemeral/study
wait_split() { until grep -q "SPLIT_DONE $1" logs/render_driver.log 2>/dev/null; do sleep 30; done; }

cpu_chain() {
  for split in test_scenes test_combos val; do
    wait_split $split
    COLMAP_MODE=per_image venv/bin/python harness/colmap_baseline.py data/$split --workers 14
    COLMAP_MODE=single venv/bin/python harness/colmap_baseline.py data/$split --workers 14
  done
  echo CPU_CHAIN_DONE
}

gpu_chain() {
  for split in test_scenes test_combos val; do
    wait_split $split
    venv-da3/bin/python harness/da3_infer.py data/$split
    venv/bin/python harness/vlm_baseline.py --splits $split --out runs/vlm
  done
  wait_split train
  venv-da3/bin/python harness/da3_infer.py data/train
  echo GPU_CHAIN_DONE
}

cpu_chain > logs/cpu_chain.log 2>&1 &
gpu_chain > logs/gpu_chain.log 2>&1 &
wait
echo BASELINES_DONE
