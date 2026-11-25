#!/usr/bin/env bash
# Baselines as splits finish. GPU: DA3 on every split (3 shards, repeated passes while
# rendering), Qwen3-VL on test_human and test_moving. CPU: COLMAP (both modes) on
# test_combos, test_human and val (val calibrates the handheld threshold).
cd /ephemeral/study
SPLITS="test_scenes test_combos test_human test_moving val train_G train_F train_H"
done_split() { grep -q "SPLIT_DONE $1 " logs/render_driver.log 2>/dev/null; }
wait_split() { until done_split $1; do sleep 30; done; }

da3_loop() {
  pass() {
    dirs=""; for sp in $SPLITS; do [ -d data/$sp ] && dirs="$dirs data/$sp"; done
    [ -z "$dirs" ] && { sleep 30; return; }
    for s in 0 1 2; do DA3_SHARD=$s DA3_NSHARDS=3 venv-da3/bin/python harness/da3_infer.py $dirs >> logs/da3_$s.log 2>&1 & done
    wait
  }
  until grep -q ALL_RENDER_DONE logs/render_driver.log 2>/dev/null; do pass; sleep 20; done
  pass
  echo "DA3_ALL_DONE $(ls data/*/*/da3.npz | wc -l) $(date -u +%H:%M:%S)"
}

vlm_chain() {
  wait_split test_moving
  venv/bin/python harness/vlm_baseline.py --splits test_human test_moving --out runs/vlm
  echo "VLM_DONE $(date -u +%H:%M:%S)"
}

cpu_chain() {
  for split in test_combos test_human val; do
    wait_split $split
    COLMAP_MODE=per_image venv/bin/python harness/colmap_baseline.py data/$split --workers 10
    COLMAP_MODE=single venv/bin/python harness/colmap_baseline.py data/$split --workers 10
  done
  echo "CPU_CHAIN_DONE $(date -u +%H:%M:%S)"
}

da3_loop > logs/da3_loop.log 2>&1 &
vlm_chain > logs/vlm_chain.log 2>&1 &
cpu_chain > logs/cpu_chain.log 2>&1 &
wait
echo BASELINES_DONE
