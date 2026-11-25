#!/usr/bin/env bash
# Render all splits, test sets first. 4 Blender processes share the GPU.
cd /ephemeral/study
for split in test_scenes test_combos val train; do
  for s in 0 1 2 3; do
    ./blender -b --factory-startup -P harness/render_clips.py -- --jobs jobs/$split.jsonl --shard $s --nshards 4 \
      --assets /ephemeral/study/assets --out /ephemeral/study/data > logs/render_${split}_$s.log 2>&1 &
  done
  wait
  echo "SPLIT_DONE $split $(ls data/$split | wc -l) $(date -u +%H:%M:%S)"
done
echo ALL_RENDER_DONE
