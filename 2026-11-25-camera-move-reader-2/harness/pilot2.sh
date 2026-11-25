#!/usr/bin/env bash
# Pilot: determinism (20 run-one test clips rendered twice) and 52 clips per new source.
cd /ephemeral/study
for s in 0 1; do
  ./blender -b --factory-startup -P harness/render_clips2.py -- --jobs jobs/det.jsonl --shard $s --nshards 2 --assets assets --out det_a > logs/det_a_$s.log 2>&1 &
  ./blender -b --factory-startup -P harness/render_clips2.py -- --jobs jobs/det.jsonl --shard $s --nshards 2 --assets assets --out det_b > logs/det_b_$s.log 2>&1 &
done
wait
venv/bin/python harness/det_check.py det_a/test_scenes det_b/test_scenes ref/r1meta/test_scenes
for sp in pilot_F pilot_H; do
  for s in 0 1; do
    ./blender -b --factory-startup -P harness/render_clips2.py -- --jobs jobs/$sp.jsonl --shard $s --nshards 2 --assets assets --out data > logs/render_${sp}_$s.log 2>&1 &
  done
done
wait
grep -h "SKIP\|Error\|Traceback" logs/render_pilot_*.log | head -20
venv/bin/python harness/pilot_stats.py data/pilot_F data/pilot_H
echo PILOT_DONE
