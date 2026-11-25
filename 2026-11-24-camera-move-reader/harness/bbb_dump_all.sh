#!/usr/bin/env bash
cd /ephemeral/study
ls ext/bbb/src/blender/2d/*.blend | grep -viE "blooper|alternative|extras|small" | xargs -P 12 -I{} sh -c 'b=$(basename {} .blend); timeout 300 ./blender -b {} -P harness/bbb_cam.py -- ext/bbb/cams/$b.json > ext/bbb/cams/$b.log 2>&1'
echo BBB_DUMP_DONE $(ls ext/bbb/cams/*.json | wc -l)
