#!/usr/bin/env bash
# Stop the render driver and every Blender render process.
for p in $(pgrep -f "bash harness/run_render"); do kill "$p"; done
for p in $(pgrep -x blender); do kill "$p"; done
sleep 3
echo "blender left: $(pgrep -cx blender)"
