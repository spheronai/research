#!/usr/bin/env bash
# DA3 on train while it renders: 3 shards, repeated passes, a final pass after rendering ends.
cd /ephemeral/study
pass() { for s in 0 1 2; do DA3_SHARD=$s DA3_NSHARDS=3 venv-da3/bin/python harness/da3_infer.py data/train > logs/da3_train_$s.log 2>&1 & done; wait; }
until grep -q "SPLIT_DONE train" logs/render_driver.log; do pass; sleep 20; done
pass
echo "DA3_TRAIN_DONE $(ls data/train/*/da3.npz | wc -l) $(date -u +%H:%M:%S)"
