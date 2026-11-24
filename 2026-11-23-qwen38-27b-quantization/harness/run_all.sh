#!/usr/bin/env bash
# Runs every arm in order. Start with: nohup bash run_all.sh > $STUDY_DIR/raw/logs/run_all.log 2>&1 &
# A to D force a BF16 KV cache (the NVFP4 checkpoint ships an FP8 KV scheme that "auto" would pick up).
# E: FP8 weights + FP8 KV. F: NVFP4 as shipped (its own FP8 KV). E and F run GSM8K and NIAH only.
# Arm A ran its evals under the first launch (run_arm.sh A /models/A bfloat16 full); its throughput sweep
# was cut and rerun here after the prompts-per-level change, so all arms use the same sweep. GPQA Diamond
# (gated dataset, no token on the VM) was replaced by MATH-500 for every arm; arm A runs it here.
R=${STUDY_DIR:-$HOME/study}  # study folder on the GPU machine
bash $R/harness/run_arm.sh A /models/A bfloat16 math500,bench
bash $R/harness/run_arm.sh B /models/B bfloat16 full
bash $R/harness/run_arm.sh C /models/C bfloat16 full
bash $R/harness/run_arm.sh D /models/D bfloat16 full
bash $R/harness/run_arm.sh E /models/B fp8 kvonly
bash $R/harness/run_arm.sh F /models/C auto kvonly
echo ALL_DONE
# Appended during the run after lm-eval[math] was installed (arm A's first MATH-500 attempt failed on missing deps):
bash $R/harness/run_arm.sh A /models/A bfloat16 math500
# Appended during the run: arms C, E and F failed at startup and are retried (see harness/retry.sh).
bash $R/harness/retry.sh
