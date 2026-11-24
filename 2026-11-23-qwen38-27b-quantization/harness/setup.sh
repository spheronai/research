#!/usr/bin/env bash
# Node setup for study 1. Runs on the VM as root. Installs pinned vLLM and lm-eval into separate
# venvs, downloads every checkpoint at its pinned commit and times each download.
set -euo pipefail
R=${STUDY_DIR:-$HOME/study}  # study folder on the GPU machine
mkdir -p $R/raw/env $R/raw/logs /models
exec > >(tee -a $R/raw/logs/setup.log) 2>&1

apt-get update -qq && apt-get install -y -qq python3-venv python3-dev build-essential jq curl >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

uv venv -q --python 3.12 /opt/vllm && VIRTUAL_ENV=/opt/vllm uv pip install -q "vllm==0.34.0" "huggingface_hub[hf_xet]" hf_transfer
uv venv -q --python 3.12 /opt/lmeval && VIRTUAL_ENV=/opt/lmeval uv pip install -q "lm-eval[api,ifeval]==0.4.13" langdetect immutabledict

# Environment record
nvidia-smi -q > $R/raw/env/nvidia-smi-q.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > $R/raw/env/gpu.csv
/opt/vllm/bin/pip freeze 2>/dev/null > $R/raw/env/pip-freeze-vllm.txt || VIRTUAL_ENV=/opt/vllm uv pip freeze > $R/raw/env/pip-freeze-vllm.txt
VIRTUAL_ENV=/opt/lmeval uv pip freeze > $R/raw/env/pip-freeze-lmeval.txt
{ uname -a; lsb_release -a 2>/dev/null; nvcc --version 2>/dev/null || true; lscpu | head -20; free -g; df -h /models; } > $R/raw/env/system.txt

# Pinned checkpoints: arm repo commit
cat > $R/models.tsv <<'EOF'
A	Qwen/Qwen3.8-27B	37561816e66b5b9153b1b64c10968c18c1e6f4a0
B	Qwen/Qwen3.8-27B-FP8	c3a462cc3d38cf351f3b26cbd763469eeb74ef81
C	unsloth/Qwen3.8-27B-NVFP4	f0b7c9e722f5565102fff8481c99e4d86ae099c7
D	cyankiwi/Qwen3.8-27B-AWQ-INT4	e3a700906e6a73ba4290b1873dc2ba8198527ab5
EOF
cp $R/models.tsv $R/raw/env/models.tsv
echo "arm,repo,commit,seconds,gb" > $R/raw/env/downloads.csv
while IFS=$'\t' read -r arm repo sha; do
  t0=$(date +%s.%N)
  HF_HUB_ENABLE_HF_TRANSFER=1 /opt/vllm/bin/hf download "$repo" --revision "$sha" --local-dir /models/$arm --quiet >/dev/null
  t1=$(date +%s.%N)
  gb=$(du -sb /models/$arm | awk '{printf "%.1f", $(1)/1e9}')
  echo "$arm,$repo,$sha,$(echo "$t1 - $t0" | bc),$gb" >> $R/raw/env/downloads.csv
done < $R/models.tsv
/opt/vllm/bin/vllm download-kernels && echo KERNELS_OK
cat $R/raw/env/downloads.csv
echo SETUP_DONE
