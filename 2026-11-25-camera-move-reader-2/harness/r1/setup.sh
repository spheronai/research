#!/usr/bin/env bash
# One-time VM setup for the camera move reader study. Idempotent.
set -euo pipefail
W=/ephemeral/study
mkdir -p "$W" && cd "$W"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libxkbcommon0 libsm6 libice6 \
  ffmpeg jq curl xz-utils git >/dev/null

BL=blender-4.5.17-linux-x64
if [ ! -x "$W/$BL/blender" ]; then
  curl -sSLO "https://download.blender.org/release/Blender4.5/$BL.tar.xz"
  tar -xf "$BL.tar.xz" && rm "$BL.tar.xz"
fi
ln -sfn "$W/$BL/blender" "$W/blender"
"$W/blender" --version | head -1

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d "$W/venv" ]; then
  uv venv -q --python 3.11 "$W/venv"
fi
VIRTUAL_ENV="$W/venv" uv pip install -q \
  "torch==2.8.*" "torchvision==0.23.*" --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV="$W/venv" uv pip install -q \
  numpy scipy pillow opencv-python-headless "transformers>=4.56" accelerate safetensors \
  pycolmap lpips scikit-image tqdm requests huggingface_hub
"$W/venv/bin/python" -c "import torch, pycolmap, transformers; print('torch', torch.__version__, torch.cuda.is_available(), 'pycolmap', pycolmap.__version__, 'transformers', transformers.__version__)"
"$W/venv/bin/pip" freeze > "$W/pip-freeze.txt" 2>/dev/null || VIRTUAL_ENV="$W/venv" uv pip freeze > "$W/pip-freeze.txt"
nvidia-smi -q > "$W/nvidia-smi-q.txt"
echo SETUP_OK
