#!/usr/bin/env bash
# Run two VM setup: run one's setup.sh, then DA3 at run one's commit, then the RealEstate10K poses.
set -euo pipefail
W=/ephemeral/study
cd "$W"
bash harness/setup.sh
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d da3 ]; then git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3 da3; fi
(cd da3 && git checkout -q 3d835ec && git log -1 --format='da3 %h %cd')
if [ ! -d venv-da3 ]; then uv venv -q --python 3.11 venv-da3; fi
VIRTUAL_ENV=venv-da3 uv pip install -q "torch==2.8.*" "torchvision==0.23.*" --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=venv-da3 uv pip install -q xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=venv-da3 uv pip install -q -r da3/requirements.txt
VIRTUAL_ENV=venv-da3 uv pip install -q -e ./da3 addict
venv-da3/bin/python -c "from depth_anything_3.api import DepthAnything3; print('da3 import ok')"
mkdir -p ext/re10k && cd ext/re10k
if [ ! -d RealEstate10K ]; then
  curl -sSLf -o RealEstate10K.tar.gz https://storage.googleapis.com/realestate10k-public-files/RealEstate10K.tar.gz
  sha256sum RealEstate10K.tar.gz > RealEstate10K.tar.gz.sha256
  tar -xzf RealEstate10K.tar.gz && rm RealEstate10K.tar.gz
fi
echo "re10k train $(ls RealEstate10K/train | wc -l) test $(ls RealEstate10K/test | wc -l)"
echo SETUP2_OK
