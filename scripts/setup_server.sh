#!/usr/bin/env bash
# Linux + CUDA installation. --existing-env reuses the active PyTorch environment.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--existing-env" ) ]]; then
    echo "Usage: bash scripts/setup_server.sh [--existing-env]" >&2
    exit 2
fi
if [[ "${1:-}" != "--existing-env" ]]; then
    RRS_ENV_NAME=${RRS_ENV_NAME:-rrs_b0}
    TORCH_CUDA=${TORCH_CUDA:-cu121}
    conda create -y -n "$RRS_ENV_NAME" python=3.10
    eval "$(conda shell.bash hook)"
    conda activate "$RRS_ENV_NAME"
    python -m pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi

python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
python -m pip install -r requirements.txt
bash scripts/build_groundingdino.sh
bash scripts/download_weights.sh
echo "基础环境安装完成。启用 MMSCopE 时继续运行 bash scripts/setup_mmscope.sh。"
