#!/usr/bin/env bash
# AutoDL: reuse the image's PyTorch and the shared installation steps.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
exec bash "$PROJECT_ROOT/scripts/setup_server.sh" --existing-env
