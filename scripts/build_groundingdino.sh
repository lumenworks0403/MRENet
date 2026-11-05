#!/usr/bin/env bash
# Build from a temporary copy; retain the vendored source for reproducibility.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAX_JOBS=${MAX_JOBS:-1}
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "未找到 $CUDA_HOME/bin/nvcc" >&2; exit 1; }
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit('未检测到 GPU，请指定 TORCH_CUDA_ARCH_LIST，例如 4090 使用 8.9。')
arches = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
print(';'.join(f'{major}.{minor}' for major, minor in sorted(arches)))
PY
    )
fi
export TORCH_CUDA_ARCH_LIST
mkdir -p work_dirs
BUILD_DIR=$(mktemp -d "$PROJECT_ROOT/work_dirs/build_groundingdino.XXXXXXXX")
python - "$BUILD_DIR" <<'PY'
from pathlib import Path
import re
import shutil
import sys

source = Path('third_party/GroundingDINO')
target = Path(sys.argv[1])
shutil.copytree(source, target, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', 'build', '*.egg-info', '*.so', '*.pyd'))
csrc = target / 'groundingdino/models/GroundingDINO/csrc/MsDeformAttn'
for name in ('ms_deform_attn_cuda.cu', 'ms_deform_attn.h'):
    path = csrc / name
    text = path.read_text(encoding='utf-8')
    text = text.replace('AT_DISPATCH_FLOATING_TYPES(value.type(),',
                        'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),')
    text = text.replace('.type().is_cuda()', '.is_cuda()')
    text = re.sub(r'\.data<([a-z0-9_]*)>\(\)', r'.data_ptr<\1>()', text)
    if '.type()' in text or '.data<' in text:
        raise RuntimeError(f'PyTorch compatibility patch incomplete: {name}')
    path.write_text(text, encoding='utf-8')
PY
(
    cd "$BUILD_DIR"
    python setup.py build_ext --inplace
) 2>&1 | tee "$BUILD_DIR/build.log"
python - "$BUILD_DIR" <<'PY'
from pathlib import Path
import shutil
import sys
extensions = list((Path(sys.argv[1]) / 'groundingdino').glob('_C*.so'))
if not extensions:
    sys.exit('编译结束但未生成 groundingdino._C，请检查 build.log。')
for extension in extensions:
    shutil.copy2(extension, Path('third_party/GroundingDINO/groundingdino') / extension.name)
sys.path.insert(0, 'third_party/GroundingDINO')
from groundingdino import _C
print('groundingdino._C:', _C.__file__)
PY
echo "编译记录: $BUILD_DIR/build.log"
