#!/usr/bin/env bash
# MMSCopE (use_mmscope=True) 的额外依赖安装 + 依赖可导入性自检。
# 在 B0_HED 目录下, conda activate rrs_b0 之后运行:
#   bash scripts/setup_mmscope.sh
#
# 本脚本做什么:
#   1. pip install einops                      (官方 VSSM 的 A_log_init / D_init 用到)
#   2. 若 timm 不可导入, 才 pip install timm   (原 requirements.txt 已含 timm, 通常跳过)
#   3. 编译安装 selective_scan_cuda_oflex      (官方 SS2D 的 CUDA 扩展, 源码在
#                                               third_party/SegMAN/kernels/selective_scan)
#   4. 先导入 torch, 再自检四个额外依赖; 任一失败 -> 非零退出并列出原因
#
# 本脚本**不做**什么:
#   - 不安装、不升级 triton: Linux 版 PyTorch 通常自带匹配版本; 自行 pip install triton
#     很可能拉入与当前 torch 不匹配的版本。triton 缺失时脚本会失败并在末尾给出处理建议, 由你决定。
#   - 不升级/降级 torch、torchvision、transformers, 不动 GroundingDINO。
#   - 不照搬 SegMAN 的整套环境 (不装 mmcv-full / mmseg / natten / fvcore)。
#
# 自检的边界 (很重要):
#   本脚本只验证"依赖对象能导入"。它**不**验证真实 CUDA 扫描算子能完成前向/反向 ——
#   那需要 GPU 上实际跑一次, 见 tools/check_ss2d.py (由 scripts/verify_mm_score.sh 调用)。
#   require_scan_ops() 同样只检查依赖是否存在, 它返回成功 **不等于** 算子已验证可运行。
#
# 版本记录 (2026-10-17):
#   此前记录为 torch 2.8.0+cu128 / Python 3.12, 以本次服务器打印为准。
#   torch.version.cuda 是 torch 的构建版本, 不能代替 nvcc -V 或驱动版本。
#   官方 SegMAN 声明 torch 2.1.2。selective_scan 扩展与 triton 在 torch 2.8 / CUDA 12.8 上
#   能否顺利编译并给出正确结果, 本项目尚未验证; 以本脚本的自检与 tools/check_ss2d.py 为准。
#   已知该组合下 setup.py 会提示 "There are no g++ version bounds defined for CUDA version 12.8",
#   这只是警告。timm 的 DropPath 位置未实测
#   (mmscope_core.py 已兼容 timm.models.layers / timm.layers 两种路径)。
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# 有卡时默认探测可见 GPU; 无卡时必须由调用者指定实际训练卡的架构, 不猜型号。
# 临时 setup.py 会移除官方硬编码 -gencode, 让该变量真正生效。
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST=$(python - <<'EOF'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit('未检测到 GPU。请指定目标卡架构后重试，例如 4090: TORCH_CUDA_ARCH_LIST="8.9" bash scripts/setup_mmscope.sh')
arches = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
print(';'.join(f'{major}.{minor}' for major, minor in arches))
EOF
    )
fi
export TORCH_CUDA_ARCH_LIST
# 单进程编译降低并发内存需求。Killed signal 表示进程被终止,
# OOM 是可能原因, 需要内核或容器事件确认。
export MAX_JOBS=${MAX_JOBS:-1}
# 只编译本项目真正 import 的那一个扩展。官方 setup.py 默认 MODES=["core","ndstate","oflex"],
# mmscope_core.py 只用 selective_scan_cuda_oflex。减少扩展数可以减少编译工作量,
# 但不能据此推断峰值内存降低为 1/3; 单个翻译单元仍可能超出容器内存限制。
# 通过临时副本改 MODES, third_party/SegMAN/kernels 保持与上游字节一致 (SHA256 可校验)。
SCAN_MODES=${SCAN_MODES:-oflex}
# nvcc 内部线程数, 默认 4 在小内存机器上也会 OOM
NVCC_THREADS=${NVCC_THREADS:-1}
mkdir -p work_dirs
RUN_DIR=$(mktemp -d "$PROJECT_ROOT/work_dirs/setup_mmscope.XXXXXXXX")
echo "本次编译日志和临时源码保存到: $RUN_DIR"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "未找到 $CUDA_HOME/bin/nvcc。请将 CUDA_HOME 设置为实际 CUDA toolkit 目录。" >&2
    exit 1
fi

echo "===== 当前环境 ====="
python - <<'EOF'
import platform, torch
print("python", platform.python_version(), "| torch", torch.__version__,
      "| torch.version.cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
try:
    import triton
    print("triton:", triton.__version__, "(随 PyTorch 提供, 本脚本不改动它)")
except Exception as e:
    print("triton: 不可导入 ->", repr(e))
EOF
echo "主机内存 (容器实际限额可能更低, 不能据此单独确认 OOM):"
free -g 2>/dev/null | sed -n '1,3p' || echo "  (无 free 命令)"
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$f" ]; then echo "$f:"; cat "$f"; fi
done
echo "实际编译器: $CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" -V
echo "编译设置: TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS" \
     " SCAN_MODES=$SCAN_MODES  NVCC_THREADS=$NVCC_THREADS"

echo ""; echo "===== 安装 ====="
python -m pip install --no-deps einops
# ninja: 没有它 torch 退回 distutils 慢速后端 (日志会提示 "could not find ninja")。
# 它不改变编译结果, 只影响速度与并行控制。
if python -c "import ninja" 2>/dev/null || command -v ninja >/dev/null; then
    echo "ninja 已存在 (跳过安装)"
else
    echo "ninja 不可用, 安装之"; python -m pip install --no-deps ninja
fi
# timm 只在不可导入时才装, 避免无谓改动现有环境
if python -c "import timm" 2>/dev/null; then
    python -c "import timm; print('timm 已存在:', timm.__version__, '(跳过安装)')"
else
    echo "timm 不可导入, 安装之 (不自动改变 torch 等依赖; 缺失依赖由末尾自检报告)"
    python -m pip install --no-deps timm
fi

# ---- selective_scan_cuda_oflex: 编译 VMamba 的 selective scan 扩展 ----
# 在临时副本里把 MODES 改成只有需要的那个, 原目录保持与上游一致 (SHA256 可校验)。
BUILD_DIR="$RUN_DIR/source"
mkdir -p "$BUILD_DIR"
# 不复制可能残留的 build/、egg-info 或已编译二进制。
cp ./third_party/SegMAN/kernels/selective_scan/setup.py "$BUILD_DIR/"
cp -r ./third_party/SegMAN/kernels/selective_scan/csrc "$BUILD_DIR/"
python - "$BUILD_DIR/setup.py" "$SCAN_MODES" "$NVCC_THREADS" <<'EOF'
import re
import sys

path, modes, nvcc_threads = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
modes = [m.strip() for m in modes]
if not modes or any(m not in {'core', 'ndstate', 'oflex', 'nrow'} for m in modes):
    sys.exit(f'无效 SCAN_MODES: {modes}')
if not nvcc_threads.isdigit() or int(nvcc_threads) < 1:
    sys.exit('NVCC_THREADS 必须是正整数')
src = open(path, encoding="utf-8").read()
new_modes = "MODES = [" + ", ".join(f'"{m.strip()}"' for m in modes) + "]"
src, n = re.subn(r'^MODES = \[[^\]]*\]', new_modes, src, count=1, flags=re.M)
assert n == 1, "未能在 setup.py 中定位 MODES"
src, n2 = re.subn(r'\["--threads", "4"\]', f'["--threads", "{nvcc_threads}"]', src, count=1)
if n2 != 1:
    sys.exit('未能唯一定位 nvcc --threads 参数, 停止编译')
# PyTorch 在 extra_compile_args 已含 arch 参数时不读取 TORCH_CUDA_ARCH_LIST。
# 精确移除官方这一段, 改用当前 torch 对环境变量的解析, 并打印实际传入的 flags。
start = '    cc_flag.append("-gencode")\n'
end = '    # HACK: The compiler flag'
if len(re.findall('^' + re.escape(start), src, flags=re.M)) != 2 or src.count(end) != 1:
    sys.exit('官方架构块发生变化, 请人工检查; 不继续编译')
begin = src.index(start)
finish = src.index(end, begin)
src = src[:begin] + (
    '    from torch.utils.cpp_extension import _get_cuda_arch_flags\n'
    '    cc_flag = _get_cuda_arch_flags()\n'
    '    if not cc_flag:\n'
    '        raise RuntimeError("未生成 CUDA 架构参数")\n'
    '    print("Effective CUDA arch flags:", cc_flag, flush=True)\n\n'
) + src[finish:]
open(path, "w", encoding="utf-8").write(src)
print(f"临时 setup.py: {new_modes}  (nvcc --threads -> {nvcc_threads}, 替换 {n2} 处)")
EOF
echo "开始编译。耗时和内存以实际环境为准; 完整日志: $RUN_DIR/build.log"
set +e
python -m pip install -v --no-build-isolation --no-deps "$BUILD_DIR" 2>&1 | tee "$RUN_DIR/build.log"
BUILD_STATUS=("${PIPESTATUS[@]}")
set -e
if [ "${BUILD_STATUS[1]}" -ne 0 ]; then
    echo "编译日志写入失败 (tee exit=${BUILD_STATUS[1]}, pip exit=${BUILD_STATUS[0]})" >&2
    exit 1
fi
if [ "${BUILD_STATUS[0]}" -ne 0 ]; then
    echo "编译失败 (pip exit=${BUILD_STATUS[0]})。请提供完整日志: $RUN_DIR/build.log" >&2
    echo "Error compiling objects for extension 只是汇总; 需查看此前第一条编译错误。" >&2
    echo "临时源码已保留: $BUILD_DIR" >&2
    exit "${BUILD_STATUS[0]}"
fi

echo ""; echo "===== 自检: 依赖可导入性 (只验证导入, 不验证算子能跑) ====="
set +e
python - <<'EOF' 2>&1 | tee "$RUN_DIR/depcheck.log"
import os
import sys
import traceback

sys.path.insert(0, os.path.join("third_party", "SegMAN"))
# 此处是一个新的 Python 进程。前面的环境打印进程 import torch 不会影响这里。
# 必须与生产模型一致先加载 torch, 让扩展依赖的 PyTorch 动态库和符号可用。
REQUIRED = ("torch", "einops", "triton", "selective_scan_cuda_oflex", "timm")
failed = {}
for mod in REQUIRED:
    try:
        m = __import__(mod)
        print(f"  {mod}: 可导入 {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  {mod}: 不可导入 -> {e!r}")
        traceback.print_exc()
        failed[mod] = repr(e)

# require_scan_ops 只检查依赖对象存在; 不是"算子已验证可运行"
try:
    from segman.mmscope_core import require_scan_ops
    require_scan_ops()
    print("  require_scan_ops(): 依赖齐全 (注意: 仅存在性检查, 未运行任何扫描)")
except Exception as e:
    print(f"  require_scan_ops(): 失败 -> {e}")
    traceback.print_exc()
    failed["require_scan_ops"] = repr(e)

if failed:
    print("")
    print("DEPCHECK: FAIL ->", ", ".join(sorted(set(failed))))
    for mod, reason in failed.items():
        print(f"  {mod}: {reason}")
    sys.exit(1)
print("")
print("DEPCHECK: OK (依赖可导入)")
EOF
DEPCHECK_STATUS=("${PIPESTATUS[@]}")
set -e
RC=${DEPCHECK_STATUS[0]}
if [ "${DEPCHECK_STATUS[1]}" -ne 0 ]; then
    echo "依赖自检日志写入失败 (tee exit=${DEPCHECK_STATUS[1]}, python exit=$RC)" >&2
    exit 1
fi
echo "依赖自检完整日志: $RUN_DIR/depcheck.log"

echo ""
if [ "$RC" -ne 0 ]; then
    cat <<'HINT'
===== 依赖自检失败, 请按缺失项处理 =====
  einops / timm 缺失        -> pip install einops / pip install timm
  triton 缺失               -> 不要盲目 pip install triton (版本需与当前 torch 匹配)。
                               先确认该 torch 构建是否自带 triton:
                                 python -c "import torch, triton; print(torch.__version__, triton.__version__)"
                               Linux 官方 wheel 通常自带匹配版本 (torch 2.8 -> triton 3.x)。
                               若确实缺失, 按 torch 版本选对应 triton, 或改用自带 triton 的 torch wheel
                               (重装 torch 会影响已编译的 groundingdino._C, 需同时重编, 请谨慎并先确认)。
  selective_scan 导入失败   -> 编译阶段已单独检查退出码; 若编译成功但导入失败,
                               请保留本次 build.log 和完整导入异常, 排查动态库/ABI 问题。
  use_mmscope=False 的旧 B0 / HED 路径不需要以上任何依赖, 仍可正常使用。
HINT
    exit "$RC"
fi

cat <<'NEXT'
===== 依赖就绪 =====
注意: 以上只证明依赖能导入。真实 CUDA 扫描算子的前向/反向尚未验证, 请接着在服务器上执行:
  python tools/check_ss2d.py          # 真实 SS2D / VSSMBlock 的前后向 + dtype + 非整除尺寸
  bash scripts/verify_mm_score.sh     # 完整验收 (含上面这一项)
use_mmscope=False 的旧 B0 / HED 路径不需要以上任何依赖。
NEXT
