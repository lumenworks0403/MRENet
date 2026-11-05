#!/usr/bin/env bash
# MM / 评分头 服务器验收。在 B0_HED 目录下, conda activate rrs_b0 后执行:
#   bash scripts/verify_mm_score.sh
#   OLD_CKPT=work_dirs/b0_hed_rislad_noDN/best.pth bash scripts/verify_mm_score.sh
#
# 产出 (每次运行一个独立目录, 不复用上一次的残留):
#   work_dirs/verify_mm_<时间戳>/summary.txt   PASS/FAIL/SKIP 汇总 (把这个发回即可)
#   work_dirs/verify_mm_<时间戳>/<项目>.log    每项的完整输出 (不截断)
#   work_dirs/verify_mm_<时间戳>/tmp_*/        本次新生成的 checkpoint 等中间产物
#
# 退出码: 所有项 PASS 或 SKIP -> 0; 任何一项 FAIL -> 1。
# SKIP 只用于"前置条件缺失"(例如没有旧 checkpoint), summary 里会写明如何补齐;
# **SKIP 绝不等于通过**。
#
# 本机 (Windows) 未执行本脚本中的任何命令。
set -uo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=work_dirs/verify_mm_${TS}
mkdir -p "$RUN_DIR" || { echo "无法创建 $RUN_DIR"; exit 1; }
OUT="$RUN_DIR/summary.txt"
: > "$OUT" || { echo "无法写入 $OUT"; exit 1; }

FAILED=0
declare -a ROWS=()

log(){   # 写 summary + 终端
    printf '%s\n' "$*" | tee -a "$OUT" >/dev/null || { echo "写入 $OUT 失败"; exit 1; }
    printf '%s\n' "$*"
}

record(){   # record <状态> <项目名> <退出码或-> <备注>
    ROWS+=("$1|$2|$3|$4")
    if [ "$1" = "FAIL" ]; then FAILED=1; fi
    log "[$1] $2 (exit=$3) $4"
    return 0
}

run(){   # run <项目id> <说明> -- <命令...>
    local id="$1"; shift
    local desc="$1"; shift
    [ "${1:-}" = "--" ] && shift
    local logf="$RUN_DIR/$id.log"
    log ""; log "===== $id: $desc ====="
    # 完整输出写文件 (不截断), 终端只显示末尾摘要。
    # 退出码必须在任何后续命令之前取出, 否则 PIPESTATUS 会被覆盖。
    "$@" > "$logf" 2>&1
    local rc=$?
    tail -25 "$logf"
    log "  (完整日志: $logf)"
    if [ "$rc" -eq 0 ]; then record PASS "$id" "$rc" "$desc"
    else                   record FAIL "$id" "$rc" "$desc; 见 $logf"
    fi
    return 0   # 失败后继续收集其余结果
}

skip(){  # skip <项目id> <说明> <如何补齐>
    log ""; log "===== $1: SKIPPED ====="
    log "  原因/补齐方式: $3"
    record SKIP "$1" "-" "$2; $3"
}

log "验收开始 $TS  输出目录 $RUN_DIR"
log ""
log "===== env ====="
python - > "$RUN_DIR/env.log" 2>&1 <<'EOF'
import os, sys, platform, torch
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
sys.path.insert(0, os.path.join("third_party", "SegMAN"))
for mod in ("einops", "triton", "selective_scan_cuda_oflex", "timm"):
    try:
        m = __import__(mod); print(f"{mod}: 可导入", getattr(m, "__version__", ""))
    except Exception as e:
        print(f"{mod}: 不可导入 -> {e!r}")
EOF
cat "$RUN_DIR/env.log" | tee -a "$OUT" >/dev/null
cat "$RUN_DIR/env.log"

# ======================================================================
# 1) 旧路径回归: 不需要 MM 依赖, 旧环境也应全部通过
# ======================================================================
run 1a_mode_entrypoints "模式/入口回归 (无 torch 依赖)" -- python tests/test_mode_entrypoints.py
run 1b_test_hed        "原 HED 测试 (未改动)"            -- python tests/test_hed.py
run 1c_test_dn         "原 DN 测试 (未改动)"             -- python tests/test_dn.py
run 1d_test_mm_score   "MM/评分头 控制流测试 (CPU, SS2D 用卷积桩)" -- python tests/test_mm_score.py
run 1e_check_b0        "旧 B0 组件验收 (新开关全关)"     -- python tools/check_b0.py --n 4
run 1f_no_mm_import    "use_mmscope=False 不触碰 triton / selective_scan" -- \
    python -c "
import sys
import rrs.model  # noqa: F401
bad = [m for m in ('triton', 'selective_scan_cuda_oflex') if m in sys.modules]
seg = [m for m in sys.modules if m.startswith('segman')]
print('导入 rrs.model 后已加载的 MM 依赖:', bad or '无')
print('已导入的 segman 模块:', seg or '无')
assert not bad, f'旧路径不应加载 MM 依赖: {bad}'
assert not seg, f'旧路径不应加载 segman: {seg}'
print('OK')
"

# ======================================================================
# 2) 真实 CUDA 扫描算子 (与"依赖能导入"分开的一项)
# ======================================================================
run 2a_check_ss2d "真实 SS2D/VSSMBlock 前后向 + dtype + 非整除尺寸 + 同权重对照" -- \
    python tools/check_ss2d.py

# ======================================================================
# 3) 四组对照的真实短测试 (真实模型 + 真实数据 + GradScaler 路径, 最多 2 个 batch)
# ======================================================================
run 3a_smoke_base     "对照1 原 B0_HED (MM 关 / Score 关)" -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_rislad.py --iters 2
run 3b_smoke_mm       "对照2 + MM"                        -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_rislad.py --iters 2
run 3c_smoke_score    "对照3 + Score"                     -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_score_rislad.py --iters 2
run 3d_smoke_mm_score "对照4 + MM + Score"                -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py --iters 2

# ======================================================================
# 4) HED+DN 兼容性 / batch / AMP / 尺寸
# ======================================================================
run 4a_dn_mm_score "HED+DN + MM + Score (DN 不得混入评分与掩码)" -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py --iters 2 --opts use_dn=True
run 4b_smoke_dn    "原 DN 短测试 (确认 DN 路径未被破坏)" -- \
    python tools/smoke_dn.py --config configs/b0_hed_dn_rislad.py --iters 2
run 4c_batch1      "batch=1" -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py --iters 2 --opts batch_size=1
run 4d_fp32        "AMP 关闭 (FP32 对照, 同一条 GradScaler 路径)" -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py --iters 2 --opts amp=False
run 4e_small_input "另一组非整除尺寸" -- \
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py --iters 2 \
      --opts short_side=600 max_size=1000

# ======================================================================
# 5) 权重初始化 / 恢复 / eval-only / 单图推理
#    全部写入本次独立目录, 不读取也不覆盖任何历史 work_dirs
# ======================================================================
INIT_DIR="$RUN_DIR/tmp_initfrom"
OLD=${OLD_CKPT:-work_dirs/b0_hed_rislad_noDN/last.pth}

if [ -f "$OLD" ]; then
    run 5a_init_from "用旧 HED 权重初始化 MM+Score 新模型 (--init-from, 2 iter)" -- \
        python tools/train.py --config configs/b0_hed_mm_score_rislad.py \
            --init-from "$OLD" --max-iters 2 \
            --opts epochs=1 eval_every=99 "output_dir=$INIT_DIR"
    # 初始化日志摘要 (kept_groups / new / dropped / pixel_decoder_replaced)
    if [ -f "$INIT_DIR/log.jsonl" ]; then
        log "  --- init_from 日志摘要 ---"
        python - "$INIT_DIR/log.jsonl" <<'EOF' 2>&1 | tee -a "$OUT"
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line)
    if "init_from" in d or "model_variant" in d:
        print("   ", json.dumps(d, ensure_ascii=False)[:1200])
EOF
    fi
else
    skip 5a_init_from "--init-from 未执行" \
      "缺少旧 HED checkpoint ($OLD)。先训出一个 (python tools/train.py --config configs/b0_hed_rislad.py), 或用 OLD_CKPT=<路径> bash scripts/verify_mm_score.sh 指定。"
fi

# 本次新生成的 checkpoint —— 必须是 5a 刚产生的, 不接受任何历史残留
NEW="$INIT_DIR/last.pth"
if [ -f "$NEW" ]; then
    run 5b_resume "同架构恢复 --resume (2 iter)" -- \
        python tools/train.py --config configs/b0_hed_mm_score_rislad.py --resume "$NEW" --max-iters 2 \
            --opts epochs=2 eval_every=99 "output_dir=$INIT_DIR"
    run 5c_eval_only "eval-only (模式必须与 checkpoint 一致)" -- \
        python tools/train.py --config configs/b0_hed_mm_score_rislad.py --resume "$NEW" --eval-only \
            --opts "output_dir=$INIT_DIR" val_data.max_samples=8

    # 从 val.jsonl 取第一条做单图推理
    python - > "$RUN_DIR/infer_input.txt" 2>"$RUN_DIR/infer_input.err" <<'EOF'
import json
s = [json.loads(l) for l in open("data/RIS-LAD/b0/val.jsonl", encoding="utf-8") if l.strip()][0]
print("data/RIS-LAD/" + s["image"])
print(s["caption"])
EOF
    if [ -s "$RUN_DIR/infer_input.txt" ]; then
        IMGP=$(head -1 "$RUN_DIR/infer_input.txt")
        CAP=$(tail -1 "$RUN_DIR/infer_input.txt")
        run 5d_infer "单图推理 (score_type 应为 ref_score)" -- \
            python tools/infer.py --ckpt "$NEW" --image "$IMGP" --text "$CAP" \
                --out "$RUN_DIR/vis.jpg"
    else
        skip 5d_infer "单图推理未执行" "无法从 data/RIS-LAD/b0/val.jsonl 读取样本, 见 $RUN_DIR/infer_input.err"
    fi

    # 权重缺失必须报错 —— 这一项用断言, 不报错就失败; 且必须检查异常信息里确实提到缺失的键
    run 5e_missing_key_raises "删除权重键后 load_model_state 必须报错 (断言, 含键名核对)" -- \
        python tools/check_strict_loading.py --ckpt "$NEW"
else
    skip 5b_resume "同架构恢复未执行" "依赖 5a 本次新生成的 $NEW (不使用历史残留 checkpoint)"
    skip 5c_eval_only "eval-only 未执行" "依赖 5a 本次新生成的 $NEW"
    skip 5d_infer "单图推理未执行" "依赖 5a 本次新生成的 $NEW"
    skip 5e_missing_key_raises "严格加载检查未执行" "依赖 5a 本次新生成的 $NEW"
fi

# ======================================================================
# 汇总
# ======================================================================
log ""
log "======================================================================"
log "验收汇总 (SKIP 不等于通过)"
log "======================================================================"
printf '%-8s %-26s %-7s %s\n' "状态" "项目" "exit" "说明" | tee -a "$OUT"
# ${ROWS[@]+"${ROWS[@]}"} 在 set -u 下对空数组也安全 (老版本 bash)
for row in ${ROWS[@]+"${ROWS[@]}"}; do
    IFS='|' read -r st id rc note <<< "$row"
    printf '%-8s %-26s %-7s %s\n' "$st" "$id" "$rc" "$note" | tee -a "$OUT"
done
# 用纯 bash 计数, 避免 grep -c 无匹配时的退出码干扰
N_PASS=0; N_FAIL=0; N_SKIP=0
for row in ${ROWS[@]+"${ROWS[@]}"}; do
    case "${row%%|*}" in
        PASS) N_PASS=$((N_PASS + 1)) ;;
        FAIL) N_FAIL=$((N_FAIL + 1)) ;;
        SKIP) N_SKIP=$((N_SKIP + 1)) ;;
    esac
done
log ""
log "PASS=$N_PASS  FAIL=$N_FAIL  SKIP=$N_SKIP"
log "完整日志: $RUN_DIR/*.log"
log ""
log "未包含在本脚本中的检查 (需要时手动执行, 耗时较长):"
log "  小样本过拟合: python tools/train.py --config configs/b0_hed_mm_score_rislad.py \\"
log "    --opts train_data.max_samples=16 val_data.max_samples=16 epochs=20 eval_every=5 \\"
log "           output_dir=work_dirs/overfit_mm_score"
log "  四组 12 轮正式对照训练: 见 docs/MMSCopE与评分头接入说明_20261016.md 第 6.4 节"
log "  与官方完整 SegMANDecoder 的逐元素数值对照: 未实现 (SyncBN->GroupNorm 等适配使其不成立)"

if [ "$FAILED" -ne 0 ]; then
    log ""; log "结论: FAIL (有 $N_FAIL 项失败)"
    exit 1
fi
log ""; log "结论: 无失败项 (PASS=$N_PASS, SKIP=$N_SKIP)"
exit 0
