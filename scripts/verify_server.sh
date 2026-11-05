#!/usr/bin/env bash
# 服务器端 B0 验收 (在 B0 目录下, conda activate rrs_b0 后执行):
#   bash scripts/verify_server.sh
# 产出 work_dirs/verify_summary.txt, 把这个文件内容发回即可。
set -u
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/third_party/GroundingDINO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p work_dirs
OUT=work_dirs/verify_summary.txt
: > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

log "===== env ====="
python - <<'EOF' 2>&1 | tee -a "$OUT"
import torch, transformers, platform
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
try:
    from groundingdino import _C  # noqa
    print("groundingdino._C: OK (CUDA 可变形注意力)")
except Exception as e:
    print("groundingdino._C: FAILED ->", repr(e))
EOF
sha256sum weights/groundingdino_swint_ogc.pth | tee -a "$OUT"

log ""; log "===== 1) check_b0: 权重加载 / 前向 / 损失 / 反传 ====="
python tools/check_b0.py > work_dirs/check_b0.stdout 2>&1
log "exit=$?"
python - <<'EOF' 2>&1 | tee -a "$OUT"
import json
r = json.load(open("work_dirs/check_b0.json"))
for k in ["missing_keys", "unexpected_keys", "deform_attn_pytorch_fallback", "pretrained_zero_shot", "shapes",
          "loss_total", "losses_last_layer", "num_loss_terms", "peak_mem_GB", "nonfinite_grads", "param_updated"]:
    print(k, "=", json.dumps(r.get(k), ensure_ascii=False))
for k, v in r["grad_check"].items():
    print("grad", k, v)
EOF
[ $? -ne 0 ] && tail -30 work_dirs/check_b0.stdout | tee -a "$OUT"

run_overfit(){  # $1 config  $2 work_dir
  local run_dir
  run_dir=$(mktemp -d "$2.XXXXXXXX") || return 1
  python tools/train.py --config "$1" --opts "output_dir=$run_dir" > "$run_dir/stdout.log" 2>&1
  log "exit=$?"
  log "run_dir=$run_dir"
  python - "$run_dir/log.jsonl" <<'EOF' 2>&1 | tee -a "$OUT"
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
for d in L:
    if "deform_attn_pytorch_fallback" in d or "train_samples" in d or "error" in d:
        print(json.dumps(d, ensure_ascii=False)[:600])
it = [d for d in L if "iter" in d and d["iter"] == 0]
print("loss@iter0 per epoch:", [round(d["loss"], 2) for d in it])
for d in L:
    if "eval" in d:
        print("epoch", d["epoch"], {k: round(v, 1) for k, v in d["eval"].items()}, "min/epoch", d.get("epoch_time_min"))
print("peak mem GB:", max((d.get("mem_GB") or 0) for d in L if "iter" in d) if it else None)
EOF
  grep -n "Traceback" -A20 "$run_dir/stdout.log" | tail -25 | tee -a "$OUT"
}

log ""; log "===== 2) RefCOCO 16 样本过拟合 (单目标, 真实掩码) ====="
run_overfit configs/b0_overfit.py work_dirs/overfit

log ""; log "===== 3) RefDrone 51 样本过拟合 (多目标/无目标, 框掩码) ====="
run_overfit configs/b0_overfit_refdrone.py work_dirs/overfit_refdrone

log ""; log "done -> $OUT"
