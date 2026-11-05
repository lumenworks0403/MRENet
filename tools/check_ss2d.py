"""真实 SS2D / VSSMBlock 算子自检 (需要 GPU + 编译好的 selective_scan_cuda_oflex + triton)。

    python tools/check_ss2d.py [--iters 1]

这是与"依赖能否导入"分开的一项检查: scripts/setup_mmscope.sh 只验证导入,
本脚本实际运行官方 VSSM(SS2D) 与 VSSMBlock 的前向和反向。

检查 (结果打印为 JSON, 任一失败 exit 1):
  1. 真实算子就位: require_scan_ops() 通过, 且 VSSM 具备官方的五个 SS2D 参数
     (A_logs / Ds / x_proj_weight / dt_projs_weight / dt_projs_bias)。
  2. VSSM 前向输出形状 = 输入形状; 非方形、非整除 (奇数、素数) H/W 都要通过。
  3. VSSM 反向: 输入梯度与五个 SS2D 参数的梯度都有限且非零。
  4. dtype: FP32 输入正常; 在 autocast(fp16) 下直接调用官方 VSSM 是否可行 —— 本项记录实测结果
     (这正是 mmscope_pixel_decoder 用 mm.scan_fp32=True 规避的问题, 不作为失败条件)。
  5. VSSMBlock (norm + cpe + VSSM + FFN) 的前后向。
  6. **同权重前后向对照**: 同一份 VSSM 权重、同一输入跑两次, 结果必须逐元素相同 (确定性);
     再与 cross_scan -> conv1d -> selective_scan -> cross_merge 的分步重算对照,
     核对本项目抽取的 VSSM 与官方 forward 的计算顺序一致。
     注意: 这是"抽取代码 vs 官方同一份代码"的自我一致性对照, **不是**与官方完整
     SegMANDecoder 的数值等价证明 —— 后者因 SyncBN→GroupNorm 等适配无法逐元素相等。

本机 (Windows) 未执行本脚本。
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "SegMAN"))

OFFICIAL_SS2D_PARAMS = ("A_logs", "Ds", "x_proj_weight", "dt_projs_weight", "dt_projs_bias")


def _grad_report(named_params):
    """返回 {有梯度, None 的, 全零的, 非有限的} —— 与 smoke 脚本同一口径。"""
    out = {"with_grad": [], "none": [], "all_zero": [], "nonfinite": []}
    for n, p in named_params:
        if p.grad is None:
            out["none"].append(n)
        elif not torch.isfinite(p.grad).all():
            out["nonfinite"].append(n)
        elif p.grad.abs().sum().item() == 0:
            out["all_zero"].append(n)
        else:
            out["with_grad"].append(n)
    return out


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384, help="扫描通道数 (默认 3*128, 与 mm.channels=128 一致)")
    args = ap.parse_args()
    report, ok = {}, True

    if not torch.cuda.is_available():
        print(
            json.dumps({
                "PASS": False,
                "error": "真实 SS2D 需要 CUDA, 当前无可用 GPU"
            }, ensure_ascii=False, indent=2))
        sys.exit(1)
    dev = "cuda"

    # ---- 1) 真实算子就位 ----
    from segman.mmscope_core import VSSM, VSSMBlock, require_scan_ops
    try:
        require_scan_ops()
        report["require_scan_ops"] = "OK (仅存在性检查)"
    except Exception as e:
        print(json.dumps({
            "PASS": False,
            "require_scan_ops": repr(e)
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    torch.manual_seed(0)
    vssm = VSSM(d_model=args.dim, k_groups=4, expansion_ratio=1).to(dev)
    names = dict(vssm.named_parameters())
    report["ss2d_params"] = {k: list(names[k].shape) for k in OFFICIAL_SS2D_PARAMS if k in names}
    ok &= all(k in names for k in OFFICIAL_SS2D_PARAMS)
    report["ss2d_params_complete"] = all(k in names for k in OFFICIAL_SS2D_PARAMS)

    # ---- 2/3) 非方形、非整除尺寸的前后向 ----
    shapes = [(1, 25, 42), (2, 25, 42), (1, 7, 13), (2, 13, 13), (1, 1, 5)]
    cases = []
    for b, h, w in shapes:
        x = torch.randn(b, args.dim, h, w, device=dev, requires_grad=True)
        vssm.zero_grad(set_to_none=True)
        y = vssm(x)
        case = {
            "in": [b, args.dim, h, w],
            "out": list(y.shape),
            "shape_ok": tuple(y.shape) == (b, args.dim, h, w),
            "out_finite": bool(torch.isfinite(y).all())
        }
        y.float().pow(2).mean().backward()
        case["x_grad_finite_nonzero"] = bool(x.grad is not None and torch.isfinite(x.grad).all()
                                             and x.grad.abs().sum() > 0)
        g = _grad_report([(k, names[k]) for k in OFFICIAL_SS2D_PARAMS if k in names])
        case["param_grad"] = {k: v for k, v in g.items() if v}
        case["param_grad_ok"] = not g["none"] and not g["nonfinite"] and not g["all_zero"]
        ok &= case["shape_ok"] and case["out_finite"] and case["x_grad_finite_nonzero"] \
            and case["param_grad_ok"]
        cases.append(case)
    report["vssm_cases"] = cases

    # ---- 4) dtype: autocast 下直接调官方 VSSM 的实测结果 (记录, 不作为失败条件) ----
    x = torch.randn(1, args.dim, 25, 42, device=dev)
    with torch.autocast("cuda", dtype=torch.float16):
        try:
            y = vssm(x)
            report["autocast_fp16_direct"] = {
                "ran": True,
                "out_dtype": str(y.dtype),
                "finite": bool(torch.isfinite(y).all())
            }
        except Exception as e:
            # 预期可能在这里失败: selective_scan_cuda_oflex 要求 u/delta/B/C dtype 一致,
            # 而 autocast 下 LayerNorm 产 fp32、conv1d 产 fp16。这正是 mm.scan_fp32=True 的理由。
            report["autocast_fp16_direct"] = {"ran": False, "error": repr(e)[:300]}
    # 项目实际走的路径: 关闭 autocast 并转 fp32
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.autocast("cuda", enabled=False):
            y = vssm(x.float())
    report["scan_fp32_path"] = {
        "out_dtype": str(y.dtype),
        "finite": bool(torch.isfinite(y).all()),
        "shape_ok": tuple(y.shape) == tuple(x.shape)
    }
    ok &= report["scan_fp32_path"]["finite"] and report["scan_fp32_path"]["shape_ok"]

    # ---- 5) VSSMBlock 前后向 ----
    torch.manual_seed(0)
    block = VSSMBlock(embed_dim=args.dim, expansion_ratio=1, channel_split=False).to(dev)
    xb = torch.randn(2, args.dim, 25, 42, device=dev, requires_grad=True)
    yb = block(xb)
    yb.float().pow(2).mean().backward()
    gb = _grad_report(list(block.named_parameters()))
    report["vssm_block"] = {
        "shape_ok":
        tuple(yb.shape) == tuple(xb.shape),
        "out_finite":
        bool(torch.isfinite(yb).all()),
        "x_grad_finite_nonzero":
        bool(xb.grad is not None and torch.isfinite(xb.grad).all() and xb.grad.abs().sum() > 0),
        "n_with_grad":
        len(gb["with_grad"]),
        "none":
        gb["none"],
        "all_zero":
        gb["all_zero"],
        "nonfinite":
        gb["nonfinite"]
    }
    ok &= (report["vssm_block"]["shape_ok"] and report["vssm_block"]["out_finite"]
           and report["vssm_block"]["x_grad_finite_nonzero"] and not gb["nonfinite"]
           and not gb["none"])

    # ---- 6) 同权重对照: 确定性 + 与官方 forward 分步重算一致 ----
    x = torch.randn(1, args.dim, 17, 29, device=dev)
    with torch.no_grad():
        y1, y2 = vssm(x), vssm(x)
        report["deterministic_same_weights"] = {
            "max_abs_diff": float((y1 - y2).abs().max()),
            "equal": bool(torch.equal(y1, y2))
        }
        ok &= report["deterministic_same_weights"]["equal"]

        # 按官方 VSSM.forward 的步骤手工重算一次, 核对计算顺序 (同一份权重, 同一输入)
        import torch.nn.functional as F
        B, _, H, W = x.shape
        _, N = vssm.A_logs.shape
        K, Dd, R = vssm.dt_projs_weight.shape
        L = H * W
        # 逐步照 mmscope_core.py::VSSM.forward (= 官方 segman_decoder.py L604-655) 的顺序重算
        xs = vssm._cross_scan(x).reshape(B, -1, L)
        x_dbl = F.conv1d(xs, vssm.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), vssm.dt_projs_weight.reshape(K * Dd, -1, 1), groups=K)
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(vssm.A_logs.to(torch.float))
        ys = vssm._selective_scan(xs, dts, As,
                                  Bs.contiguous().reshape(B, K, N, L),
                                  Cs.contiguous().reshape(B, K, N, L), vssm.Ds.to(torch.float),
                                  vssm.dt_projs_bias.reshape(-1).to(torch.float),
                                  delta_softplus=True, ssoflex=True)
        y_manual = vssm._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)
        report["stepwise_recompute_matches"] = {
            "max_abs_diff": float((y1 - y_manual).abs().max()),
            "equal": bool(torch.equal(y1, y_manual))
        }
        ok &= report["stepwise_recompute_matches"]["equal"]
    report["note"] = ("第 6 项是抽取代码的自我一致性对照 (确定性 + 计算顺序), "
                      "不是与官方完整 SegMANDecoder 的数值等价证明; "
                      "后者因 SyncBN->GroupNorm 等适配无法逐元素相等, 仍列为待验证。")

    report["PASS"] = bool(ok)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    os.makedirs("work_dirs", exist_ok=True)
    with open("work_dirs/check_ss2d.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
