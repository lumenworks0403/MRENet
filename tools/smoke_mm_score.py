"""MM / 评分头 服务器短测试: 真实模型、真实数据、真实 SS2D 算子, 最多 2 个 batch 后退出。

optimizer.step 默认调用 (核对参数确实更新), 加 --no-step 可跳过。

    python tools/smoke_mm_score.py --config configs/b0_hed_mm_score_rislad.py [--iters 2]
    python tools/smoke_mm_score.py --config configs/b0_hed_mm_rislad.py
    python tools/smoke_mm_score.py --config configs/b0_hed_score_rislad.py
    python tools/smoke_mm_score.py --config configs/b0_hed_rislad.py      # 旧路径回归

有限梯度时使用 tools/train.py 的 scale->backward->unscale_->clip->step->update 路径。
非有限梯度时先记录原始梯度, 不做会污染其他参数的裁剪; AMP 由 scaler 跳过 step,
FP32 则显式阻止 step。--amp-init-scale 只用于诊断, 默认仍使用正式训练的 65536。
grad_accum 压到 min(cfg.grad_accum, iters) 以保证本次测试内至少一次更新机会
(正式训练仍用配置值, 生产逻辑未改)。AMP 关闭时走同一条代码路径 (scaler.enabled=False)。

检查 (结果打印为 JSON, 任一失败则 exit 1):
  1. 模式与实际模型一致; MM 开启时像素支路确实是 MMSCopEPixelDecoder 且含真实 SS2D 参数。
  2. 训练前向: 评分字段形状 (B,Q) / (L,B,Q); DN 开启时评分只覆盖普通查询。
  3. 总损失有限; **评分损失进入 total loss 的数值等式**: 固定并恢复同一份 CPU+CUDA 随机状态后
     两次调用 criterion, 断言 total 差值 ≈ Σ(各层评分损失)×权重 (容差内), 且其余损失项逐项不变。
  4. 梯度在 unscale_ 之后逐参数分类报告 (有梯度 / None 且冻结 / None 且可训练 / 全零 / NaN-Inf);
     任何应参与训练的参数出现 NaN/Inf 即失败; 模块完全无梯度 (断路) 即失败;
     新增模块 (评分头、SS2D 核心) 不允许"可训练但无梯度"。
  5. 新增模块的代表性参数确实在优化器里, 且 optimizer.step 后发生有限的实际更新;
     MM 开启时必须覆盖全部五个官方 SS2D 参数 (只看第一个投影层不算);
     被 scaler 因非有限梯度跳过的 step 判为失败。
  6. eval 前向不需要 targets; 新评分启用时 top1 由新评分决定 (手工改写 hed_all_ref_logits
     后 q* 必须跟着变), token 分数只作诊断。
  7. checkpoint 往返 (内存中, 不写盘): use_mmscope / mm / use_ref_score 正确恢复。
  8. 显存峰值记录。

不保存权重, 不写训练日志, 不影响任何已有 work_dirs (报告写 work_dirs/smoke_mm_score_*.json)。
本机 (Windows) 未执行本脚本。
"""
import argparse
import io
import json
import math
import os
import random
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.criterion import build_criterion  # noqa: E402
from rrs.data import RefSegDataset, collate_fn  # noqa: E402
from rrs.evaluate import postprocess_for_model  # noqa: E402
from rrs.model import build_model, load_model_state  # noqa: E402
from rrs.model_mode import model_variant, resolve_model_mode  # noqa: E402
from rrs.config import load_cfg  # noqa: E402
from rrs.data import load_samples  # noqa: E402
from rrs.training import build_grad_scaler, build_optimizer, to_device  # noqa: E402

# 模块分组: prefix -> (是否要求"至少有一个参数拿到有效梯度")
# 冻结的 gd.bert 单列, 它的参数 requires_grad=False, 没有梯度是**预期**行为。
BASE_GROUPS = [
    ("pixel_decoder", "pixel_decoder.", True),
    ("mask_head", "mask_head.", True),
    ("c4_norm", "c4_norm.", True),
    ("decoder_layer0", "gd.transformer.decoder.layers.0.", True),
    ("decoder_layer5", "gd.transformer.decoder.layers.5.", True),
    ("encoder", "gd.transformer.encoder.", True),
    ("swin", "gd.backbone.0.", True),
    ("bert(frozen)", "gd.bert.", False),
]


def _collect_grad_report(model, cfg):
    """逐参数分类报告梯度状态, 而不是只数"有几个非零"。

    对每个模块分组给出:
      n_params / n_trainable          参数总数与可训练数
      with_grad                       有限且非零 (计数)
      none                            .grad is None 的参数名 —— 再细分:
                                        frozen: requires_grad=False (预期, 如冻结的 BERT)
                                        unused: 可训练但本次没参与反传 (需人工判断是否合理)
      all_zero                        梯度有限但全零的参数名 (单个 batch 下可能是正常的)
      nonfinite                       出现 NaN/Inf 的参数名 —— **任何应参与训练的参数出现即失败**

    判定 ok 的规则:
      - 任何 requires_grad=True 的参数出现 NaN/Inf -> 失败;
      - require_grad 的分组里至少要有一个参数拿到有效梯度 -> 否则失败 (断路);
      - 全零 **不**作为失败条件 (单 batch 下某些参数本就可能零梯度);
      - unused (可训练但无梯度) 只记录, 不直接失败, 但新增模块 (MM 核心 / 评分头) 例外, 见下。
    """
    # 全模型组兜底, 防止 input_proj 等未单列的参数出现 NaN/Inf 被漏掉。
    groups = [("all_parameters", "", True)] + list(BASE_GROUPS)
    if cfg["use_ref_score"]:
        groups.append(("ref_score_head", "ref_score_head.", True))
    if cfg["use_mmscope"]:
        # 真实 SS2D 的核心参数单独成组, 必须拿到有效梯度
        groups.append(("mm_ss2d_core", "pixel_decoder.vssm.token_mixer.", True))
    if cfg["use_dn"]:
        groups.append(("dn_generator", "dn_generator.", True))

    named = list(model.named_parameters())
    out, ok = {}, True
    for name, prefix, need_grad in groups:
        rec = {
            "n_params": 0,
            "n_trainable": 0,
            "with_grad": 0,
            "none_frozen": [],
            "none_unused": [],
            "all_zero": [],
            "nonfinite": []
        }
        for n, p in named:
            if not n.startswith(prefix):
                continue
            rec["n_params"] += 1
            rec["n_trainable"] += int(p.requires_grad)
            if p.grad is None:
                (rec["none_frozen"] if not p.requires_grad else rec["none_unused"]).append(n)
            elif not torch.isfinite(p.grad).all():
                rec["nonfinite"].append(n)
            elif p.grad.abs().sum().item() == 0:
                rec["all_zero"].append(n)
            else:
                rec["with_grad"] += 1
        # 完整保留参数名, 便于定位; 不再仅保存前 8 个。
        for k in ("none_frozen", "none_unused", "all_zero", "nonfinite"):
            rec[f"n_{k}"] = len(rec[k])
        rec["fail_reasons"] = []
        if rec["n_nonfinite"]:
            rec["fail_reasons"].append("NaN/Inf 梯度")
        if need_grad and rec["n_params"] and rec["with_grad"] == 0 and not rec["n_nonfinite"]:
            rec["fail_reasons"].append("该模块没有任何参数拿到有效梯度 (疑似断路)")
        if name in ("ref_score_head", "mm_ss2d_core") and rec["n_none_unused"]:
            # 新增模块必须全部参与反传; 这里不容忍"可训练但没梯度"
            rec["fail_reasons"].append("新增模块存在可训练但无梯度的参数")
        rec["ok"] = not rec["fail_reasons"]
        ok &= rec["ok"]
        out[name] = rec
    out["ok"] = bool(ok)
    return out


def _finish_accumulation(model, optimizer, scaler, cfg, do_step=True):
    """诊断一次累积边界: 先取未裁剪的梯度报告, 再尝试有限梯度更新。

    optimizer post-hook 只在真实 optimizer.step 被调用后触发,
    因而不会把 GradScaler 跳过的更新统计为成功。
    """
    scaler.unscale_(optimizer)
    rec = {
        "grad_report_stage": "after_unscale_before_clip",
        "grad_report": _collect_grad_report(model, cfg),
        "grad_norm": None,
        "clip_applied": False
    }
    raw_finite = rec["grad_report"]["all_parameters"]["n_nonfinite"] == 0
    if raw_finite:
        try:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_max_norm"],
                                                error_if_nonfinite=True)
        except RuntimeError as e:
            if "non-finite" not in str(e):
                raise
            # 参数元素有限也可能在范数求和时溢出; error_if_nonfinite=True 在修改梯度前报错。
            rec["clip_error"] = str(e)
        else:
            rec["grad_norm"] = float(gn)
            rec["clip_applied"] = True
    else:
        rec["clip_skip_reason"] = "未裁剪梯度已有 NaN/Inf, 保留原始诊断结果"

    rec["post_clip_nonfinite"] = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    safe_step = raw_finite and rec["clip_applied"] and not rec["post_clip_nonfinite"]
    scale_before = scaler.get_scale() if scaler.is_enabled() else None
    actual_steps = 0

    def counted_step(opt, args, kwargs):
        nonlocal actual_steps
        actual_steps += 1

    handle = optimizer.register_step_post_hook(counted_step)
    try:
        if do_step:
            if safe_step or (scaler.is_enabled() and not raw_finite):
                # AMP 的 unscale_ 已记录非有限梯度, 后一种情况应由 scaler 正常跳过 step。
                scaler.step(optimizer)
            else:
                rec["step_blocked_reason"] = "梯度或范数非有限, 不执行 optimizer.step"
        scaler.update()  # no-step 模式也重置本次 unscale 状态, 支持下一个累积边界。
    finally:
        handle.remove()
    rec["optimizer_step_called"] = actual_steps == 1
    rec["scaler"] = {
        "scale_before":
        scale_before,
        "scale_after":
        scaler.get_scale() if scaler.is_enabled() else None,
        "step_skipped_due_to_nonfinite_grad":
        bool(do_step and scaler.is_enabled() and not raw_finite and actual_steps == 0),
    }
    rec["update_ok"] = bool(rec["grad_report"]["ok"] and safe_step
                            and (not do_step or actual_steps == 1))
    optimizer.zero_grad(set_to_none=True)
    return rec


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--amp-init-scale", type=float, default=None,
                    help="仅诊断: 显式设置 GradScaler 初始倍率, 不改变正式训练配置")
    ap.add_argument("--no-step", action="store_true", help="跳过 optimizer.step 的更新检查")
    args = ap.parse_args()
    assert 1 <= args.iters <= 2, "短测试最多 2 个 batch"
    cfg = resolve_model_mode(load_cfg(args.config, args.opts), None, args.opts)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg["amp"] and device == "cuda"
    if args.amp_init_scale is not None:
        if not amp or not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
            ap.error("--amp-init-scale 要求启用 CUDA AMP, 且值为有限正数")
    report = {
        "variant": model_variant(cfg),
        "use_mmscope": cfg["use_mmscope"],
        "mm": cfg["mm"] if cfg["use_mmscope"] else None,
        "use_ref_score": cfg["use_ref_score"],
        "ref_score_loss_coef": cfg["ref_score_loss_coef"],
        "device": device,
        "amp": amp
    }
    ok = True

    model, info = build_model(cfg["gdino_cfg"], cfg["gdino_ckpt"], cfg["bert_path"], device,
                              use_checkpoint=cfg["use_checkpoint"], use_hed=cfg["use_hed"],
                              use_dn=cfg["use_dn"], dn_cfg=cfg["dn"],
                              use_mmscope=cfg["use_mmscope"], mm_cfg=cfg["mm"],
                              use_ref_score=cfg["use_ref_score"])
    report["pretrained_missing_keys"] = len(info.missing_keys) if info else None

    # ---- 1) 模块身份: 不能只是改了名字 ----
    report["pixel_decoder_class"] = type(model.pixel_decoder).__name__
    report["has_ref_score_head"] = model.ref_score_head is not None
    if cfg["use_mmscope"]:
        names = [n for n, _ in model.pixel_decoder.named_parameters()]
        # 真实 SS2D 的参数名 (官方 VSSM): A_logs / Ds / x_proj_weight / dt_projs_weight
        ss2d = [n for n in names if "token_mixer" in n]
        report["ss2d_param_names"] = sorted(n for n in ss2d if any(
            k in n for k in ("A_logs", "Ds", "x_proj_weight", "dt_projs_weight", "dt_projs_bias")))
        ok &= type(model.pixel_decoder).__name__ == "MMSCopEPixelDecoder"
        ok &= len(report["ss2d_param_names"]) == 5
        report["mm_params_M"] = round(
            sum(p.numel() for p in model.pixel_decoder.parameters()) / 1e6, 3)
    else:
        ok &= type(model.pixel_decoder).__name__ == "FPNPixelDecoder"
    ok &= model.ref_score_head is not None if cfg["use_ref_score"] else model.ref_score_head is None

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    # 默认沿用 tools/train.py 的 GradScaler 初始倍率, 显式参数仅用于诊断对照。
    scaler_kwargs = {} if args.amp_init_scale is None else {"init_scale": args.amp_init_scale}
    scaler = build_grad_scaler(amp, **scaler_kwargs)
    report["amp_initial_scale"] = scaler.get_scale() if amp else None
    report["diagnostic_scale_override"] = args.amp_init_scale
    # 梯度累积: 短测试只跑 args.iters 个 batch, 如果沿用配置里的 grad_accum (可能是 2 或 4)
    # 就可能一次 optimizer.step 都没发生 —— 那样"参数已更新"的结论是假的。
    # 这里把 grad_accum 压到 min(cfg.grad_accum, iters), 保证至少一次更新机会,
    # 并把实际使用值记入报告 (正式训练仍用配置里的值, 生产逻辑未改)。
    grad_accum = max(1, min(int(cfg["grad_accum"]), args.iters))
    report["grad_accum"] = {
        "cfg": int(cfg["grad_accum"]),
        "used_in_smoke": grad_accum,
        "iters": args.iters,
        "note": "短测试压低 grad_accum 以保证至少一次更新机会, 实际更新另行计数"
    }
    ds = RefSegDataset(load_samples(cfg["train_data"]), cfg["train_data"]["image_root"], train=True,
                       short_side=cfg["short_side"], max_size=cfg["max_size"], hflip=cfg["hflip"])
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                                         num_workers=0, collate_fn=collate_fn)
    nq = model.gd.num_queries
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 参数更新检查的观察点: 每个新增模块都取「代表性参数」, 其中 MM 必须包含真实 SS2D 的核心参数
    # (只看像素支路第一个投影层不能代表 SS2D 是否在训练)。
    watch = {}
    if cfg["use_mmscope"]:
        for pname in ("vssm.token_mixer.A_logs", "vssm.token_mixer.Ds",
                      "vssm.token_mixer.x_proj_weight", "vssm.token_mixer.dt_projs_weight",
                      "vssm.token_mixer.dt_projs_bias", "linear_fuse.0.weight", "out.weight"):
            p = dict(model.pixel_decoder.named_parameters()).get(pname)
            if p is not None:
                watch[f"pixel_decoder.{pname}"] = p
    else:
        watch["pixel_decoder.lateral4.weight"] = model.pixel_decoder.lateral4.weight
    if cfg["use_ref_score"]:
        watch["ref_score_head.net.0.weight"] = model.ref_score_head.net[0].weight
        watch["ref_score_head.net.2.weight"] = model.ref_score_head.net[2].weight
    watch["mask_head.mask_embed.0.weight"] = model.mask_head.mask_embed[0].weight
    report["watched_params"] = sorted(watch)
    # MM 开启时必须盯住全部五个官方 SS2D 参数, 否则这项检查不成立
    if cfg["use_mmscope"]:
        need = {
            f"pixel_decoder.vssm.token_mixer.{k}"
            for k in ("A_logs", "Ds", "x_proj_weight", "dt_projs_weight", "dt_projs_bias")
        }
        report["ss2d_watched_complete"] = need <= set(watch)
        ok &= report["ss2d_watched_complete"]
    before = {k: p.detach().clone() for k, p in watch.items()}
    # optimizer 覆盖检查: 新增模块的参数必须真的在某个参数组里
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    report["watched_in_optimizer"] = {k: (id(p) in in_opt) for k, p in watch.items()}
    ok &= all(report["watched_in_optimizer"].values())

    # ---- 2~4) 训练前向 / 损失 / 反传 ----
    model.train()
    iters, stepped, attempted = [], 0, 0
    for it, (imgs, targets) in enumerate(loader):
        if it >= args.iters:
            break
        imgs = [i.to(device) for i in imgs]
        targets = to_device(targets, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(imgs, [t["caption"] for t in targets], targets)
        rec = {
            "iter": it,
            "n_targets": [int(len(t["boxes"])) for t in targets],
            "pred_masks": list(out["pred_masks"].shape),
            "pad_hw": list(out["pad_mask"].shape[-2:])
        }
        if cfg["use_ref_score"]:
            rec["pred_ref_logits"] = list(out["pred_ref_logits"].shape)
            rec["hed_all_ref_logits"] = list(out["hed_all_ref_logits"].shape)
            # 评分只覆盖普通查询: 最后一维必须是 num_queries, 与 DN 数无关
            ok &= tuple(out["pred_ref_logits"].shape) == (len(targets), nq)
            ok &= tuple(out["hed_all_ref_logits"].shape)[1:] == (len(targets), nq)
            ok &= all(
                tuple(a["pred_ref_logits"].shape) == (len(targets), nq) for a in out["aux_outputs"])
        else:
            ok &= "pred_ref_logits" not in out and "hed_all_ref_logits" not in out
        if cfg["use_dn"]:
            rec["has_dn_outputs"] = "dn_outputs" in out
            ok &= ("dn_outputs" in out) or all(len(t["boxes"]) == 0 for t in targets)

        # ---- 评分损失是否真的进入 total loss: 固定随机数状态后两次调用, 做数值相等断言 ----
        # criterion 内部的匹配代价与掩码损失都有随机点采样, 所以两次调用前必须恢复**同一份**
        # CPU + CUDA 随机数状态, 否则其他损失项也会变, 差值无法解释。
        # 这里只改测试流程 (保存/恢复 RNG), 不改生产训练逻辑。
        if cfg["use_ref_score"]:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else None

            def _restore():
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

            w = criterion.weight_dict["loss_ref_score"]
            _restore()
            loss, ld, _ = criterion(out, targets)
            ref_keys = sorted(k for k in ld if k.startswith("loss_ref_score"))
            expect = sum(ld[k].item() for k in ref_keys) * w
            criterion.weight_dict["loss_ref_score"] = 0.0
            _restore()
            loss0, ld0, _ = criterion(out, targets)
            criterion.weight_dict["loss_ref_score"] = w
            delta = loss.item() - loss0.item()
            # 容差: 相对 1e-4, 并给一个与损失量级相称的绝对下限 (fp32 求和的累积误差)
            tol = max(1e-3, abs(expect) * 1e-4, abs(loss.item()) * 1e-6)
            # RNG 恢复是否生效的旁证: 关权重那次, 其余损失项应与第一次逐项相同
            other_max_diff = max((abs(ld[k].item() - ld0[k].item())
                                  for k in ld if not k.startswith("loss_ref_score")), default=0.0)
            rec["ref_loss_keys"] = len(ref_keys)
            rec["ref_loss_in_total"] = {
                "delta": delta,
                "expect": expect,
                "tol": tol,
                "abs_err": abs(delta - expect),
                "equal_within_tol": abs(delta - expect) <= tol,
                "other_losses_max_diff": other_max_diff,
                "rng_restore_effective": other_max_diff <= 1e-6
            }
            ok &= len(ref_keys) == len(model.gd.transformer.decoder.layers)
            ok &= expect > 0
            ok &= rec["ref_loss_in_total"]["equal_within_tol"]
            ok &= rec["ref_loss_in_total"]["rng_restore_effective"]
            # 关闭评分权重后不应再有 ref 项参与总损失, 但 key 仍在 (权重为 0)
            ok &= sorted(k for k in ld0 if k.startswith("loss_ref_score")) == ref_keys
        else:
            loss, ld, _ = criterion(out, targets)
            ok &= not [k for k in ld if "ref_score" in k]

        rec["loss_total"] = round(loss.item(), 3)
        rec["losses_last_layer"] = {
            k: round(v.item(), 4)
            for k, v in ld.items()
            if not k.endswith("_interm") and not k.rsplit("_", 1)[-1].isdigit()
        }
        ok &= math.isfinite(loss.item()) and all(torch.isfinite(v).all() for v in ld.values())

        # ---- 与 tools/train.py 一致的 AMP 反传路径 ----
        scaler.scale(loss / grad_accum).backward()
        rec["amp_scale_before_step"] = scaler.get_scale() if amp else None
        if (it + 1) % grad_accum == 0:
            update = _finish_accumulation(model, optimizer, scaler, cfg, do_step=not args.no_step)
            rec.update(update)
            ok &= update["update_ok"]
            attempted += int(not args.no_step)
            stepped += int(update["optimizer_step_called"])
        iters.append(rec)

    report["iters"] = iters
    report["optimizer_steps"] = stepped
    report["optimizer_step_attempts"] = attempted
    # AMP 与 FP32 走的是同一条代码路径: GradScaler(enabled=False) 时 unscale_/update 是空操作,
    # step 直接转发给 optimizer.step —— 与 tools/train.py 的行为一致。
    report["amp_path"] = {
        "amp": amp,
        "scaler_enabled": bool(scaler.is_enabled()),
        "final_scale": scaler.get_scale() if amp else None
    }
    # 梯度报告只在发生累积边界时产生; 一次都没产生说明本次测试其实没检查过梯度
    ok &= any("grad_report" in r for r in iters)
    report["grad_report_collected"] = sum("grad_report" in r for r in iters)
    if not args.no_step:
        # 至少完成一次真实更新, 否则"参数已更新"无从谈起
        ok &= stepped >= 1
        report["param_updated"] = {
            k: bool((watch[k] - before[k]).abs().max().item() > 0)
            for k in watch
        }
        report["param_update_maxabs"] = {
            k: float((watch[k] - before[k]).abs().max())
            for k in watch
        }
        report["param_update_finite"] = {
            k: bool(torch.isfinite(watch[k]).all() and torch.isfinite(watch[k] - before[k]).all())
            for k in watch
        }
        ok &= all(report["param_updated"].values())
        ok &= all(report["param_update_finite"].values())

    if device == "cuda":
        report["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # ---- 5) eval 前向与 top1 选择 ----
    model.eval()
    imgs, targets = next(iter(loader))
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model([i.to(device) for i in imgs], [t["caption"] for t in targets])
    report["eval_has_dn_outputs"] = "dn_outputs" in out
    ok &= "dn_outputs" not in out
    res = postprocess_for_model(model, out, targets, cfg["postprocess"], cfg["hed_mask_mode"])
    report["eval_selection"] = {
        "queries": [r["queries"] for r in res],
        # B0 的 postprocess 不写 score_type (旧路径), 默认按 token 记录
        "score_type": [r.get("score_type", "token") for r in res],
        "scores": [[round(s, 4) for s in r["scores"]] for r in res]
    }
    if cfg["use_ref_score"]:
        ok &= all(r.get("score_type") == "ref_score" for r in res)
        # 手工把某个查询的评分拉满 -> q* 必须跟着变 (证明选择依据是新评分, 不是 token 分数)
        forced = dict(out)
        ref = out["hed_all_ref_logits"].clone()
        target_q = (int(res[0]["queries"][0]) + 1) % ref.shape[-1]
        ref[:] = -20.0
        ref[:, :, target_q] = 20.0
        forced["hed_all_ref_logits"] = ref
        res2 = postprocess_for_model(model, forced, targets, cfg["postprocess"],
                                     cfg["hed_mask_mode"])
        report["forced_q_star"] = {"forced": target_q, "got": [r["queries"][0] for r in res2]}
        ok &= all(int(r["queries"][0]) == target_q for r in res2)
        report["token_scores_recorded"] = all("token_scores" in r for r in res)
        ok &= report["token_scores_recorded"]
    else:
        ok &= all(r.get("score_type", "token") == "token" for r in res)
        ok &= all("token_scores" not in r for r in res)

    # ---- 6) checkpoint 往返 (内存中) ----
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": 0}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cpu")
    cfg2 = resolve_model_mode({}, ck)
    report["roundtrip"] = {
        "variant": model_variant(cfg2),
        "use_mmscope": cfg2["use_mmscope"],
        "use_ref_score": cfg2["use_ref_score"],
        "mm_channels": cfg2["mm"]["channels"]
    }
    ok &= (cfg2["use_mmscope"] == cfg["use_mmscope"]
           and cfg2["use_ref_score"] == cfg["use_ref_score"]
           and model_variant(cfg2) == model_variant(cfg))
    missing = load_model_state(model, ck["model"])
    report["roundtrip_missing"] = missing
    ok &= not missing

    report["PASS"] = bool(ok)
    os.makedirs("work_dirs", exist_ok=True)
    name = os.path.splitext(os.path.basename(args.config))[0]
    # 每次生成独立报告, FP32/AMP 对照不会覆盖此前失败记录。
    with tempfile.NamedTemporaryFile(mode="w", prefix=f"smoke_mm_score_{name}_", suffix=".json",
                                     dir="work_dirs", delete=False, encoding="utf-8") as f:
        report["report_path"] = os.path.abspath(f.name)
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
