"""B0 训练 / 验证入口。

    python tools/train.py --config configs/b0_refcoco.py [--opts key=value ...]
    python tools/train.py --config configs/b0_refcoco.py --eval-only --resume work_dirs/xxx/best.pth

--resume   同架构恢复: 模型 + 优化器 + 调度器 + epoch/best, 模式从 checkpoint 恢复。
--init-from 仅用权重初始化新架构 (例如用旧 B0_HED 权重初始化 MM/Score 新模型):
           结构完全由当前配置决定, 不恢复优化器 / epoch / best; 与 --resume 互斥,
           且不能与 --eval-only 同用 (随机初始化的新头不能冒充完整模型去评估)。
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.config import load_cfg  # noqa: E402
from rrs.training import build_grad_scaler, build_optimizer, to_device  # noqa: E402
from rrs.criterion import build_criterion  # noqa: E402
from rrs.data import RefSegDataset, collate_fn, load_samples  # noqa: E402
from rrs.evaluate import RefSegEvaluator, postprocess_for_model  # noqa: E402
from rrs.model import DEFORM_ATTN_PYTORCH_FALLBACK, build_model, init_model_weights, load_model_state  # noqa: E402
from rrs.model_mode import MODE_FLAGS, model_variant, resolve_model_mode  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, amp, max_iters=None, pp=None, hed_mask_mode="avg"):
    model.eval()
    ev = RefSegEvaluator()
    for it, (imgs, targets) in enumerate(loader):
        if max_iters and it >= max_iters:
            break
        imgs = [i.to(device) for i in imgs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(imgs, [t["caption"] for t in targets])
        res = postprocess_for_model(model, out, targets, pp, hed_mask_mode)
        ev.update(res, targets)
    model.train()
    return ev.summarize()


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*")
    ap.add_argument("--resume", default=None, help="同架构恢复 (模型+优化器+调度器+epoch)")
    ap.add_argument("--init-from", default=None, help="仅用该 checkpoint 的权重初始化当前配置的架构; 不恢复优化器/epoch")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-iters", type=int, default=None, help="每个 epoch 最多迭代数 (调试用)")
    args = ap.parse_args()
    if args.resume and args.init_from:
        sys.exit("--resume 与 --init-from 互斥: 前者恢复同架构训练, 后者只用权重初始化新架构")
    if args.init_from and args.eval_only:
        sys.exit("--init-from 不能与 --eval-only 同用: 新增模块是随机初始化的, 不能当成完整模型评估")
    cfg = load_cfg(args.config, args.opts)
    # 必须先读模式再构建模型；相同参数名称不能区分串行/并行结构。
    ck = torch.load(args.resume, map_location="cpu") if args.resume else None
    # --init-from 只取权重: 结构由新配置决定, 因此这里不把它的 cfg 传给 resolve_model_mode。
    init_ck = torch.load(args.init_from, map_location="cpu") if args.init_from else None
    cfg = resolve_model_mode(cfg, ck, args.opts)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg["amp"] and device == "cuda"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_f = open(os.path.join(cfg["output_dir"], "log.jsonl"), "a", encoding="utf-8")

    def log(d):
        print(json.dumps(d, ensure_ascii=False), flush=True)
        log_f.write(json.dumps(d, ensure_ascii=False) + "\n")
        log_f.flush()

    log({
        "use_hed": cfg["use_hed"],
        "hed_mask_mode": cfg["hed_mask_mode"],
        "use_dn": cfg["use_dn"],
        "dn": cfg["dn"] if cfg["use_dn"] else None,
        "use_mmscope": cfg["use_mmscope"],
        "mm": cfg["mm"] if cfg["use_mmscope"] else None,
        "use_ref_score": cfg["use_ref_score"],
        "ref_score_loss_coef": cfg["ref_score_loss_coef"] if cfg["use_ref_score"] else None,
        "model_variant": model_variant(cfg),
        "resume": args.resume,
        "init_from": args.init_from
    })
    model, load_info = build_model(cfg["gdino_cfg"], cfg["gdino_ckpt"], cfg["bert_path"], device,
                                   use_checkpoint=cfg["use_checkpoint"],
                                   use_hed=cfg.get("use_hed", False), use_dn=cfg["use_dn"],
                                   dn_cfg=cfg["dn"], use_mmscope=cfg["use_mmscope"],
                                   mm_cfg=cfg["mm"], use_ref_score=cfg["use_ref_score"])
    if load_info is not None:
        log({
            "pretrained": cfg["gdino_ckpt"],
            "missing_keys": load_info.missing_keys,
            "unexpected_keys": load_info.unexpected_keys
        })
    log({"deform_attn_pytorch_fallback": DEFORM_ATTN_PYTORCH_FALLBACK})
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg["lr_drop"], 0.1)
    # 兼容 PyTorch 2.1 的 GradScaler 接口。
    scaler = build_grad_scaler(amp)

    start_epoch, best = 0, -1.0
    if args.resume:
        # 只允许缺 dn_generator.* (从 HED-no-DN checkpoint 开启 DN 时); 其余不匹配直接报错
        dn_missing = load_model_state(model, ck["model"])
        if dn_missing:
            log({"resume_new_dn_params": dn_missing})
        if not args.eval_only and "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"] + 1
            best = ck.get("best", best)
    elif args.init_from:
        # weights-only: 逐类报告保留 / 新建 / 丢弃的键; 优化器与 epoch/best 保持初始状态。
        old_cfg = init_ck.get("cfg") or {}
        log({
            "init_from": args.init_from,
            "init_from_flags": {
                k: old_cfg.get(k)
                for k, _ in MODE_FLAGS
            },
            **init_model_weights(model, init_ck["model"], old_cfg)
        })
    del ck, init_ck

    val_ds = RefSegDataset(load_samples(cfg["val_data"]), cfg["val_data"]["image_root"],
                           train=False, short_side=cfg["short_side"], max_size=cfg["max_size"])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                             num_workers=cfg["num_workers"], collate_fn=collate_fn)
    if args.eval_only:
        log({
            "eval":
            evaluate(model, val_loader, device, amp, pp=cfg["postprocess"],
                     hed_mask_mode=cfg.get("hed_mask_mode", "avg"))
        })
        return

    train_ds = RefSegDataset(load_samples(cfg["train_data"]), cfg["train_data"]["image_root"],
                             train=True, short_side=cfg["short_side"], max_size=cfg["max_size"],
                             hflip=cfg["hflip"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                               num_workers=cfg["num_workers"],
                                               collate_fn=collate_fn, drop_last=True)
    log({
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "trainable_params_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    })

    model.train()
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for it, (imgs, targets) in enumerate(train_loader):
            if args.max_iters and it >= args.max_iters:
                break
            imgs = [i.to(device) for i in imgs]
            targets = to_device(targets, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                # targets 只被 HED+DN 训练前向用来生成 DN 查询; B0 / no-DN 忽略它
                out = model(imgs, [t["caption"] for t in targets], targets)
            loss, ld, _ = criterion(out, targets)
            if not math.isfinite(loss.item()):
                log({"error": "non-finite loss", "losses": {k: v.item() for k, v in ld.items()}})
                sys.exit(1)
            scaler.scale(loss / cfg["grad_accum"]).backward()
            if (it + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_max_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if it % cfg.get("log_every", 20) == 0:
                log({
                    "epoch":
                    epoch,
                    "iter":
                    it,
                    "loss":
                    loss.item(),
                    # 只记录最后一层 (无层号/interm 后缀); 含最后一层 DN 损失 loss_*_dn
                    **{
                        k: round(v.item(), 4)
                        for k, v in ld.items() if not k.endswith("_interm") and not k.rsplit(
                            "_", 1)[-1].isdigit()
                    },
                    "grad_norm":
                    float(gn) if (it + 1) % cfg["grad_accum"] == 0 else None,
                    "amp_scale":
                    scaler.get_scale() if amp else None,
                    "lr":
                    optimizer.param_groups[0]["lr"],
                    "mem_GB":
                    round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0
                })
        scheduler.step()
        ck = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "cfg": cfg
        }
        if (epoch + 1) % cfg["eval_every"] == 0:
            res = evaluate(model, val_loader, device, amp, pp=cfg["postprocess"],
                           hed_mask_mode=cfg.get("hed_mask_mode", "avg"))
            log({"epoch": epoch, "eval": res, "epoch_time_min": round((time.time() - t0) / 60, 2)})
            if res["mIoU"] > best:
                best = res["mIoU"]
                ck["best"] = best
                torch.save(ck, os.path.join(cfg["output_dir"], "best.pth"))
        torch.save(ck, os.path.join(cfg["output_dir"], "last.pth"))


if __name__ == "__main__":
    main()
