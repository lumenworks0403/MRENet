"""B0 真实组件验收测试 (不使用随机张量):
  1) 官方权重加载 (missing/unexpected keys)
  2) 预训练 GroundingDINO 在真实 RefCOCO 样本上的检测是否合理 (框 Acc@0.5, 验证组件适配未破坏预训练行为)
  3) 完整模型前向 + 匹配 + 损失 + 反向传播, 检查梯度到达 新增模块 / 解码器 / encoder / Swin, BERT 冻结
  4) 一次优化器 step 后参数确实更新

    python tools/check_b0.py [--n 8]
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.data import RefSegDataset, collate_fn, load_refcoco_parquet  # noqa: E402
from rrs.evaluate import RefSegEvaluator, postprocess  # noqa: E402
from rrs.config import load_cfg  # noqa: E402
from rrs.training import build_optimizer, to_device  # noqa: E402
from rrs.criterion import build_criterion  # noqa: E402
from rrs.model import DEFORM_ATTN_PYTORCH_FALLBACK, build_model  # noqa: E402


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/b0_refcoco.py")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    cfg = load_cfg(args.config, None)
    torch.manual_seed(0)
    dev = "cuda"
    report = {}

    model, info = build_model(cfg["gdino_cfg"], cfg["gdino_ckpt"], cfg["bert_path"], dev,
                              use_checkpoint=cfg["use_checkpoint"])
    report["missing_keys"] = info.missing_keys
    report["unexpected_keys"] = info.unexpected_keys
    report["deform_attn_pytorch_fallback"] = DEFORM_ATTN_PYTORCH_FALLBACK

    samples = load_refcoco_parquet(cfg["val_data"]["path"])[:args.n]
    ds = RefSegDataset(samples, cfg["val_data"]["image_root"], train=False,
                       short_side=cfg["short_side"], max_size=cfg["max_size"])
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    # ---- 2) 预训练检测能力 (掩码头未训练, 只看框) ----
    model.eval()
    ev = RefSegEvaluator()
    with torch.no_grad():
        for imgs, targets in loader:
            out = model([i.to(dev) for i in imgs], [t["caption"] for t in targets])
            ev.update(postprocess(out, targets), targets)
    report["pretrained_zero_shot"] = ev.summarize()

    # ---- 3) 前向 + 损失 + 反向 ----
    model.train()
    criterion = build_criterion(cfg)
    opt = build_optimizer(model, cfg)
    imgs, targets = next(iter(loader))
    targets = to_device(targets, dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
        out = model([i.to(dev) for i in imgs], [t["caption"] for t in targets])
    report["shapes"] = {
        "pred_logits": list(out["pred_logits"].shape),
        "pred_boxes": list(out["pred_boxes"].shape),
        "pred_masks": list(out["pred_masks"].shape),
        "pad_hw": list(out["pad_mask"].shape[-2:])
    }
    loss, ld, indices = criterion(out, targets)
    report["loss_total"] = loss.item()
    report["losses_last_layer"] = {
        k: round(v.item(), 4)
        for k, v in ld.items() if k.count("_") == 1
    }
    report["num_loss_terms"] = len(ld)
    report["matched_query"] = [int(i[0][0]) for i in indices]
    loss.backward()
    report["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    groups = {
        "pixel_decoder(new)": "pixel_decoder.",
        "mask_head(new)": "mask_head.",
        "c4_norm(new)": "c4_norm.",
        "decoder": "gd.transformer.decoder.",
        "feature_enhancer(encoder)": "gd.transformer.encoder.",
        "swin": "gd.backbone.0.",
        "input_proj": "gd.input_proj.",
        "feat_map": "gd.feat_map.",
        "bert(frozen)": "gd.bert.",
    }
    grad = {}
    for name, prefix in groups.items():
        ps = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        with_g = [p for _, p in ps if p.grad is not None and p.grad.abs().sum() > 0]
        grad[name] = {
            "params": len(ps),
            "trainable": sum(p.requires_grad for _, p in ps),
            "nonzero_grad": len(with_g)
        }
    report["grad_check"] = grad
    nonfinite = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    report["nonfinite_grads"] = nonfinite[:5]

    # ---- 4) 参数更新 ----
    w = model.mask_head.mask_embed[0].weight.detach().clone()
    wd = model.gd.transformer.decoder.layers[0].linear1.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_max_norm"])
    opt.step()
    report["param_updated"] = {
        "mask_head": (model.mask_head.mask_embed[0].weight - w).abs().max().item() > 0,
        "decoder_layer0":
        (model.gd.transformer.decoder.layers[0].linear1.weight - wd).abs().max().item() > 0
    }

    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))
    os.makedirs("work_dirs", exist_ok=True)
    with open("work_dirs/check_b0.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
