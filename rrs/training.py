"""训练及冒烟检查共用的优化器和设备处理。"""
import torch


def to_device(targets, device):
    keys = ("boxes", "masks", "size")
    return [{k: (v.to(device) if k in keys else v) for k, v in t.items()} for t in targets]


def build_optimizer(model, cfg):
    """冻结指定参数，按 backbone 与其余参数设置学习率。

    保持原训练策略：SS2D 的 _no_weight_decay 标记不单独处理。
    """
    for n, p in model.named_parameters():
        if any(k in n for k in cfg["freeze_keywords"]):
            p.requires_grad_(False)
    bb, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if any(k in n for k in cfg["lr_backbone_names"]) else rest).append(p)
    return torch.optim.AdamW([{
        "params": rest,
        "lr": cfg["lr"]
    }, {
        "params": bb,
        "lr": cfg["lr_backbone"]
    }], weight_decay=cfg["weight_decay"])


def build_grad_scaler(enabled, **kwargs):
    """兼容 PyTorch 2.1 及新版 AMP 接口。"""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled, **kwargs)
    return torch.cuda.amp.GradScaler(enabled=enabled, **kwargs)
