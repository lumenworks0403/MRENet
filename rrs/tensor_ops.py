"""模型分支共用的张量运算。"""
import torch
import torch.nn.functional as F


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def sigmoid_focal_loss(logits, targets, normalizer, alpha=0.25, gamma=2.0):
    """FP32 sigmoid focal loss，求和后按目标数量归一化。"""
    logits = logits.float()
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * ce * ((1 - p_t)**gamma)).sum() / normalizer
