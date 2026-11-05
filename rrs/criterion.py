"""
查询匹配与损失。

参考 Open-GroundingDino (commit d248268):
  - models/GroundingDINO/matcher.py::HungarianMatcher           -> token 级 focal 分类代价 + L1 + GIoU
  - models/GroundingDINO/groundingdino.py::SetCriterion
        token_sigmoid_binary_focal_loss / loss_boxes / aux_outputs / interm_outputs 的处理方式
新增 (参考 Mask2Former 的 dice/BCE 掩码损失写法):
  - 匹配代价中加入掩码 BCE + Dice 代价 (点采样)
  - 掩码损失 loss_mask (BCE) + loss_dice

指代设定: 每个样本对应一条表达，可包含零个、一个或多个目标； 其正 token 为整句 caption 中除特殊符号外的所有 token。
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .model import _GD_ROOT  # noqa: F401  (确保 groundingdino 在 sys.path 中)
from groundingdino.util import box_ops
from .dn import dn_target_indices
from .tensor_ops import sigmoid_focal_loss


def build_positive_map(tokenized, max_text_len=256):
    """(B, max_text_len): caption 中所有非特殊 token 置 1 ([CLS] [SEP] '.' 以及 padding 置 0)。
    对应 GroundingDINO 把整句短语映射到 token span 的做法。"""
    ids = tokenized["input_ids"]
    att = tokenized["attention_mask"].bool()
    pos = att.clone()
    pos &= ids != 101  # [CLS]
    pos &= ids != 102  # [SEP]
    pos &= ids != 1012  # '.'
    out = torch.zeros(ids.shape[0], max_text_len, device=ids.device)
    out[:, :ids.shape[1]] = pos.float()
    return out


def _dice_cost(inputs, targets):
    inputs = inputs.sigmoid()
    num = 2 * inputs @ targets.T
    den = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (num + 1) / (den + 1)


def _bce_cost(inputs, targets):
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    return (pos @ targets.T + neg @ (1 - targets).T) / hw


def _point_sample(x, coords):
    """x: (N, 1, H, W); coords: (N or 1, P, 2) in [0,1] -> (N, P)"""
    grid = 2 * coords[:, :, None, :] - 1
    if grid.shape[0] != x.shape[0]:
        grid = grid.expand(x.shape[0], -1, -1, -1)
    return F.grid_sample(x, grid, align_corners=False)[:, 0, :, 0]


def _pad_target_mask(tgt_mask, img_hw, pad_hw):
    """把 (n, h, w) 的 GT 掩码放到 batch padding 后的画布上 (预测掩码覆盖整个 padded 画布, 点采样坐标需一致)。"""
    canvas = tgt_mask.new_zeros((tgt_mask.shape[0], ) + tuple(pad_hw))
    canvas[:, :img_hw[0], :img_hw[1]] = tgt_mask
    return canvas


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, cost_mask=5.0, cost_dice=5.0,
                 focal_alpha=0.25, num_points=12544):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        self.cost_mask, self.cost_dice = cost_mask, cost_dice
        self.focal_alpha = focal_alpha
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets, pos_map, pad_hw):
        indices = []
        bs = outputs["pred_boxes"].shape[0]
        alpha, gamma = self.focal_alpha, 2.0
        for b in range(bs):
            if len(targets[b]["boxes"]) == 0:  # 无目标表达: 所有查询都是负样本
                indices.append((torch.zeros(0,
                                            dtype=torch.int64), torch.zeros(0, dtype=torch.int64)))
                continue
            # Compute focal costs in FP32: 1e-8 underflows in FP16 and
            # sigmoid saturation otherwise produces infinite matching costs.
            prob = outputs["pred_logits"][b].float().sigmoid()  # (Q, 256)
            neg = (1 - alpha) * (prob**gamma) * (-(1 - prob + 1e-8).log())
            pos = alpha * ((1 - prob)**gamma) * (-(prob + 1e-8).log())
            pm = pos_map[b:b + 1] / pos_map[b:b + 1].sum(-1, keepdim=True).clamp(min=1)
            c_cls = (pos - neg) @ pm.T  # (Q, 1); 同一表达的所有目标共享同一组正 token, 广播到 (Q, n)

            out_bbox = outputs["pred_boxes"][b].float()
            tgt_bbox = targets[b]["boxes"].float()
            c_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            c_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                  box_ops.box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_class * c_cls + self.cost_bbox * c_bbox + self.cost_giou * c_giou

            if outputs.get("pred_masks") is not None and (self.cost_mask > 0 or self.cost_dice > 0):
                pm_logits = outputs["pred_masks"][b][:, None]  # (Q, 1, h, w)
                tm = _pad_target_mask(targets[b]["masks"], targets[b]["size"].tolist(),
                                      pad_hw)[:, None]
                coords = torch.rand(1, self.num_points, 2, device=pm_logits.device)
                p = _point_sample(pm_logits.float(), coords)
                t = _point_sample(tm.float(), coords)
                C = C + self.cost_mask * _bce_cost(p, t) + self.cost_dice * _dice_cost(p, t)

            C = C.float().cpu()
            if not torch.isfinite(C).all():
                raise FloatingPointError(
                    f"Non-finite matching cost in batch item {b}; stopped instead of assigning zero cost."
                )
            i, j = linear_sum_assignment(C.numpy())
            indices.append(
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))
        return indices


class SetCriterion(nn.Module):

    def __init__(self, matcher, weight_dict, focal_alpha=0.25, focal_gamma=2.0, num_points=12544,
                 oversample_ratio=3.0, importance_sample_ratio=0.75):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_boxes, pos_map, text_mask):
        logits = outputs["pred_logits"].float()  # FP32 focal loss and reduction
        tgt = torch.zeros_like(logits)
        for b, (src, _) in enumerate(indices):
            tgt[b, src] = pos_map[b]
        valid = text_mask[:, None, :].expand_as(logits)
        logits, tgt = logits[valid], tgt[valid]
        return {
            "loss_ce": sigmoid_focal_loss(logits, tgt, num_boxes, self.focal_alpha,
                                          self.focal_gamma)
        }

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        if sum(len(s) for s, _ in indices) == 0:
            z = (outputs["pred_boxes"].float().nan_to_num(0.0, 0.0, 0.0) *
                 0).sum()  # 保持计算图; 先乘 0 再求和, 避免 fp16 sum 溢出成 inf*0=NaN
            return {"loss_bbox": z, "loss_giou": z}
        b_idx = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        s_idx = torch.cat([s for s, _ in indices])
        src = outputs["pred_boxes"][b_idx, s_idx]
        tgt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
        l1 = F.l1_loss(src, tgt, reduction="none").sum() / num_boxes
        giou = (1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src),
                                        box_ops.box_cxcywh_to_xyxy(tgt)))).sum() / num_boxes
        return {"loss_bbox": l1, "loss_giou": giou}

    def loss_masks(self, outputs, targets, indices, num_boxes, pad_hw):
        """Mask2Former 式不确定性点采样 BCE + Dice。"""
        if sum(len(s) for s, _ in indices) == 0:
            z = (outputs["pred_masks"].float().nan_to_num(0.0, 0.0, 0.0) * 0).sum()
            return {"loss_mask": z, "loss_dice": z}
        src_list, tgt_list = [], []
        for b, (s, j) in enumerate(indices):
            src_list.append(outputs["pred_masks"][b, s])
            tm = _pad_target_mask(targets[b]["masks"], targets[b]["size"].tolist(), pad_hw)
            tgt_list.append(tm[j])
        src = torch.cat(src_list)[:, None].float()  # (N, 1, h, w)
        tgt = torch.cat(tgt_list)[:, None].float()  # (N, 1, H, W)
        with torch.no_grad():
            n = src.shape[0]
            k = int(self.num_points * self.oversample_ratio)
            coords = torch.rand(n, k, 2, device=src.device)
            unc = -_point_sample(src, coords).abs()
            n_imp = int(self.importance_sample_ratio * self.num_points)
            idx = torch.topk(unc, n_imp, dim=1)[1]
            coords = torch.gather(coords, 1, idx[..., None].expand(-1, -1, 2))
            coords = torch.cat(
                [coords, torch.rand(n, self.num_points - n_imp, 2, device=src.device)], dim=1)
            t = _point_sample(tgt, coords)
        p = _point_sample(src, coords)
        bce = F.binary_cross_entropy_with_logits(p, t, reduction="none").mean(1).sum() / num_boxes
        ps = p.sigmoid()
        dice = (1 - (2 * (ps * t).sum(1) + 1) / (ps.sum(1) + t.sum(1) + 1)).sum() / num_boxes
        return {"loss_mask": bce, "loss_dice": dice}

    def loss_ref_score(self, ref_logits, indices, num_boxes):
        """新增指代评分头的损失 (本项目新增, 非官方迁移)。

        标签直接复用调用方那一层自己的 HungarianMatcher 结果: 匹配到 GT 的查询为 1,
        其余为 0 —— 不重新调用匹配器 (匹配器带随机点采样, 再调用会得到第二套不一致的匹配),
        也不用模型自己的最高分当标签。
        损失: 二元 sigmoid focal loss (alpha/gamma 与 loss_labels 相同), FP32,
        逐元素求和后除以 max(batch GT 数, 1) —— 与其余损失同一归一化。
        全 batch 无目标时标签全 0, 损失仍有定义且可反传。
        """
        logits = ref_logits.float()  # (B, Q)
        tgt = torch.zeros_like(logits)
        for b, (src, _) in enumerate(indices):
            tgt[b, src.to(logits.device)] = 1.0
        return {
            "loss_ref_score":
            sigmoid_focal_loss(logits, tgt, num_boxes, self.focal_alpha, self.focal_gamma)
        }

    def _single(self, outputs, targets, pos_map, text_mask, pad_hw, num_boxes, with_mask,
                with_ref_score=False):
        outs = {k: v for k, v in outputs.items() if k in ("pred_logits", "pred_boxes")}
        if with_mask:
            outs["pred_masks"] = outputs["pred_masks"]
        indices = self.matcher(outs, targets, pos_map, pad_hw)
        losses = {}
        losses.update(self.loss_labels(outs, targets, indices, num_boxes, pos_map, text_mask))
        losses.update(self.loss_boxes(outs, targets, indices, num_boxes))
        if with_mask:
            losses.update(self.loss_masks(outs, targets, indices, num_boxes, pad_hw))
        if with_ref_score:
            # 评分开关开启时, 普通查询分支必须带 pred_ref_logits; 缺失即报错, 不静默跳过。
            if outputs.get("pred_ref_logits") is None:
                raise ValueError("use_ref_score=True 但该层输出缺少 pred_ref_logits")
            losses.update(self.loss_ref_score(outputs["pred_ref_logits"], indices, num_boxes))
        return losses, indices

    def forward(self, outputs, targets):
        pos_map = build_positive_map(outputs["tokenized"]).to(outputs["pred_logits"].device)
        text_mask = torch.zeros_like(pos_map, dtype=torch.bool)
        tm = outputs["text_token_mask"]
        text_mask[:, :tm.shape[1]] = tm
        pad_hw = tuple(outputs["pad_mask"].shape[-2:])
        num_boxes = max(float(sum(len(t["boxes"]) for t in targets)), 1.0)

        # 评分损失按模型实际是否输出评分 logits 决定 (model.py 只在 use_ref_score=True 时写该键)。
        # interm_outputs (encoder 中间层) 与 dn_outputs 明确不加评分损失: 它们不是普通查询分支。
        with_ref = outputs.get("pred_ref_logits") is not None
        losses, indices = self._single(outputs, targets, pos_map, text_mask, pad_hw, num_boxes,
                                       True, with_ref_score=with_ref)
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            l, _ = self._single(aux, targets, pos_map, text_mask, pad_hw, num_boxes,
                                aux.get("pred_masks") is not None, with_ref_score=with_ref)
            losses.update({f"{k}_{i}": v for k, v in l.items()})
        if "interm_outputs" in outputs:
            l, _ = self._single(outputs["interm_outputs"], targets, pos_map, text_mask, pad_hw,
                                num_boxes, False)
            losses.update({f"{k}_interm": v for k, v in l.items()})
        if outputs.get("dn_outputs") is not None:
            # DN 不参与 Hungarian 匹配, 也不改变上面普通查询损失的 num_boxes。
            dn = outputs["dn_outputs"]
            n_layers = dn["pred_logits"].shape[0]
            for i in range(n_layers):
                l = self.loss_dn_single(dn["pred_logits"][i], dn["pred_boxes"][i], targets,
                                        dn["layer_meta"][i], pos_map, text_mask)
                # 与官方 DINOHead.loss_by_feat 命名对应: 最后一层无后缀, 其余层带层号
                sfx = "" if i == n_layers - 1 else f"_{i}"
                losses.update({f"{k}{sfx}": v for k, v in l.items()})
        total = sum(v * self.weight_dict[re.sub(r"_(\d+|interm)$", "", k)]
                    for k, v in losses.items())
        return total, losses, indices

    def loss_dn_single(self, logits, boxes, targets, meta, pos_map, text_mask):
        """一层 DN 损失, 迁移自官方 grounding_dino_head_HED.py::_loss_dn_single /
        _get_dn_targets_single (同 dino_head.py 的定位规则)。

        监督直接按该层自己的 metadata 定位, 不做匹配:
          正样本 DN (每组前 M 个中的前 n_b 个): 分类目标 = 该描述的 positive token map
              (方案 A: 一条描述的所有目标共用同一组正 token), 另算 L1 + GIoU 对应 GT 框。
          负样本 DN 与 padding 位: 分类目标全 0 (label_weights 官方全为 1, padding 也计入),
              不算框损失。
        归一化: 分类与框损失都除以 batch 内正样本 DN 总数 (至少 1)。官方 bg_cls_weight=0,
              所以 cls_avg_factor 也等于正样本数。单卡训练, 不做 reduce_mean。
        与官方的差异: GIoU 在归一化坐标上算 (与 B0 普通查询 loss_boxes 一致),
              官方在像素坐标上算; 宽高比不为 1 时两者数值略有不同。
        """
        logits = logits.float()
        boxes = boxes.float()
        B, ndn, _ = logits.shape
        assert ndn == meta["num_denoising_queries"], "DN 预测数与该层 metadata 不一致"
        tgt = torch.zeros_like(logits)
        src_boxes, tgt_boxes = [], []
        num_pos = 0
        for b in range(B):
            pos, gt_idx, _neg = dn_target_indices(meta, b, device=logits.device)
            if len(pos) == 0:  # 无目标描述: 该图所有 DN 位都是全 0 分类目标
                continue
            tgt[b, pos] = pos_map[b]
            src_boxes.append(boxes[b, pos])
            tgt_boxes.append(targets[b]["boxes"].float()[gt_idx])
            num_pos += len(pos)
        avg = max(float(num_pos), 1.0)

        valid = text_mask[:, None, :].expand_as(logits)
        lg, tg = logits[valid], tgt[valid]
        loss_ce = sigmoid_focal_loss(lg, tg, avg, self.focal_alpha, self.focal_gamma)

        if num_pos == 0:
            z = (boxes.nan_to_num(0.0, 0.0, 0.0) * 0).sum()
            return {"loss_ce_dn": loss_ce, "loss_bbox_dn": z, "loss_giou_dn": z}
        src, tb = torch.cat(src_boxes), torch.cat(tgt_boxes)
        l1 = F.l1_loss(src, tb, reduction="none").sum() / avg
        giou = (1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src),
                                        box_ops.box_cxcywh_to_xyxy(tb)))).sum() / avg
        return {"loss_ce_dn": loss_ce, "loss_bbox_dn": l1, "loss_giou_dn": giou}


def build_criterion(cfg):
    matcher = HungarianMatcher(cfg["set_cost_class"], cfg["set_cost_bbox"], cfg["set_cost_giou"],
                               cfg["set_cost_mask"], cfg["set_cost_dice"],
                               num_points=cfg["num_points"])
    weight_dict = {
        "loss_ce": cfg["cls_loss_coef"],
        "loss_bbox": cfg["bbox_loss_coef"],
        "loss_giou": cfg["giou_loss_coef"],
        "loss_mask": cfg["mask_loss_coef"],
        "loss_dice": cfg["dice_loss_coef"]
    }
    # DN 损失与普通查询共用同一组 cls/bbox/giou 系数 (官方 DINOHead 的 DN 损失复用
    # self.loss_cls / loss_bbox / loss_iou, 权重相同), 再乘 dn.loss_coef (论文 λ_dn;
    # 官方代码中无此额外系数, 等价于 1.0)。没有 DN 掩码损失。
    lam = float(cfg.get("dn", {}).get("loss_coef", 1.0))
    weight_dict.update({
        "loss_ce_dn": cfg["cls_loss_coef"] * lam,
        "loss_bbox_dn": cfg["bbox_loss_coef"] * lam,
        "loss_giou_dn": cfg["giou_loss_coef"] * lam
    })
    # 新增评分头损失的独立权重 (未调参, 起步 1.0)。六层逐项加权求和, 与其余辅助损失同样处理;
    # 这里始终注册, 评分关闭时根本不会产生 loss_ref_score* 这些键, 总损失不变。
    weight_dict["loss_ref_score"] = float(cfg.get("ref_score_loss_coef", 1.0))
    return SetCriterion(matcher, weight_dict, num_points=cfg["num_points"])
