"""逐组指代选择及原图概率掩码融合，对应方法公式 (12)-(16)。"""
import torch
import torch.nn.functional as F

from .criterion import build_positive_map


def postprocess_for_model(model, outputs, targets, pp=None, hed_mask_mode="avg"):
    options = dict(pp or {})
    if getattr(model, "use_hed", False):
        if "hed_all_logits" not in outputs or "hed_all_boxes" not in outputs:
            raise ValueError("SPED 模型缺少分组预测，不能回退为单层后处理")
        options["mask_mode"] = hed_mask_mode
        options["use_ref_score"] = bool(getattr(model, "use_ref_score", False))
        return postprocess_hed(outputs, targets, **options)
    return postprocess(outputs, targets, **options)


def _restore_mask(m, pad_hw, h, w, h0, w0):
    """先恢复 padding 画布并裁去 padding，再恢复原图，返回 logits。"""
    m = F.interpolate(m, size=pad_hw, mode="bilinear", align_corners=False)[..., :h, :w]
    return F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)[0]


def _token_scores(logits, tokenized):
    """TM 消融：每个候选取有效表达 token 的最大 sigmoid 概率。"""
    eligible = build_positive_map(tokenized, max_text_len=logits.shape[-1]).to(logits.device).bool()
    while eligible.ndim < logits.ndim:
        eligible = eligible.unsqueeze(-2)
    return logits.float().sigmoid().masked_fill(~eligible, 0.0).amax(-1)


@torch.no_grad()
def postprocess_hed(outputs, targets, mode="top1", score_thr=0.3, max_dets=50, mask_mode="avg",
                    use_ref_score=False):
    """SPED 的每一组各选一个候选；恢复六张 logits 后 sigmoid，再等权平均。

    queries/scores 为逐组的索引/分数；prob_mask 和 inst_masks 是最终融合结果。
    selected_boxes 保留各组选中的框；boxes 中的均值框仅用于兼容框诊断。
    mask_mode='last' 为仅使用最后一组的消融，不改变组内选取规则。
    """
    if mode != "top1":
        raise NotImplementedError("SPED/RCA 论文路径只支持单目标 top1")
    if mask_mode not in ("avg", "last"):
        raise ValueError("mask_mode 必须为 'avg' 或 'last'")
    for key in ("hed_all_logits", "hed_all_boxes", "hed_mask_embeds", "hed_pix"):
        if outputs.get(key) is None:
            raise ValueError(f"SPED 后处理缺少 {key}，需要 eval 前向的逐组输出")
    all_logits = outputs["hed_all_logits"]
    all_boxes = outputs["hed_all_boxes"]
    embeds = outputs["hed_mask_embeds"]
    pix = outputs["hed_pix"]
    if all_logits.ndim != 4 or all_boxes.shape != (*all_logits.shape[:3], 4):
        raise ValueError("分组预测应为 (L,B,Q,T) 和 (L,B,Q,4)")
    if embeds.shape[:3] != all_logits.shape[:3]:
        raise ValueError("分组掩码嵌入必须与候选组、batch 和查询维一致")
    if not torch.isfinite(all_boxes).all() or not torch.isfinite(
            embeds).all() or not torch.isfinite(pix).all():
        raise FloatingPointError("非有限的候选框或掩码特征")
    token_scores = torch.stack([_token_scores(layer, outputs["tokenized"]) for layer in all_logits])
    if use_ref_score:
        if outputs.get("hed_all_ref_logits") is None:
            raise ValueError("use_ref_score=True 但模型输出缺少 hed_all_ref_logits")
        ref_logits = outputs["hed_all_ref_logits"]
        if ref_logits.shape != all_logits.shape[:3] or not torch.isfinite(ref_logits).all():
            raise ValueError("RCA 分数必须是有限的 (L,B,Q) logits")
        score = ref_logits.float().sigmoid()
        # argmax(logits) 与 argmax(sigmoid(logits)) 数学等价；前者避免极端值饱和导致假并列。
        selections = ref_logits.float().argmax(-1)
    else:
        score = token_scores
        selections = score.argmax(-1)
    layer_ids = torch.arange(all_logits.shape[0], device=all_logits.device)
    pad_hw = outputs["pad_mask"].shape[-2:]
    results = []
    for b, target in enumerate(targets):
        keep = selections[:, b]
        h, w = target["size"].tolist()
        h0, w0 = target["orig_size"].tolist()
        selected_embeds = embeds[layer_ids, b, keep]
        layers = range(len(layer_ids)) if mask_mode == "avg" else [len(layer_ids) - 1]
        prob_sum = None
        for lid in layers:
            logit = torch.einsum("c,chw->hw", selected_embeds[lid], pix[b])[None, None].float()
            restored = _restore_mask(logit, pad_hw, h, w, h0, w0)[0]
            probability = restored.sigmoid()
            prob_sum = probability if prob_sum is None else prob_sum + probability
        probability = prob_sum / len(layers)
        selected = all_boxes[layer_ids, b, keep].float()
        cx, cy, bw, bh = selected.unbind(-1)
        xyxy = torch.stack(
            ((cx - bw / 2) * w0, (cy - bh / 2) * h0, (cx + bw / 2) * w0, (cy + bh / 2) * h0), -1)
        result = {
            "mask": probability > 0.5,
            "prob_mask": probability,
            "inst_masks": (probability > 0.5)[None],
            "queries": keep.tolist(),
            "scores": score[layer_ids, b, keep].tolist(),
            "selected_boxes": xyxy.tolist(),
            "boxes": xyxy.mean(0, keepdim=True).tolist(),
            "score_type": "ref_score" if use_ref_score else "token",
        }
        if use_ref_score:
            result["token_scores"] = token_scores[layer_ids, b, keep].tolist()
        results.append(result)
    return results


# Paper terminology for callers; legacy entry point remains importable.
postprocess_sped = postprocess_hed


@torch.no_grad()
def postprocess(outputs, targets, mode="top1", score_thr=0.3, max_dets=50):
    """每个查询的指代分数 = 其在有效描述 token 上的最大 sigmoid 概率。
    mode="top1"  : 单目标指代 (RefCOCO), 取分数最高的 1 个查询。
    mode="multi" : 多目标/无目标指代 (RefDrone), 取分数 > score_thr 的查询 (至多 max_dets 个),
                   最终掩码为这些查询掩码的并集; 若没有查询过阈值则输出空掩码 (预测为无目标)。
    返回原图尺寸的二值掩码 (并集) 以及各实例的框 / 分数。"""
    score = _token_scores(outputs["pred_logits"], outputs["tokenized"])
    pad_hw = outputs["pad_mask"].shape[-2:]
    results = []
    for b, t in enumerate(targets):
        if mode == "top1":
            keep = score[b].argmax()[None]
        else:
            s, order = score[b].sort(descending=True)
            keep = order[:max_dets][s[:max_dets] > score_thr]
        h, w = t["size"].tolist()
        h0, w0 = t["orig_size"].tolist()
        if len(keep):
            m = outputs["pred_masks"][b, keep][None].float()
            m = F.interpolate(m, size=pad_hw, mode="bilinear", align_corners=False)[..., :h, :w]
            m = F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)[0]
            inst = m.sigmoid() > 0.5
            union = inst.any(0)
        else:
            inst = torch.zeros((0, h0, w0), dtype=torch.bool, device=score.device)
            union = torch.zeros((h0, w0), dtype=torch.bool, device=score.device)
        boxes = []
        for q in keep.tolist():
            cx, cy, bw, bh = outputs["pred_boxes"][b, q].tolist()
            boxes.append([(cx - bw / 2) * w0, (cy - bh / 2) * h0, (cx + bw / 2) * w0,
                          (cy + bh / 2) * h0])
        results.append({
            "mask": union,
            "inst_masks": inst,
            "boxes": boxes,
            "scores": score[b, keep].tolist(),
            "queries": keep.tolist()
        })
    return results


def _box_iou_xyxy(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(uni, 1e-6)


class RefSegEvaluator:
    """
    掩码指标 (按所有目标的并集计算, 与 GRES/gRefCOCO 一致):
      mIoU/gIoU : 逐样本 IoU 平均; 无目标样本预测为空则 IoU=1, 否则 0
      oIoU/cIoU : 累计交集 / 累计并集
      Prec@X    : IoU >= X 的样本比例
      N-acc     : 无目标样本中预测为空的比例 (只在存在无目标样本时有意义)
    框指标:
      box_Acc@0.5 : 单目标时为 top-1 框 IoU>=0.5 的比例; 多目标时为
                    「预测框与 GT 框一一匹配 (IoU>=0.5) 且数量相等」的样本比例 (严格版)
      box_P / box_R : 多目标下 IoU>=0.5 贪心匹配的实例级 precision / recall
    """

    def __init__(self):
        self.ious, self.inter, self.union, self.box_hits = [], 0, 0, []
        self.empty_total = self.empty_hit = 0
        self.tp = self.npred = self.ngt = 0

    def update(self, results, targets):
        for r, t in zip(results, targets):
            pred = r["mask"].cpu()
            gt = t["orig_mask"].bool()
            i = (pred & gt).sum().item()
            u = (pred | gt).sum().item()
            self.inter += i
            self.union += u
            self.ious.append(i / u if u > 0 else 1.0)
            if gt.sum() == 0:
                self.empty_total += 1
                self.empty_hit += int(pred.sum() == 0)

            gts = [[x, y, x + w, y + h] for x, y, w, h in t["orig_boxes"].tolist()]
            preds = r["boxes"]
            used, tp = set(), 0
            for p in preds:
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if j not in used:
                        iou = _box_iou_xyxy(p, g)
                        if iou > best:
                            best, bj = iou, j
                if best >= 0.5:
                    used.add(bj)
                    tp += 1
            self.tp += tp
            self.npred += len(preds)
            self.ngt += len(gts)
            self.box_hits.append(float(tp == len(gts) == len(preds)))

    def summarize(self):
        n = max(len(self.ious), 1)
        ious = torch.tensor(self.ious) if self.ious else torch.zeros(1)
        res = {
            "num": len(self.ious),
            "mIoU": ious.mean().item() * 100,
            "oIoU": self.inter / max(self.union, 1) * 100,
            "box_Acc@0.5": sum(self.box_hits) / max(len(self.box_hits), 1) * 100,
            "box_P": self.tp / max(self.npred, 1) * 100,
            "box_R": self.tp / max(self.ngt, 1) * 100,
        }
        for th in (0.5, 0.6, 0.7, 0.8, 0.9):
            res[f"Prec@{th}"] = (ious >= th).float().sum().item() / n * 100
        if self.empty_total:
            res["N-acc"] = self.empty_hit / self.empty_total * 100
        return res
