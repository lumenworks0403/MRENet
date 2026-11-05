"""推理后处理与指代分割评估指标。

B0_HED 新增:
  postprocess_hed: HED 推理后处理。
    - 分类: 6层各自 sigmoid 后对同一查询的概率求均值（不能均值 logits 再 sigmoid）
    - 框: 6层预测框直接均值
    - 掩码: 对选定查询逐层计算、累加，再除以层数得均值概率，最后阈值化
    NOTE: "掩码概率平均"是本项目的分割适配，FT-FSOD 原文无此设计，此处明确注明。
  2026-10-16: use_ref_score=True 时 top1 查询改由新增 RefScoreHead 的六层 sigmoid 均值决定
    (原 token 均分降为诊断值 token_scores)。掩码生成、空间恢复与六层概率平均的规则不变。
"""
import torch
import torch.nn.functional as F

from .criterion import build_positive_map


def postprocess_for_model(model, outputs, targets, pp=None, hed_mask_mode="avg"):
    """验证和单图预测共用同一个模式入口，旧模型默认为 B0。

    评分模式取自 **模型实例** 的 use_ref_score (不靠输出里有没有 pred_ref_logits 猜),
    这样 CLI 配置与实际模型不一致时会报错而不是静默换一套选择规则。"""
    options = dict(pp or {})
    if getattr(model, "use_hed", False):
        if "hed_all_logits" not in outputs or "hed_all_boxes" not in outputs:
            raise ValueError("HED 模型缺少六层预测，不能回退为单层后处理")
        options["mask_mode"] = hed_mask_mode
        options["use_ref_score"] = bool(getattr(model, "use_ref_score", False))
        return postprocess_hed(outputs, targets, **options)
    return postprocess(outputs, targets, **options)


def _restore_mask(m, pad_hw, h, w, h0, w0):
    """与原 B0 postprocess 相同的空间恢复: 恢复 padding 画布 → 裁掉 padding → 恢复原图。
    m: (1, K, H/4, W/4) logits, 返回 (K, h0, w0) logits。"""
    m = F.interpolate(m, size=pad_hw, mode="bilinear", align_corners=False)[..., :h, :w]
    return F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)[0]


@torch.no_grad()
def postprocess_hed(outputs, targets, mode="top1", score_thr=0.3, max_dets=50, mask_mode="avg",
                    use_ref_score=False):
    """HED 推理后处理。

    use_ref_score=False (默认): 查询分数 = 六层 token 分类概率均值在描述正 token 上的均分
              (原路径, 与 2026-10-13 版逐元素相同)。
    use_ref_score=True: 查询分数 = 六层 pred_ref_logits 分别 sigmoid 后求均值 (新评分头,
              见 model.py / ref_score_head.py)。选中的 q* 完全由新分数决定, 旧 token 均分
              仍然计算并作为 "token_scores" 随结果返回, 只作诊断, 不参与选择, 也不做加权融合。
              缺少 hed_all_ref_logits 时报错, 不静默退回旧分数。

    分类分数 (旧路径): 6层各自 sigmoid 后对同一查询的概率求均值（先 sigmoid 再均值，
              与官方 predict_by_feat 中 all_layers_cls_scores.sigmoid().mean(0) 一致）。
    框: 6层预测框均值（官方同策略）。
    掩码 (mask_mode="avg", 默认): 按汇总分数选出查询后, 只对这些查询逐层计算掩码,
          每层恢复到原图后 sigmoid, 六层概率平均, 再按 0.5 阈值化。
          NOTE: "掩码概率平均"是本项目分割适配，FT-FSOD 原文无此设计。
    掩码 (mask_mode="last"): 只用最后一层掩码, 用于对照。
    空间恢复顺序: 恢复 padding 画布 → 裁掉 padding → 恢复原图（与原 B0 一致）。
    选择查询只依据模型预测分数, 不使用真实标注。

    输出字典需含 hed_all_logits / hed_all_boxes, mask_mode="avg" 时还需
    hed_mask_embeds / hed_pix (仅 eval 模式前向会给出)。
    若无 hed_all_logits（use_hed=False 的模型），回退到原 postprocess。
    """
    if "hed_all_logits" not in outputs:
        return postprocess(outputs, targets, mode=mode, score_thr=score_thr, max_dets=max_dets)
    assert mask_mode in ("avg", "last"), mask_mode
    # 本阶段 HED 后处理只支持单目标 top1: multi 会把不同查询的掩码取并集,
    # 超出当前约束。需要 multi 时应先单独设计并审查, 不能静默合并。
    if mode != "top1":
        raise NotImplementedError(f"HED-structure/no-DN 后处理目前只支持 mode='top1' (当前 {mode!r})")
    if mask_mode == "avg" and outputs.get("hed_mask_embeds") is None:
        raise ValueError("mask_mode='avg' 需要 eval 模式前向输出的 hed_mask_embeds")

    all_logits = outputs["hed_all_logits"]  # (6, B, Q, T)
    all_boxes = outputs["hed_all_boxes"]  # (6, B, Q, 4)

    # 各层分别 sigmoid，再对同一查询平均概率。
    # nan_to_num(0.0) 与原 B0 postprocess 一致, 在逐层 sigmoid 之后、平均之前做;
    # 有限输入时结果不变。
    probs_avg = all_logits.sigmoid().nan_to_num(0.0).mean(0)  # (B, Q, T)
    boxes_avg = all_boxes.mean(0)  # (B, Q, 4)

    pos_map = build_positive_map(outputs["tokenized"]).to(probs_avg.device)
    pm = pos_map / pos_map.sum(-1, keepdim=True).clamp(min=1)
    token_score = torch.einsum("bqt,bt->bq", probs_avg.float(), pm)  # (B, Q) 旧 token 均分

    if use_ref_score:
        if outputs.get("hed_all_ref_logits") is None:
            raise ValueError("use_ref_score=True 但模型输出缺少 hed_all_ref_logits")
        # 沿用 HED 的"同一查询索引、各层先 sigmoid 再平均"约定 (继承的设计假设:
        # 六层同一索引视作同一候选; 本次不新增跨分支候选聚类)。
        score = outputs["hed_all_ref_logits"].float().sigmoid().mean(0)  # (B, Q)
    else:
        score = token_score

    pad_hw = outputs["pad_mask"].shape[-2:]
    results = []
    for b, t in enumerate(targets):
        keep = score[b].argmax()[None]  # top1 (multi 已在入口拒绝)

        h, w = t["size"].tolist()
        h0, w0 = t["orig_size"].tolist()

        if len(keep):
            if mask_mode == "avg":
                # 只对选中的查询逐层计算掩码: 每层先做与 B0 相同的空间恢复
                # (padding 画布 → 裁 padding → 原图), 再 sigmoid 累加, 最后除以层数。
                embeds = outputs["hed_mask_embeds"][:, b, keep]  # (6, K, C)
                pix = outputs["hed_pix"][b]  # (C, H/4, W/4)
                prob_sum = None
                for lid in range(embeds.shape[0]):
                    m = torch.einsum("kc,chw->khw", embeds[lid], pix)[None].float()
                    m = _restore_mask(m, pad_hw, h, w, h0, w0)
                    p = m.sigmoid()
                    prob_sum = p if prob_sum is None else prob_sum + p
                prob = prob_sum / embeds.shape[0]
            else:
                # 对照选项: 只用最后一层掩码 (与原 B0 取法相同)
                m = outputs["pred_masks"][b, keep][None].float()
                prob = _restore_mask(m, pad_hw, h, w, h0, w0).sigmoid()
            inst = prob > 0.5  # (1, h0, w0)
            union = inst[0]  # top1: K=1, 不做跨查询并集
        else:
            inst = torch.zeros((0, h0, w0), dtype=torch.bool, device=score.device)
            union = torch.zeros((h0, w0), dtype=torch.bool, device=score.device)

        boxes = []
        for q in keep.tolist():
            cx, cy, bw, bh = boxes_avg[b, q].tolist()
            boxes.append([(cx - bw / 2) * w0, (cy - bh / 2) * h0, (cx + bw / 2) * w0,
                          (cy + bh / 2) * h0])
        # scores 始终是"实际用于选 q* 的分数"; 新模式下额外给出旧 token 均分作诊断,
        # score_type 标明含义, 避免对外展示的分数与选择依据不一致。
        r = {
            "mask": union,
            "inst_masks": inst,
            "boxes": boxes,
            "scores": score[b, keep].tolist(),
            "queries": keep.tolist(),
            "score_type": "ref_score" if use_ref_score else "token"
        }
        if use_ref_score:
            r["token_scores"] = token_score[b, keep].tolist()
        results.append(r)
    return results


@torch.no_grad()
def postprocess(outputs, targets, mode="top1", score_thr=0.3, max_dets=50):
    """每个查询的指代分数 = 其在描述正 token 上的平均 sigmoid 概率。
    mode="top1"  : 单目标指代 (RefCOCO), 取分数最高的 1 个查询。
    mode="multi" : 多目标/无目标指代 (RefDrone), 取分数 > score_thr 的查询 (至多 max_dets 个),
                   最终掩码为这些查询掩码的并集; 若没有查询过阈值则输出空掩码 (预测为无目标)。
    返回原图尺寸的二值掩码 (并集) 以及各实例的框 / 分数。"""
    pos_map = build_positive_map(outputs["tokenized"]).to(outputs["pred_logits"].device)
    prob = outputs["pred_logits"].sigmoid().nan_to_num(0.0)  # (B, Q, 256)
    pm = pos_map / pos_map.sum(-1, keepdim=True).clamp(min=1)
    score = torch.einsum("bqt,bt->bq", prob.float(), pm)
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
