"""
HED+DN 的去噪查询生成（无 groundingdino / mmdet 依赖，便于单元测试直接导入）。

逐函数迁移自官方 FT-FSOD 根项目（非 NTIRE 子项目）:
  FT-FSOD-main/mmdet/models/layers/transformer/dino_layers.py::CdnQueryGenerator
  commit c68cea00fac282ec526d59bca55f27b3203daf37
  本地快照 tmp/hed_reference/dino_layers.py
  SHA256 514579e808a27ea478d6bbbc7abd2276486707ea7b40d037f759255b5c56b252
默认值出处:
  FT-FSOD-main/configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py 的 dn_cfg
    (label_noise_scale=0.5, box_noise_scale=1.0, dynamic, num_dn_queries=100)
  FT-FSOD-main/configs_cdfsod/final_configs/grounding_dino_swin-b_finetune_ArTaxOr_10shot.py
    (rand_dnquery_rate=0.5; bbox_head.num_classes=数据集类别数 -> DINO.__init__ 把它
     传给 dn_cfg['num_classes'], 即 label_embedding 的行数)

与官方的接口差异 (本项目适配, 逐条在 docs/HED_DN接入说明_20261013.md 说明):
  1. 输入: 官方读 DetDataSample.gt_instances (像素 xyxy + 类别 labels);
     这里读 B0 targets 的 boxes (归一化 cxcywh), 先转成归一化 xyxy 再走官方加噪公式。
  2. 类别: B0 是文本指代, 没有类别编号。按"一条描述 = 一个类别"处理, 所有 GT 的
     label 都是 0, label_book_size 默认 1 (对应官方微调配置 num_classes=数据集类别数)。
     此时 25% 的类别替换总是换回 0, 类别噪声实际不起作用 —— 这是如实迁移的结果,
     不是遗漏。分类监督不看噪声类别, 只用该描述的 positive token map (见 criterion.py)。
  3. 整个 batch 都没有目标时返回 None (不生成 DN)。官方此时生成 0 个 DN 查询,
     效果相同, 这里直接跳过以免构造空张量。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from .tensor_ops import inverse_sigmoid


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def box_xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)


@dataclass
class DNQueries:
    """一次 CdnQueryGenerator 调用的全部产物。每个重新生成 DN 的分支各有一份,
    不与其他分支共享索引。"""
    label_query: torch.Tensor  # (B, Ndn, C) 内容查询 (label_embedding)
    bbox_unsig: torch.Tensor  # (B, Ndn, 4) inverse_sigmoid(加噪后的归一化 cxcywh)
    attn_mask: torch.Tensor  # (Ndn+Q, Ndn+Q) bool, True = 看不见 (nn.MultiheadAttention 约定)
    meta: dict  # num_denoising_queries / num_denoising_groups / max_num_target / num_target_list

    @property
    def num_queries(self):
        return self.meta["num_denoising_queries"]


def dn_target_indices(meta, b, device=None):
    """第 b 张图在 DN 序列中的正/负样本位置, 与官方 _get_dn_targets_single 相同。

    每组长度 2*M (M = max_num_target): 前 M 个位置是正样本区, 后 M 个是负样本区,
    第 b 张图只占每区前 n_b 个, 其余为 padding。
    返回 (pos_inds, pos_gt_inds, neg_inds), 均为 long, 长度 n_b * G。
    """
    n = meta["num_target_list"][b]
    G = meta["num_denoising_groups"]
    M = meta["max_num_target"]
    t = torch.arange(n, dtype=torch.long, device=device).unsqueeze(0).repeat(G, 1)  # (G, n)
    pos = (torch.arange(G, dtype=torch.long, device=device).unsqueeze(1) * 2 * M + t).flatten()
    return pos, t.flatten(), pos + M


class CdnQueryGenerator(nn.Module):
    """官方 CdnQueryGenerator 的 B0 版本。只在 训练 且 use_dn=True 时调用。"""

    def __init__(self, embed_dims, num_matching_queries, label_book_size=1, label_noise_scale=0.5,
                 box_noise_scale=1.0, num_dn_queries=100):
        super().__init__()
        assert label_noise_scale > 0, "官方 generate_dn_label_query 要求 label_noise_scale > 0"
        assert box_noise_scale > 0, "官方 generate_dn_bbox_query 要求 box_noise_scale > 0"
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_book_size = label_book_size
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.num_dn_queries = num_dn_queries
        # 与官方相同: nn.Embedding 默认 N(0,1) 初始化, DINO.init_weights 不再覆盖它。
        self.label_embedding = nn.Embedding(label_book_size, embed_dims)

    def get_num_groups(self, max_num_target):
        """官方 dynamic dn groups: 组数 = num_dn_queries // max_num_target, 至少 1。"""
        if max_num_target == 0:
            return 1
        return max(int(self.num_dn_queries // max_num_target), 1)

    def forward(self, gt_boxes_list, gt_labels_list=None):
        """gt_boxes_list: list[(n_b, 4)] 归一化 cxcywh (B0 targets["boxes"])。
        gt_labels_list: 可选 list[(n_b,)] long; 缺省时全部为 0 (方案 A)。
        返回 DNQueries; 整个 batch 无目标时返回 None。"""
        device = self.label_embedding.weight.device
        num_target_list = [int(len(b)) for b in gt_boxes_list]
        max_num_target = max(num_target_list) if num_target_list else 0
        if max_num_target == 0:
            return None
        num_groups = self.get_num_groups(max_num_target)

        if gt_labels_list is None:
            gt_labels_list = [
                torch.zeros(n, dtype=torch.long, device=device) for n in num_target_list
            ]
        gt_labels = torch.cat([l.to(device).long() for l in gt_labels_list])
        # 官方输入是像素 xyxy / img_shape = 归一化 xyxy; B0 的 cxcywh 同样以原图宽高归一化
        gt_bboxes = box_cxcywh_to_xyxy(torch.cat([b.to(device).float() for b in gt_boxes_list]))

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        batch_idx = torch.cat([
            torch.full((n, ), i, dtype=torch.long, device=device)
            for i, n in enumerate(num_target_list)
        ])
        dn_label_query, dn_bbox_query = self.collate_dn_queries(dn_label_query,
                                                                dn_bbox_query, batch_idx,
                                                                len(gt_boxes_list), num_groups)
        attn_mask = self.generate_dn_mask(max_num_target, num_groups, device)
        meta = dict(num_denoising_queries=int(max_num_target * 2 * num_groups),
                    num_denoising_groups=num_groups, max_num_target=max_num_target,
                    num_target_list=num_target_list)
        return DNQueries(dn_label_query, dn_bbox_query, attn_mask, meta)

    def generate_dn_label_query(self, gt_labels, num_groups):
        """官方: 随机选 label_noise_scale*0.5 比例的位置, 替换为 [0, num_classes) 的随机类别。"""
        gt_labels_expand = gt_labels.repeat(2 * num_groups, 1).view(-1)
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(-1)
        new_labels = torch.randint_like(chosen_indice, 0, self.label_book_size)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice, new_labels)
        return self.label_embedding(noisy_labels_expand)

    def generate_dn_bbox_query(self, gt_bboxes, num_groups):
        """官方: 在归一化 xyxy 上加噪。
        正样本: 偏移 rand(-1,1) * wh/2 * box_noise_scale;
        负样本: 偏移 ±[1,2) * wh/2 * box_noise_scale。
        clamp 到 [0,1] -> cxcywh -> inverse_sigmoid(eps=1e-3)。"""
        device = gt_bboxes.device
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        positive_idx = torch.arange(len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(num_groups, dtype=torch.long,
                                                          device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        rand_sign = torch.randint_like(gt_bboxes_expand, low=0, high=2,
                                       dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(gt_bboxes_expand)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign

        bboxes_whwh = box_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(rand_part,
                                                           bboxes_whwh) * self.box_noise_scale / 2
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = box_xyxy_to_cxcywh(noisy_bboxes_expand)
        return inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)

    def collate_dn_queries(self, input_label_query, input_bbox_query, batch_idx, batch_size,
                           num_groups):
        """官方 collate: 第 i 段 (i=0..2G-1) 放在偏移 i*M 处, 不足 M 的位置补 0。
        补 0 的 bbox_unsig 经 sigmoid 后是 (0.5,0.5,0.5,0.5), 与官方相同。"""
        device = input_label_query.device
        num_target_list = [int(torch.sum(batch_idx == idx).item()) for idx in range(batch_size)]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat([torch.arange(n, device=device) for n in num_target_list])
        map_query_index = torch.cat(
            [map_query_index + max_num_target * i for i in range(2 * num_groups)]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(batch_size, num_denoising_queries, self.embed_dims,
                                          device=device, dtype=input_label_query.dtype)
        batched_bbox_query = torch.zeros(batch_size, num_denoising_queries, 4, device=device,
                                         dtype=input_bbox_query.dtype)
        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(self, max_num_target, num_groups, device):
        """官方 mask (True = 看不见):
          - 普通查询看不见任何 DN 查询 (防止 GT 信息泄漏);
          - DN 组之间互相看不见;
          - DN 查询可以看见普通查询; 普通查询之间互相可见。"""
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(num_queries_total, num_queries_total, device=device,
                                dtype=torch.bool)
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        for i in range(num_groups):
            row_scope = slice(max_num_target * 2 * i, max_num_target * 2 * (i + 1))
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1), num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask
