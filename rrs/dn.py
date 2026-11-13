"""训练专用目标重建查询，参照第 3.4 节公式 (9)-(11)。
默认单标签表、组预算 100、正负框扰动、组间注意力隔离。"""
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
    """批量 DN 内容、inverse-sigmoid 参考框、注意力 mask 和监督布局。"""
    label_query: torch.Tensor
    bbox_unsig: torch.Tensor
    attn_mask: torch.Tensor
    meta: dict

    @property
    def num_queries(self):
        return self.meta["num_denoising_queries"]


def dn_target_indices(meta, b, device=None):
    """按每组 [M 个正位置 | M 个负位置] 返回有效目标位置，跳过 padding。"""
    n = meta["num_target_list"][b]
    G = meta["num_denoising_groups"]
    M = meta["max_num_target"]
    t = torch.arange(n, dtype=torch.long, device=device).unsqueeze(0).repeat(G, 1)
    pos = (torch.arange(G, dtype=torch.long, device=device).unsqueeze(1) * 2 * M + t).flatten()
    return pos, t.flatten(), pos + M


class CdnQueryGenerator(nn.Module):
    """构建正负 DN 查询；整个 batch 无目标时返回 None。"""

    def __init__(self, embed_dims, num_matching_queries, label_book_size=1, label_noise_scale=0.5,
                 box_noise_scale=1.0, num_dn_queries=100):
        super().__init__()
        assert label_noise_scale > 0, "DN 标签扰动要求 label_noise_scale > 0"
        assert box_noise_scale > 0, "DN 框扰动要求 box_noise_scale > 0"
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_book_size = label_book_size
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.num_dn_queries = num_dn_queries

        self.label_embedding = nn.Embedding(label_book_size, embed_dims)

    def get_num_groups(self, max_num_target):
        """G=max(floor(Kdn/M),1)。"""
        if max_num_target == 0:
            return 1
        return max(int(self.num_dn_queries // max_num_target), 1)

    def forward(self, gt_boxes_list, gt_labels_list=None):
        """输入每张图的归一化 cxcywh 框，可选标签默认为 0。"""
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
        """以 rho/2 概率替换标签；单标签表中替换仍为 0。"""
        gt_labels_expand = gt_labels.repeat(2 * num_groups, 1).view(-1)
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(-1)
        new_labels = torch.randint_like(chosen_indice, 0, self.label_book_size)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice, new_labels)
        return self.label_embedding(noisy_labels_expand)

    def generate_dn_bbox_query(self, gt_bboxes, num_groups):
        """归一化 xyxy 上加正负扰动，clip 后转 cxcywh，并用 eps=1e-3 编码。"""
        device = gt_bboxes.device
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)

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
        """按每张图的有效目标数填入 DN 分组，其余位置补零。"""
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
        """普通查询不能读取 DN；DN 组之间不可见；DN 可读取组内及普通查询。"""
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
