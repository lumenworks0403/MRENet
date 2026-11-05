"""
B0_HED  HED+DN — CPU 单元测试 (在服务器上运行)。

调用的都是生产函数:
  rrs/dn.py            CdnQueryGenerator / dn_target_indices
  rrs/hed_decoder.py   hed_parallel_decode (no-DN) / hed_parallel_decode_dn / sample_branch_dns
  rrs/criterion.py     SetCriterion.loss_dn_single / forward (需要可导入 groundingdino)
解码器层用最小 stub 代替 (真正的 self-attention + 用参考框的线性项), 以便在 CPU 上
验证 mask 方向、索引配对与梯度; 真实模型的前向/反向见 tools/smoke_dn.py。

运行:
    cd B0_HED
    python tests/test_dn.py
"""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from rrs.dn import CdnQueryGenerator, dn_target_indices, inverse_sigmoid
from rrs.hed_decoder import hed_parallel_decode, hed_parallel_decode_dn, sample_branch_dns

try:  # criterion 依赖 groundingdino (box_ops) 与 scipy
    from rrs.criterion import SetCriterion, HungarianMatcher
    _CRIT_ERR = None
except Exception as e:  # noqa: BLE001
    _CRIT_ERR = repr(e)

C, Q, B = 16, 6, 2


def _sine_stub(pos):
    """gen_sineembed_for_position 的 stub: 线性展开为 2 倍维度, 让参考框真正影响 query_pos。"""
    return torch.cat([pos, pos], -1).repeat(1, 1, C // 4)  # (nq, B, 8*(C//4)) = (nq, B, 2C)


class _Layer(nn.Module):
    """stub 解码器层: 真正的 self-attention (使用 self_attn_mask) + 参考框线性项。"""

    def __init__(self):
        super().__init__()
        self.sa = nn.MultiheadAttention(C, 2)
        self.ref = nn.Linear(4, C)

    def forward(self, tgt, tgt_query_pos=None, tgt_reference_points=None, self_attn_mask=None,
                **kw):
        q = tgt + tgt_query_pos
        out = tgt + self.sa(q, q, tgt, attn_mask=self_attn_mask)[0]
        return out + self.ref(tgt_reference_points[:, :, 0, :])


class _Decoder(nn.Module):

    def __init__(self, n=6):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])
        self.norm = nn.LayerNorm(C)
        self.bbox_embed = nn.ModuleList([nn.Linear(C, 4) for _ in range(n)])
        self.ref_point_head = nn.Linear(2 * C, C)


def _targets(ns):
    g = torch.Generator().manual_seed(7)
    out = []
    for n in ns:
        cxcy = torch.rand(n, 2, generator=g) * 0.6 + 0.2
        wh = torch.rand(n, 2, generator=g) * 0.3 + 0.1
        out.append({"boxes": torch.cat([cxcy, wh], 1)})
    return out


def _dec_kwargs(decoder, tgt=None):
    torch.manual_seed(0)
    nlv = 2
    mem = torch.randn(20, B, C)
    return dict(
        decoder=decoder,
        tgt=torch.randn(Q, B, C) if tgt is None else tgt,
        memory=mem,
        valid_ratios=torch.ones(B, nlv, 2),
        spatial_shapes=torch.tensor([[4, 4], [2, 2]]),
        level_start_index=torch.tensor([0, 16]),
        memory_key_padding_mask=torch.zeros(B, 20, dtype=torch.bool),
        refpoints_unsigmoid=torch.randn(Q, B, 4) * 0.5,
        memory_text=torch.randn(B, 5, C),
        text_attention_mask=torch.zeros(B, 5, dtype=torch.bool),
        memory_pos=torch.randn_like(mem),
        gen_sineembed_for_position=_sine_stub,
    )


class TestGenerator(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.gen = CdnQueryGenerator(C, Q, label_book_size=1, num_dn_queries=6)

    # 测试2: 数量、分组、索引、padding
    def test_counts_groups_layout(self):
        tg = _targets([1, 3])
        d = self.gen([t["boxes"] for t in tg])
        M, G = 3, 6 // 3
        self.assertEqual(d.meta["max_num_target"], M)
        self.assertEqual(d.meta["num_denoising_groups"], G)
        self.assertEqual(d.num_queries, M * 2 * G)
        self.assertEqual(d.label_query.shape, (B, M * 2 * G, C))
        self.assertEqual(d.bbox_unsig.shape, (B, M * 2 * G, 4))
        # 图0 只有 1 个目标: 每组正区/负区只有第 0 位非零, 其余为 padding (全 0)
        for g in range(G):
            base = g * 2 * M
            for k in range(2 * M):
                filled = k in (0, M)
                self.assertEqual(
                    bool(d.bbox_unsig[0, base + k].abs().sum() > 0)
                    or bool(d.label_query[0, base + k].abs().sum() > 0), filled, (g, k))
        pos, gt, neg = dn_target_indices(d.meta, 1)
        self.assertEqual(pos.tolist(), [0, 1, 2, 6, 7, 8])
        self.assertEqual(gt.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(neg.tolist(), [3, 4, 5, 9, 10, 11])
        pos0, _, _ = dn_target_indices(d.meta, 0)
        self.assertEqual(pos0.tolist(), [0, 6])

    def test_dynamic_groups(self):
        self.assertEqual(CdnQueryGenerator(C, Q, num_dn_queries=100).get_num_groups(1), 100)
        self.assertEqual(CdnQueryGenerator(C, Q, num_dn_queries=100).get_num_groups(30), 3)
        self.assertEqual(CdnQueryGenerator(C, Q, num_dn_queries=100).get_num_groups(300), 1)

    # 测试2: 噪声范围 (正样本角点偏移 < wh/2, 负样本在 [wh/2, wh) 之间, 未 clamp 时)
    def test_noise_range(self):
        torch.manual_seed(1)
        tg = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}] * 1
        gen = CdnQueryGenerator(C, Q, num_dn_queries=200)
        d = gen([tg[0]["boxes"]])
        boxes = d.bbox_unsig[0].sigmoid()
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        gt = torch.tensor([0.4, 0.4, 0.6, 0.6])
        dev = (xyxy - gt).abs() / 0.1  # 以 wh/2=0.1 为单位
        G = d.meta["num_denoising_groups"]
        pos = torch.arange(G) * 2
        neg = pos + 1
        tol = 1e-3  # inverse_sigmoid/sigmoid 往返与 cxcywh 换算的浮点误差 (以 wh/2 为单位)
        self.assertTrue((dev[pos] < 1 + tol).all(), "正样本偏移应 < box_noise_scale*wh/2")
        # 负样本的角点可能交叉 (x0>x1, 官方同样不处理), 宽高变负后被 inverse_sigmoid
        # 截到 eps, 无法从 cxcywh 还原角点; 只检查未交叉的负样本
        ok = (w[neg] > 2e-3) & (h[neg] > 2e-3)
        self.assertGreater(int(ok.sum()), G // 4)
        dn_ = dev[neg][ok]
        self.assertTrue((dn_ >= 1 - tol).all() and (dn_ < 2 + tol).all(), "负样本偏移应在 [1,2)*wh/2")

    def test_label_query_uses_embedding_row0(self):
        d = self.gen([t["boxes"] for t in _targets([2, 2])])
        # label_book_size=1: 类别噪声换回 0, 所有非 padding 位都是 embedding 第 0 行
        row = self.gen.label_embedding.weight[0]
        self.assertTrue(torch.allclose(d.label_query[0, 0], row))
        self.assertTrue(torch.allclose(d.label_query[1, d.num_queries - 1], row))

    def test_empty_batch_returns_none(self):
        self.assertIsNone(self.gen([torch.zeros(0, 4), torch.zeros(0, 4)]))

    # 测试3: attention mask 方向、分组隔离 (显式小矩阵)
    def test_attn_mask_explicit(self):
        gen = CdnQueryGenerator(C, num_matching_queries=2, num_dn_queries=2)
        m = gen.generate_dn_mask(max_num_target=1, num_groups=2, device="cpu").int()
        # 序列: [g0P g0N g1P g1N | m0 m1]; 1 = 看不见
        expected = torch.tensor([
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ])
        self.assertTrue(torch.equal(m, expected), m)

    # 测试4: 固定随机种子可复现
    def test_reproducible(self):
        tg = [t["boxes"] for t in _targets([2, 3])]
        torch.manual_seed(5)
        a = self.gen(tg)
        torch.manual_seed(5)
        b = self.gen(tg)
        self.assertTrue(torch.equal(a.bbox_unsig, b.bbox_unsig))
        self.assertTrue(torch.equal(a.label_query, b.label_query))
        torch.manual_seed(6)
        c = self.gen(tg)
        self.assertFalse(torch.equal(a.bbox_unsig, c.bbox_unsig))


class TestDecoderDN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.dec = _Decoder()
        self.gen = CdnQueryGenerator(C, Q, num_dn_queries=4)
        self.tg = [t["boxes"] for t in _targets([1, 2])]

    def _run(self, rate, seed=0, mode="sigmoid", kw=None):
        torch.manual_seed(seed)
        dn0 = self.gen(self.tg)
        branch = sample_branch_dns(self.gen, self.tg, 5, rate, random.Random(seed))
        kw = kw or _dec_kwargs(self.dec)
        return dn0, branch, hed_parallel_decode_dn(dn0=dn0, branch_dns=branch, branch_ref_mode=mode,
                                                   **kw)

    # 测试1 + 6: 普通查询部分与修改前 no-DN 版逐元素相同 (DN 不会挤错或泄漏到普通查询)
    def test_matching_part_equals_noDN(self):
        kw = _dec_kwargs(self.dec)
        hs0, refs0, irefs0 = hed_parallel_decode(**kw)
        for rate in (0.0, 0.5, 1.0):
            _, _, out = self._run(rate, kw=kw)
            n = out["num_dn"]
            for i in range(6):
                self.assertTrue(torch.allclose(out["hs"][i][:, n:], hs0[i], atol=1e-5),
                                f"rate={rate} 层{i}")
                self.assertTrue(torch.allclose(out["input_refs"][i][:, n:], irefs0[i], atol=1e-6))
            self.assertTrue(torch.allclose(out["refs"][1][:, n:], refs0[1], atol=1e-6))

    # 测试3: 普通查询不能偷看 DN —— 扰动 DN 内容, 普通部分不变; DN 部分会变
    def test_no_leak_into_matching(self):
        kw = _dec_kwargs(self.dec)
        torch.manual_seed(0)
        dn0 = self.gen(self.tg)
        branch = [None] * 5
        a = hed_parallel_decode_dn(dn0=dn0, branch_dns=branch, **kw)
        dn0.label_query = dn0.label_query + torch.randn_like(dn0.label_query) * 3
        b = hed_parallel_decode_dn(dn0=dn0, branch_dns=branch, **kw)
        n = a["num_dn"]
        for i in range(6):
            self.assertTrue(torch.allclose(a["hs"][i][:, n:], b["hs"][i][:, n:], atol=1e-5))
            self.assertFalse(torch.allclose(a["hs"][i][:, :n], b["hs"][i][:, :n], atol=1e-5))

    # 测试3: DN 组之间隔离 —— 只扰动第 1 组, 第 0 组输出不变
    def test_group_isolation(self):
        kw = _dec_kwargs(self.dec)
        torch.manual_seed(0)
        dn0 = self.gen(self.tg)
        M = dn0.meta["max_num_target"]
        self.assertGreaterEqual(dn0.meta["num_denoising_groups"], 2)
        a = hed_parallel_decode_dn(dn0=dn0, branch_dns=[None] * 5, **kw)
        lq = dn0.label_query.clone()
        lq[:, 2 * M:4 * M] += 5.0
        dn0.label_query = lq
        b = hed_parallel_decode_dn(dn0=dn0, branch_dns=[None] * 5, **kw)
        self.assertTrue(torch.allclose(a["hs"][0][:, :2 * M], b["hs"][0][:, :2 * M], atol=1e-5))
        self.assertFalse(
            torch.allclose(a["hs"][0][:, 2 * M:4 * M], b["hs"][0][:, 2 * M:4 * M], atol=1e-5))

    # 测试5: τ=0 全部继承 (仍有 DN), τ=1 全部重新生成
    def test_reinit_rate_bounds(self):
        dn0, branch, out = self._run(0.0)
        self.assertTrue(all(b is None for b in branch))
        self.assertGreater(out["num_dn"], 0, "τ=0 不等于关闭 DN")
        self.assertTrue(all(d is dn0 for d in out["layer_dn"]))
        # 继承: 分支 DN 输入 = 共享层 DN 输出
        n = out["num_dn"]
        for i in range(1, 6):
            self.assertTrue(torch.allclose(out["input_refs"][i], out["refs"][1].detach(),
                                           atol=1e-6))
        dn0, branch, out = self._run(1.0)
        self.assertTrue(all(b is not None for b in branch))
        for i in range(1, 6):
            self.assertIs(out["layer_dn"][i], branch[i - 1])
            self.assertIsNot(out["layer_dn"][i], dn0)

    # 测试7: 每个重新生成分支用自己的参考框和 mask, 不混用其他分支
    def test_branch_uses_own_dn(self):
        dn0, branch, out = self._run(1.0)
        n = out["num_dn"]
        for i in range(1, 6):
            own = branch[i - 1]
            self.assertTrue(
                torch.allclose(out["input_refs"][i][:, :n], own.bbox_unsig.sigmoid(), atol=1e-6))
            self.assertTrue(
                torch.allclose(out["head_refs"][i][:, :n], own.bbox_unsig.sigmoid(), atol=1e-6))
            for j in range(1, 6):
                if j != i:
                    self.assertFalse(
                        torch.allclose(out["input_refs"][i][:, :n],
                                       branch[j - 1].bbox_unsig.sigmoid(), atol=1e-4))
        # official_raw: 照搬官方, 分支 DN 参考框 = 未 sigmoid 的值
        _, branch, out = self._run(1.0, mode="official_raw")
        self.assertTrue(torch.allclose(out["input_refs"][1][:, :n], branch[0].bbox_unsig,
                                       atol=1e-6))

    def test_rate_half_reproducible(self):
        _, b1, o1 = self._run(0.5, seed=3)
        _, b2, o2 = self._run(0.5, seed=3)
        self.assertEqual([x is None for x in b1], [x is None for x in b2])
        for i in range(6):
            self.assertTrue(torch.equal(o1["hs"][i], o2["hs"][i]))

    # 测试9 (stub 级): 共享层、分支、DN embedding 都有有限梯度
    def test_gradients(self):
        _, branch, out = self._run(0.5, seed=1)
        loss = sum(h.sum() for h in out["hs"]) + sum(r.sum() for r in out["refs"][1:])
        loss.backward()
        for name, p in list(self.dec.named_parameters()) + list(self.gen.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(self.gen.label_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.dec.layers[0].sa.in_proj_weight.grad.abs().sum().item(), 0)


@unittest.skipIf(_CRIT_ERR is not None, f"criterion 导入失败 (需服务器环境): {_CRIT_ERR}")
class TestDNLoss(unittest.TestCase):
    T = 8  # 文本长度 (max_text_len 用 256 以对齐 build_positive_map)

    def _crit(self):
        m = HungarianMatcher(cost_mask=0, cost_dice=0)
        wd = {
            k: 1.0
            for k in ("loss_ce", "loss_bbox", "loss_giou", "loss_mask", "loss_dice", "loss_ce_dn",
                      "loss_bbox_dn", "loss_giou_dn")
        }
        return SetCriterion(m, wd)

    def _setup(self, ns):
        torch.manual_seed(0)
        gen = CdnQueryGenerator(C, Q, num_dn_queries=4)
        tg = _targets(ns)
        d = gen([t["boxes"] for t in tg])
        return gen, tg, d

    # 测试8: 正负样本监督与归一化
    def test_pos_neg_targets_and_norm(self):
        gen, tg, d = self._setup([1, 2])
        crit = self._crit()
        ndn = d.num_queries
        pos_map = torch.zeros(B, 256)
        pos_map[0, 1:3] = 1
        pos_map[1, 1:4] = 1
        text_mask = torch.zeros(B, 256, dtype=torch.bool)
        text_mask[:, :self.T] = True
        # 构造 "完美" 预测: 正样本框 = GT, logits 与目标一致 -> 框损失为 0
        boxes = torch.full((B, ndn, 4), 0.5)
        logits = torch.full((B, ndn, 256), -20.0)
        npos = 0
        for b in range(B):
            pos, gi, neg = dn_target_indices(d.meta, b)
            boxes[b, pos] = tg[b]["boxes"][gi]
            logits[b, pos] = pos_map[b] * 40 - 20
            npos += len(pos)
        l = crit.loss_dn_single(logits, boxes, tg, d.meta, pos_map, text_mask)
        self.assertLess(l["loss_bbox_dn"].item(), 1e-6)
        # 框完全重合时 1-GIoU 不为 0: box_ops.generalized_box_iou 的 iou = inter/(union+1e-6),
        # 残差 = 1 - a/(a+1e-6)。按理论值比较, 框若没对上仍会失败。
        expected = 0.0
        for b in range(B):
            _, gi, _ = dn_target_indices(d.meta, b)
            area = tg[b]["boxes"][gi, 2] * tg[b]["boxes"][gi, 3]
            expected += (1 - area / (area + 1e-6)).sum().item()
        expected /= npos
        self.assertAlmostEqual(l["loss_giou_dn"].item(), expected, delta=1e-6)
        self.assertLess(l["loss_ce_dn"].item(), 1e-4)
        # 把一个负样本 DN 的 logits 也设成正 -> 只有分类损失上升, 框损失不变 (负样本不算框)
        _, _, neg = dn_target_indices(d.meta, 1)
        logits2 = logits.clone()
        logits2[1, neg[0]] = pos_map[1] * 40 - 20
        boxes2 = boxes.clone()
        boxes2[1, neg[0]] = 0.9
        l2 = crit.loss_dn_single(logits2, boxes2, tg, d.meta, pos_map, text_mask)
        self.assertGreater(l2["loss_ce_dn"].item(), l["loss_ce_dn"].item() + 1e-3)
        self.assertAlmostEqual(l2["loss_bbox_dn"].item(), l["loss_bbox_dn"].item(), places=6)
        # 归一化: 某个正样本框偏移 0.1 (4 个坐标) -> L1 = 0.4 / 正样本总数
        boxes3 = boxes.clone()
        pos1, _, _ = dn_target_indices(d.meta, 1)
        boxes3[1, pos1[0]] += 0.1
        l3 = crit.loss_dn_single(logits, boxes3, tg, d.meta, pos_map, text_mask)
        self.assertAlmostEqual(l3["loss_bbox_dn"].item(), 0.4 / npos, places=5)
        # padding 文本位不计入: 改 text_mask 外的 logits 不影响损失
        logits4 = logits.clone()
        logits4[:, :, self.T:] = 30.0
        l4 = crit.loss_dn_single(logits4, boxes, tg, d.meta, pos_map, text_mask)
        self.assertAlmostEqual(l4["loss_ce_dn"].item(), l["loss_ce_dn"].item(), places=6)

    # 测试8: 空目标图片 / 混合精度 loss 有限
    def test_empty_and_fp16_finite(self):
        gen, tg, d = self._setup([0, 2])
        crit = self._crit()
        pos_map = torch.zeros(B, 256)
        pos_map[:, 1:3] = 1
        text_mask = torch.zeros(B, 256, dtype=torch.bool)
        text_mask[:, :self.T] = True
        logits = torch.randn(B, d.num_queries, 256).half()
        boxes = torch.rand(B, d.num_queries, 4).half() * 0.5 + 0.25
        l = crit.loss_dn_single(logits, boxes, tg, d.meta, pos_map, text_mask)
        for k, v in l.items():
            self.assertTrue(torch.isfinite(v).all(), k)
        self.assertEqual(dn_target_indices(d.meta, 0)[0].numel(), 0)

    # 测试10 + 8: DN 不进入 Hungarian 匹配, 不改变普通损失的值与归一化
    def test_dn_not_in_matching(self):
        crit = self._crit()
        gen, tg, d = self._setup([1, 2])
        for t in tg:
            t["masks"] = torch.zeros(len(t["boxes"]), 8, 8)
            t["size"] = torch.tensor([8, 8])
        tok = {
            "input_ids": torch.tensor([[101, 2000, 2001, 102, 0, 0, 0, 0]] * B),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]] * B)
        }
        base = {
            "pred_logits": torch.randn(B, Q, 256),
            "pred_boxes": torch.rand(B, Q, 4) * 0.5 + 0.25,
            "pred_masks": torch.randn(B, Q, 2, 2),
            "aux_outputs": [],
            "text_token_mask": tok["attention_mask"].bool(),
            "tokenized": tok,
            "pad_mask": torch.zeros(B, 8, 8, dtype=torch.bool)
        }
        torch.manual_seed(0)
        tot0, ld0, idx0 = crit(dict(base), tg)
        dn_out = {
            "pred_logits": torch.randn(6, B, d.num_queries, 256),
            "pred_boxes": torch.rand(6, B, d.num_queries, 4) * 0.5 + 0.25,
            "layer_meta": [d.meta] * 6
        }
        torch.manual_seed(0)
        tot1, ld1, idx1 = crit({**base, "dn_outputs": dn_out}, tg)
        for (a, b), (c, e) in zip(idx0, idx1):
            self.assertTrue(torch.equal(a, c) and torch.equal(b, e))
            self.assertTrue((a < Q).all())
        for k in ld0:
            self.assertAlmostEqual(ld0[k].item(), ld1[k].item(), places=6, msg=k)
        dn_keys = sorted(k for k in ld1 if "_dn" in k)
        self.assertEqual(len(dn_keys), 18)  # 6 层 × (ce, bbox, giou)
        self.assertIn("loss_ce_dn", ld1)
        self.assertIn("loss_ce_dn_0", ld1)
        self.assertAlmostEqual((tot1 - tot0).item(), sum(ld1[k] for k in dn_keys).item(), places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
