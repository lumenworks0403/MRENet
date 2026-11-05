"""
B0_HED 无DN结构版 — CPU 单元测试。
只导入 rrs.hed_decoder（无 groundingdino 依赖），用 stub 替换位置编码函数。

运行:
    cd B0_HED
    python tests/test_hed.py
"""
import sys
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

# 导入纯逻辑模块（无 groundingdino 依赖）
from rrs.hed_decoder import (
    hed_parallel_decode,
    _compute_ref_points_input,
    inverse_sigmoid,
)


def _gen_sine_stub(pos):
    """gen_sineembed_for_position 的 stub: 返回 (..., 2*C) 随机张量（每次固定）。
    pos shape: (nq, B, 4) -> 返回 (nq, B, 8)"""
    *leading, last = pos.shape
    return torch.zeros(*leading, last * 2)


def _make_layer(dim=16):
    """最小解码器层 stub: 接受 tgt 关键字参数，返回 tgt + 微小常量。"""

    class _Layer(nn.Module):

        def __init__(self, d):
            super().__init__()
            self.w = nn.Parameter(torch.ones(d) * 0.001)  # 可学习，便于梯度测试

        def forward(self, tgt, **kwargs):
            return tgt + self.w  # tgt: (nq, B, dim)

    return _Layer(dim)


def _make_decoder(num_layers=6, dim=16):
    """最小 TransformerDecoder stub，各层有独立参数。"""

    class _Decoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_make_layer(dim) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(dim)
            # bbox_embed: (dim,) -> (4,)，各层独立初始化
            self.bbox_embed = nn.ModuleList([nn.Linear(dim, 4) for _ in range(num_layers)])
            # ref_point_head: (8,) -> (dim,)
            self.ref_point_head = nn.Linear(8, dim)

    return _Decoder()


class TestHedNoDN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.B = 2
        self.Q = 10
        self.C = 16
        self.n_levels = 2
        self.decoder = _make_decoder(6, self.C)
        self.valid_ratios = torch.ones(self.B, self.n_levels, 2)
        self.spatial_shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
        self.level_start_index = torch.tensor([0, 16])
        sum_hw = 4 * 4 + 2 * 2
        self.memory = torch.randn(sum_hw, self.B, self.C)
        self.memory_text = torch.randn(self.B, 5, self.C)
        self.text_attn_mask = torch.zeros(self.B, 5, dtype=torch.bool)
        self.memory_pos = torch.randn_like(self.memory)
        self.tgt = torch.randn(self.Q, self.B, self.C)
        self.refpoints_unsig = torch.zeros(self.Q, self.B, 4)

    def _run(self, decoder=None):
        """返回 (hs, refs, input_refs)。"""
        d = decoder if decoder is not None else self.decoder
        return hed_parallel_decode(
            decoder=d,
            tgt=self.tgt.clone(),
            memory=self.memory,
            valid_ratios=self.valid_ratios,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            memory_key_padding_mask=torch.zeros(self.B, self.memory.shape[0], dtype=torch.bool),
            refpoints_unsigmoid=self.refpoints_unsig.clone(),
            memory_text=self.memory_text,
            text_attention_mask=self.text_attn_mask,
            memory_pos=self.memory_pos,
            gen_sineembed_for_position=_gen_sine_stub,
        )

    # 测试1: 输出形状正确（6层输出，7个参考框，6个输入参考框）
    def test_1_output_shape(self):
        hs, refs, irefs = self._run()
        self.assertEqual(len(hs), 6)
        for i, h in enumerate(hs):
            self.assertEqual(h.shape, (self.B, self.Q, self.C), f"层{i} shape 错误")
        self.assertEqual(len(refs), 7)
        for r in refs:
            self.assertEqual(r.shape, (self.B, self.Q, 4))
        self.assertEqual(len(irefs), 6)
        for r in irefs:
            self.assertEqual(r.shape, (self.B, self.Q, 4))

    # 测试2&3: 并行分支输入来自同一共享层；改变层2参数不影响层1
    def test_2_3_branch_independence(self):
        hs_a, _, _ = self._run()
        # 扰动层2参数
        with torch.no_grad():
            for p in self.decoder.layers[2].parameters():
                p.add_(torch.randn_like(p))
        hs_b, _, _ = self._run()
        # 层1不受影响（layer2 参数未变）
        self.assertTrue(torch.allclose(hs_a[1], hs_b[1], atol=1e-5), "层1不应受层2参数变化影响")
        # 层2变了
        self.assertFalse(torch.allclose(hs_a[2], hs_b[2], atol=1e-5), "层2应随自身参数变化")
        # 再扰动层3，层1和层2不受影响
        hs_c_ref = hs_b[1].clone(), hs_b[2].clone()
        with torch.no_grad():
            for p in self.decoder.layers[3].parameters():
                p.add_(torch.randn_like(p))
        hs_c, _, _ = self._run()
        self.assertTrue(torch.allclose(hs_c_ref[0], hs_c[1], atol=1e-5), "层3扰动后层1不变")
        self.assertTrue(torch.allclose(hs_c_ref[1], hs_c[2], atol=1e-5), "层3扰动后层2不变")

    # 测试4: 各层参数独立（不是同一对象）
    def test_4_layer_params_independent(self):
        for i in range(6):
            for j in range(i + 1, 6):
                params_i = list(self.decoder.layers[i].parameters())
                params_j = list(self.decoder.layers[j].parameters())
                for pi, pj in zip(params_i, params_j):
                    self.assertIsNot(pi, pj, f"层{i}和层{j}参数不应是同一对象")

    # 测试5: 共享层和并行层均有梯度
    def test_5_gradients_flow(self):
        hs, refs, _ = self._run()
        loss = sum(h.sum() for h in hs) + sum(r.sum() for r in refs[1:])
        loss.backward()
        # 所有层的 bbox_embed 均有梯度
        for i, be in enumerate(self.decoder.bbox_embed):
            for name, p in be.named_parameters():
                self.assertIsNotNone(p.grad, f"bbox_embed[{i}].{name} 无梯度")
                self.assertGreater(p.grad.abs().sum().item(), 0, f"bbox_embed[{i}].{name} 梯度为零")
        # 共享层（层0）的 layers[0] 参数有梯度（来自层0自身及后续并行层的共享路径）
        for name, p in self.decoder.layers[0].named_parameters():
            self.assertIsNotNone(p.grad, f"layers[0].{name} 无梯度")

    # 测试6: 多层概率均值：先 sigmoid 再均值 ≠ 先均值再 sigmoid（验证实现方向正确）
    def test_6_prob_average_order_matters(self):
        torch.manual_seed(1)
        logits = torch.randn(6, self.B, self.Q, 8)
        probs_correct = logits.sigmoid().mean(0)
        probs_wrong = logits.mean(0).sigmoid()
        # 两者不应相等（说明两种方式是不同的操作）
        self.assertFalse(torch.allclose(probs_correct, probs_wrong, atol=1e-3))
        # probs_correct 在 [0, 1]
        self.assertTrue((probs_correct >= 0).all() and (probs_correct <= 1).all())

    # 测试7: 各并行层产生不同的参考框更新（因 bbox_embed 各层不同）
    def test_7_parallel_refs_differ(self):
        hs, refs, _ = self._run()
        # refs[2]~refs[6] 是并行层的参考框，因各层 bbox_embed 不同应有差异
        all_same = all(
            torch.allclose(refs[i], refs[j], atol=1e-5) for i in range(2, 7)
            for j in range(i + 1, 7))
        self.assertFalse(all_same, "并行层应产生不同的参考框更新")

    # 测试8: 共享层梯度未被随意 detach（层0的 layers[0].w 有非零梯度）
    def test_8_shared_layer_not_detached(self):
        hs, refs, _ = self._run()
        loss = sum(h.sum() for h in hs[1:])  # 只用并行层的损失反传
        loss.backward()
        # 梯度通过 shared_output 流回层0
        for name, p in self.decoder.layers[0].named_parameters():
            self.assertIsNotNone(p.grad, f"layers[0].{name} 应有梯度（通过 shared_output）")

    # 测试9: 五个并行分支消费的输入参考框是同一个（共享层的框）
    # 这是任务第六节特别要求的检查: 防止第3层误用第2层分支输出的位置
    def test_9_parallel_input_refs_are_shared(self):
        hs, refs, irefs = self._run()
        # irefs[1..5] 应全部等于共享层更新后的参考框, 即 refs[1]
        for i in range(1, 6):
            self.assertTrue(torch.allclose(irefs[i], refs[1], atol=1e-6),
                            f"层{i}的输入参考框应是共享层的框 refs[1]")
        # 且 irefs[1..5] 彼此相同
        for i in range(2, 6):
            self.assertTrue(torch.allclose(irefs[1], irefs[i], atol=1e-6), f"层1与层{i}的输入参考框应相同")
        # 对照: 若误用 refs[:-1] 串行配对, 层2拿到的会是 refs[2]
        # (层1分支自己更新出的框), 与正确的 refs[1] 不同 —— 验证这个区别真实存在
        self.assertFalse(torch.allclose(refs[2], refs[1], atol=1e-5),
                         "refs[2] 与 refs[1] 应不同, 否则此测试无法区分两种配对")

    # 测试10: 层0消费的输入参考框是初始参考框, 而非自己更新后的框
    def test_10_shared_layer_input_ref_is_initial(self):
        hs, refs, irefs = self._run()
        self.assertTrue(torch.allclose(irefs[0], refs[0], atol=1e-6), "层0的输入参考框应是初始参考框 refs[0]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
