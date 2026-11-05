"""MMSCopE 像素支路与 RefScoreHead 的控制流 / 形状 / 损失测试 (CPU, 不需要 CUDA 与扫描算子)。

    python tests/test_mm_score.py

覆盖:
  1. RefScoreHead 输出形状 (B=1 / Q=1 时不丢维)。
  2. 评分损失: 标签来自各层自己的匹配结果、归一化、无目标 batch 可反传、
     进入 total loss、辅助层后缀能找到权重、缺字段报错、DN 不增加评分损失。
  3. 后处理: 新评分决定 q*、六层同一索引的掩码集成、token 分数只作诊断、缺字段报错。
  4. MMSCopEPixelDecoder: 非方形/非整除尺寸下 pad→裁回 的输出尺寸与 c4 对齐,
     以及梯度能从 pix 回传到 c4 / enc_maps (真实 SS2D 用桩替换, 只验证接线与几何)。
  5. init_model_weights 的键差异规则。

注意: 第 4 项把 VSSMBlock 换成了桩, 不构成对真实 selective scan 的验证;
真实算子的前向/反向在 tools/smoke_mm_score.py (需 GPU + 编译好的扩展) 里跑。
"""
import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rrs.ref_score_head import RefScoreHead  # noqa: E402

try:
    # rrs.model / rrs.criterion 需要 groundingdino 可导入 (服务器环境)
    from rrs.criterion import HungarianMatcher, SetCriterion
    from rrs.evaluate import postprocess_hed
    from rrs.model import REF_SCORE_PREFIX, init_model_weights
    _ERR = None
except Exception as e:  # pragma: no cover
    REF_SCORE_PREFIX = "ref_score_head."
    _ERR = e

B, Q, L, T = 2, 5, 6, 8  # batch / 普通查询数 / HED 层数 / 文本长度


class TestRefScoreHead(unittest.TestCase):

    def test_shapes(self):
        head = RefScoreHead(16)
        self.assertEqual(head(torch.randn(3, 7, 16)).shape, (3, 7))
        self.assertEqual(head(torch.randn(1, 4, 16)).shape, (1, 4))  # B=1 不能丢 batch 维
        self.assertEqual(head(torch.randn(1, 1, 16)).shape, (1, 1))  # Q=1 也不能丢

    def test_shared_across_layers(self):
        head = RefScoreHead(16)
        hs = [torch.randn(B, Q, 16) for _ in range(L)]
        outs = [head(h) for h in hs]
        self.assertEqual(len(outs), L)
        # 同一输入必须给出同一输出 (六层共用一份权重, 不是六个独立评分器)
        self.assertTrue(torch.equal(head(hs[0]), outs[0]))


def _targets(ns, hw=8):
    out = []
    for n in ns:
        out.append({
            "boxes": torch.rand(n, 4) * 0.4 + 0.3,
            "masks": torch.zeros(n, hw, hw),
            "size": torch.tensor([hw, hw])
        })
    return out


def _tokenized():
    return {
        "input_ids": torch.tensor([[101, 2000, 2001, 102, 0, 0, 0, 0]] * B),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]] * B)
    }


@unittest.skipIf(_ERR is not None, f"criterion/evaluate 导入失败 (需服务器环境): {_ERR}")
class TestRefScoreLoss(unittest.TestCase):

    def _crit(self, coef=1.0):
        m = HungarianMatcher(cost_mask=0, cost_dice=0)
        wd = {k: 1.0 for k in ("loss_ce", "loss_bbox", "loss_giou", "loss_mask", "loss_dice")}
        wd["loss_ref_score"] = coef
        return SetCriterion(m, wd)

    def _outputs(self, ns=(1, 2), with_ref=True, layers=L):
        tok = _tokenized()
        out = {
            "pred_logits": torch.randn(B, Q, 256),
            "pred_boxes": torch.rand(B, Q, 4) * 0.4 + 0.3,
            "pred_masks": torch.randn(B, Q, 2, 2),
            "text_token_mask": tok["attention_mask"].bool(),
            "tokenized": tok,
            "pad_mask": torch.zeros(B, 8, 8, dtype=torch.bool)
        }
        out["aux_outputs"] = [{
            "pred_logits": torch.randn(B, Q, 256),
            "pred_boxes": torch.rand(B, Q, 4) * 0.4 + 0.3,
            "pred_masks": torch.randn(B, Q, 2, 2)
        } for _ in range(layers - 1)]
        if with_ref:
            out["pred_ref_logits"] = torch.randn(B, Q, requires_grad=True)
            for aux in out["aux_outputs"]:
                aux["pred_ref_logits"] = torch.randn(B, Q, requires_grad=True)
        return out, _targets(ns)

    def test_label_comes_from_that_layers_matching(self):
        """标签 = 该层自己的匹配结果; 不是最后一层的标签复制给其他层, 也不是模型自己的最高分。"""
        crit = self._crit()
        ref = torch.zeros(B, Q)
        tg = _targets([1, 1])
        indices = [(torch.tensor([2]), torch.tensor([0])), (torch.tensor([4]), torch.tensor([0]))]
        # 用极端 logits 反推标签: 匹配位给 +20, 其余 -20 -> 损失应接近 0
        ref[0, 2] = 20.0
        ref[1, 4] = 20.0
        ref = ref - 20.0 * (ref == 0).float()
        num_boxes = float(sum(len(t["boxes"]) for t in tg))
        l = crit.loss_ref_score(ref, indices, num_boxes)["loss_ref_score"]
        self.assertLess(l.item(), 1e-5)
        # 把标签换成另一层的匹配 (查询 1) -> 同一预测的损失必须变大
        other = [(torch.tensor([1]), torch.tensor([0])), (torch.tensor([1]), torch.tensor([0]))]
        self.assertGreater(crit.loss_ref_score(ref, other, num_boxes)["loss_ref_score"].item(), 1.0)

    def test_normalization_matches_other_losses(self):
        """归一化除以 max(batch GT 数, 1), 与其余损失一致。"""
        crit = self._crit()
        ref = torch.zeros(B, Q)  # logits=0 -> p=0.5
        idx = [(torch.zeros(0, dtype=torch.int64), ) * 2] * B  # 全负样本
        # 每个元素: (1-alpha) * 0.5^gamma * (-log(0.5)) = 0.75 * 0.25 * ln2
        per = 0.75 * 0.25 * torch.log(torch.tensor(2.0)).item()
        for num_boxes in (1.0, 3.0):
            got = crit.loss_ref_score(ref, idx, num_boxes)["loss_ref_score"].item()
            self.assertAlmostEqual(got, per * B * Q / num_boxes, places=5)

    def test_empty_batch_defined_and_backward(self):
        """全无目标 batch: 标签全负, 损失有定义且可反传 (不制造正目标)。"""
        crit = self._crit()
        ref = torch.zeros(B, Q, requires_grad=True)
        idx = [(torch.zeros(0, dtype=torch.int64), ) * 2] * B
        l = crit.loss_ref_score(ref, idx, 1.0)["loss_ref_score"]
        self.assertTrue(torch.isfinite(l))
        l.backward()
        self.assertTrue(torch.isfinite(ref.grad).all() and ref.grad.abs().sum() > 0)

    def test_enters_total_loss_with_coef_and_suffixes(self):
        out, tg = self._outputs()
        torch.manual_seed(0)
        tot1, ld1, _ = self._crit(coef=1.0)(out, tg)
        torch.manual_seed(0)
        tot2, ld2, _ = self._crit(coef=2.0)(out, tg)
        ref_keys = sorted(k for k in ld1 if k.startswith("loss_ref_score"))
        self.assertEqual(len(ref_keys), L)  # 六层各一项
        self.assertIn("loss_ref_score", ref_keys)  # 最后一层无后缀
        self.assertIn("loss_ref_score_0", ref_keys)  # 辅助层带层号
        # 逐项加权求和 (不取平均): 系数翻倍 -> 总损失正好多出一份评分损失之和
        delta = sum(ld1[k] for k in ref_keys).item()
        self.assertAlmostEqual((tot2 - tot1).item(), delta, places=4)
        self.assertGreater(delta, 0.0)
        # interm 分支不加评分损失
        self.assertNotIn("loss_ref_score_interm", ld1)

    def test_delta_equals_weighted_sum_with_rng_restored(self):
        """与 smoke 脚本同一口径的数值断言: 恢复同一份 RNG 状态后两次调用 criterion,
        total 的差值必须等于 "各层评分损失 × 权重" 之和 (容差内), 其余损失项逐项不变。"""
        out, tg = self._outputs()
        crit = self._crit(coef=1.5)
        state = torch.get_rng_state()

        torch.set_rng_state(state)
        tot1, ld1, _ = crit(out, tg)
        ref_keys = sorted(k for k in ld1 if k.startswith("loss_ref_score"))
        expect = sum(ld1[k].item() for k in ref_keys) * 1.5

        crit.weight_dict["loss_ref_score"] = 0.0
        torch.set_rng_state(state)
        tot0, ld0, _ = crit(out, tg)

        # RNG 恢复生效的旁证: 其余损失项必须逐项相同
        for k in ld1:
            if not k.startswith("loss_ref_score"):
                self.assertAlmostEqual(ld1[k].item(), ld0[k].item(), places=6, msg=k)
        delta = tot1.item() - tot0.item()
        tol = max(1e-4, abs(expect) * 1e-4)
        self.assertAlmostEqual(delta, expect, delta=tol)
        self.assertGreater(expect, 0.0)

    def test_total_loss_backward_reaches_ref_logits(self):
        out, tg = self._outputs()
        tot, _, _ = self._crit()(out, tg)
        tot.backward()
        self.assertTrue(out["pred_ref_logits"].grad.abs().sum() > 0)
        self.assertTrue(out["aux_outputs"][0]["pred_ref_logits"].grad.abs().sum() > 0)

    def test_off_means_no_ref_terms_and_same_total(self):
        """评分关闭 (输出里没有 pred_ref_logits): 没有评分损失项, 总损失与原来一致。"""
        out, tg = self._outputs(with_ref=True)
        bare = {k: v for k, v in out.items() if k != "pred_ref_logits"}
        bare["aux_outputs"] = [{
            k: v
            for k, v in a.items() if k != "pred_ref_logits"
        } for a in out["aux_outputs"]]
        torch.manual_seed(0)
        tot_off, ld_off, _ = self._crit()(bare, tg)
        self.assertFalse([k for k in ld_off if "ref_score" in k])
        torch.manual_seed(0)
        tot_on, ld_on, _ = self._crit()(out, tg)
        ref_sum = sum(ld_on[k] for k in ld_on if k.startswith("loss_ref_score")).item()
        self.assertAlmostEqual((tot_on - tot_off).item(), ref_sum, places=4)

    def test_missing_ref_logits_raises(self):
        """评分开启却某层缺字段 -> 报错, 不解释成"这一层不用训练"。"""
        crit = self._crit()
        out, tg = self._outputs()
        out["aux_outputs"][0].pop("pred_ref_logits")
        with self.assertRaises(ValueError):
            crit(out, tg)

    def test_dn_adds_no_ref_loss(self):
        out, tg = self._outputs()
        meta = {
            "num_denoising_queries": 4,
            "num_denoising_groups": 2,
            "max_num_target": 1,
            "num_target_list": [1, 1]
        }
        out["dn_outputs"] = {
            "pred_logits": torch.randn(L, B, 4, 256),
            "pred_boxes": torch.rand(L, B, 4, 4) * 0.4 + 0.3,
            "layer_meta": [meta] * L
        }
        crit = self._crit()
        crit.weight_dict.update({"loss_ce_dn": 1.0, "loss_bbox_dn": 1.0, "loss_giou_dn": 1.0})
        _, ld, _ = crit(out, tg)
        self.assertEqual(len([k for k in ld if k.startswith("loss_ref_score")]), L)
        self.assertFalse([k for k in ld if "ref_score" in k and "_dn" in k])


@unittest.skipIf(_ERR is not None, f"criterion/evaluate 导入失败 (需服务器环境): {_ERR}")
class TestPostprocessRefScore(unittest.TestCase):
    H4 = W4 = 4

    def _outputs(self, ref_logits=None):
        tok = _tokenized()
        # token 分数故意让查询 0 最高, 新评分让另一个查询最高, 以便区分两条路径
        logits = torch.full((L, B, Q, 256), -10.0)
        logits[:, :, 0, 1:3] = 10.0
        embeds = torch.randn(L, B, Q, 3)
        out = {
            "hed_all_logits": logits,
            "hed_all_boxes": torch.rand(L, B, Q, 4) * 0.4 + 0.3,
            "hed_mask_embeds": embeds,
            "hed_pix": torch.randn(B, 3, self.H4, self.W4),
            "pred_masks": torch.randn(B, Q, self.H4, self.W4),
            "tokenized": tok,
            "text_token_mask": tok["attention_mask"].bool(),
            "pad_mask": torch.zeros(B, self.H4 * 4, self.W4 * 4, dtype=torch.bool)
        }
        if ref_logits is not None:
            out["hed_all_ref_logits"] = ref_logits
        return out

    def _targets(self):
        return [{
            "size": torch.tensor([self.H4 * 4, self.W4 * 4]),
            "orig_size": torch.tensor([self.H4 * 4, self.W4 * 4]),
            "orig_mask": torch.zeros(self.H4 * 4, self.W4 * 4, dtype=torch.bool),
            "orig_boxes": torch.zeros(0, 4)
        } for _ in range(B)]

    def test_new_score_decides_q_star(self):
        ref = torch.full((L, B, Q), -5.0)
        ref[:, 0, 3] = 5.0  # 第 0 张图: 新评分选查询 3
        ref[:, 1, 2] = 5.0  # 第 1 张图: 新评分选查询 2
        out, tg = self._outputs(ref), self._targets()
        res = postprocess_hed(out, tg, use_ref_score=True)
        self.assertEqual([r["queries"][0] for r in res], [3, 2])
        self.assertEqual([r["score_type"] for r in res], ["ref_score"] * B)
        # 关掉新评分 -> 回到 token 均分, 选查询 0
        res_old = postprocess_hed(out, tg, use_ref_score=False)
        self.assertEqual([r["queries"][0] for r in res_old], [0, 0])
        self.assertEqual(res_old[0]["score_type"], "token")
        self.assertNotIn("token_scores", res_old[0])

    def test_scores_are_the_ones_used_and_token_kept_as_diagnostic(self):
        ref = torch.full((L, B, Q), -5.0)
        ref[:, :, 1] = 1.0
        res = postprocess_hed(self._outputs(ref), self._targets(), use_ref_score=True)
        expect = ref.float().sigmoid().mean(0)[0, 1].item()
        self.assertAlmostEqual(res[0]["scores"][0], expect, places=6)  # scores = 实际选择依据
        self.assertIn("token_scores", res[0])  # 旧分数仅作诊断
        self.assertNotAlmostEqual(res[0]["token_scores"][0], expect, places=3)

    def test_changing_ref_logits_changes_selection(self):
        base = torch.full((L, B, Q), -5.0)
        base[:, :, 1] = 5.0
        r1 = postprocess_hed(self._outputs(base), self._targets(), use_ref_score=True)
        moved = base.clone()
        moved[:, :, 1] = -5.0
        moved[:, :, 4] = 5.0
        r2 = postprocess_hed(self._outputs(moved), self._targets(), use_ref_score=True)
        self.assertEqual([r["queries"][0] for r in r1], [1, 1])
        self.assertEqual([r["queries"][0] for r in r2], [4, 4])

    def test_mask_uses_same_query_index_across_layers(self):
        """q* 选定后, 六层掩码必须取同一个查询索引, 且是"逐层 sigmoid 再平均概率"
        (不是先平均 mask embedding, 也不是 logit 平均后 sigmoid)。"""
        from rrs.evaluate import _restore_mask
        ref = torch.full((L, B, Q), -5.0)
        ref[:, :, 2] = 5.0
        out = self._outputs(ref)
        res = postprocess_hed(out, self._targets(), use_ref_score=True, mask_mode="avg")
        q = res[0]["queries"][0]
        self.assertEqual(q, 2)
        hw = (self.H4 * 4, self.W4 * 4)

        def restore(m):  # m: (H4,W4) logits -> (h0,w0) logits, 与 _restore_mask 同一路径
            return _restore_mask(m[None, None].float(), hw, hw[0], hw[1], hw[0], hw[1])[0]

        layer_logits = [
            torch.einsum("c,chw->hw", out["hed_mask_embeds"][l, 0, q], out["hed_pix"][0])
            for l in range(L)
        ]
        want = torch.stack([restore(m).sigmoid() for m in layer_logits]).mean(0) > 0.5
        self.assertTrue(torch.equal(res[0]["mask"], want))
        # 概率平均不等于"先平均 embedding 再算掩码": 两者的概率图应有差异
        avg_embed = out["hed_mask_embeds"][:, 0, q].mean(0)
        prob_avg = torch.stack([restore(m).sigmoid() for m in layer_logits]).mean(0)
        prob_embed = restore(torch.einsum("c,chw->hw", avg_embed, out["hed_pix"][0])).sigmoid()
        self.assertGreater((prob_avg - prob_embed).abs().max().item(), 1e-6)
        # mask_mode="last" 仍走最后一层全查询掩码的老路径, q* 不变
        res_last = postprocess_hed(out, self._targets(), use_ref_score=True, mask_mode="last")
        self.assertEqual(res_last[0]["queries"][0], 2)
        want_last = restore(out["pred_masks"][0, q]).sigmoid() > 0.5
        self.assertTrue(torch.equal(res_last[0]["mask"], want_last))

    def test_missing_ref_logits_raises(self):
        with self.assertRaises(ValueError):
            postprocess_hed(self._outputs(None), self._targets(), use_ref_score=True)

    def test_multi_mode_still_rejected(self):
        ref = torch.full((L, B, Q), 0.0)
        with self.assertRaises(NotImplementedError):
            postprocess_hed(self._outputs(ref), self._targets(), mode="multi", use_ref_score=True)


class _StubVSSMBlock(nn.Module):
    """桩: 只保留 "输入输出同形状且可导" 这一点, 替换真实 SS2D 以便在 CPU 上验证接线与几何。
    绝不代表真实 selective scan 的数值; 真实算子见 tools/smoke_mm_score.py。"""

    def __init__(self, embed_dim=64, **kw):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        return self.conv(x)


class TestMMSCopEPixelDecoder(unittest.TestCase):

    def _build(self, **kw):
        with patch("rrs.mmscope_pixel_decoder.VSSMBlock", _StubVSSMBlock):
            from rrs.mmscope_pixel_decoder import MMSCopEPixelDecoder
            return MMSCopEPixelDecoder(c4_channels=96, enc_dim=32, out_dim=16, channels=32,
                                       feat_proj_dim=32, gn_groups=8, scan_fp32=False, **kw)

    def _feats(self, h4, w4):
        c4 = torch.randn(1, 96, h4, w4, requires_grad=True)
        h8, w8 = (h4 + 1) // 2, (w4 + 1) // 2
        enc = [
            torch.randn(1, 32, h8, w8, requires_grad=True),
            torch.randn(1, 32, (h8 + 1) // 2, (w8 + 1) // 2, requires_grad=True),
            torch.randn(1, 32, (h8 + 3) // 4, (w8 + 3) // 4, requires_grad=True)
        ]
        return c4, enc

    def test_output_aligns_with_c4_on_nonsquare_nondivisible(self):
        dec = self._build()
        # 非方形且 1/8 尺寸不是 4 的倍数 (对应 800x1333 的真实情况)
        for h4, w4 in ((200, 334), (26, 42), (13, 7), (8, 8)):
            c4, enc = self._feats(h4, w4)
            pix = dec(c4, enc)
            self.assertEqual(pix.shape, (1, 16, h4, w4), f"{h4}x{w4}")

    def test_batch_sizes(self):
        dec = self._build()
        for b in (1, 2):
            c4 = torch.randn(b, 96, 26, 42)
            enc = [torch.randn(b, 32, 13, 21), torch.randn(b, 32, 7, 11), torch.randn(b, 32, 4, 6)]
            self.assertEqual(dec(c4, enc).shape, (b, 16, 26, 42))

    def test_gradients_reach_c4_and_enc_maps(self):
        dec = self._build()
        c4, enc = self._feats(26, 42)
        dec(c4, enc).sum().backward()
        self.assertTrue(c4.grad is not None and c4.grad.abs().sum() > 0)
        for i, e in enumerate(enc):
            self.assertTrue(e.grad is not None and e.grad.abs().sum() > 0, f"enc_maps[{i}]")

    def test_no_detach_or_no_grad_in_module(self):
        """静态检查: 新支路不得 detach 或 no_grad 掉上游特征。"""
        src = open(os.path.join(ROOT, "rrs", "mmscope_pixel_decoder.py"), encoding="utf-8").read()
        self.assertNotIn(".detach()", src)
        self.assertNotIn("no_grad", src)

    def test_requires_real_ops(self):
        """不打桩时必须用真实 VSSMBlock; 缺算子环境下报错而不是静默退化。"""
        import rrs.mmscope_pixel_decoder as mpd  # 导入时把 third_party/SegMAN 加入 sys.path
        from segman.mmscope_core import require_scan_ops
        self.assertIs(mpd.MMSCopEPixelDecoder.__mro__[0], mpd.MMSCopEPixelDecoder)
        self.assertEqual(mpd.VSSMBlock.__module__, "segman.mmscope_core")
        try:
            require_scan_ops()
        except ImportError as e:
            self.assertIn("selective_scan", str(e))
        else:
            self.skipTest("本环境已装好真实算子, 由 smoke_mm_score.py 覆盖")


class TestGradReportLogic(unittest.TestCase):
    """smoke 脚本里 _collect_grad_report 的分类与失败判定 (用小模型造出各种梯度状态)。
    不需要 CUDA, 也不需要 MM 依赖。"""

    @staticmethod
    def _load_fn(finish=False):
        """按 AST 取出 smoke 脚本里的 BASE_GROUPS 与 _collect_grad_report 原代码,
        避免导入 smoke 脚本本体 (它会 import rrs.model / 切换 cwd)。
        测的是**生产代码本身**, 不是复制品。"""
        import ast
        path = os.path.join(ROOT, "tools", "smoke_mm_score.py")
        tree = ast.parse(open(path, encoding="utf-8").read())
        ns = {"torch": torch}
        wanted = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "BASE_GROUPS" for t in node.targets):
                wanted.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in (
                ("_collect_grad_report", "_finish_accumulation") if finish else
                ("_collect_grad_report", )):
                wanted.append(node)
        assert len(wanted) == (3 if finish else 2), "未能取到全部诊断定义"
        exec(compile(ast.Module(body=wanted, type_ignores=[]), path, "exec"), ns)
        return ns["_finish_accumulation" if finish else "_collect_grad_report"]

    def _report(self, model, cfg):
        return self._load_fn()(model, cfg)

    def _model(self):
        m = nn.Module()
        m.ref_score_head = RefScoreHead(4)
        m.mask_head = nn.Linear(4, 4)
        m.pixel_decoder = nn.Linear(4, 4)
        m.c4_norm = nn.LayerNorm(4)
        return m

    CFG = {"use_ref_score": True, "use_mmscope": False, "use_dn": False}

    def test_nonfinite_grad_fails(self):
        m = self._model()
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        m.ref_score_head.net[0].weight.grad = torch.full_like(m.ref_score_head.net[0].weight,
                                                              float("nan"))
        r = self._report(m, self.CFG)
        self.assertFalse(r["ok"])
        self.assertEqual(r["ref_score_head"]["n_nonfinite"], 1)
        self.assertIn("NaN/Inf 梯度", r["ref_score_head"]["fail_reasons"])

    def test_one_good_param_does_not_mask_nan(self):
        """关键回归: 模块里有一个参数正常、另一个 NaN 时不能通过。"""
        m = self._model()
        m.ref_score_head.net[0].weight.grad = torch.ones_like(m.ref_score_head.net[0].weight)
        m.ref_score_head.net[0].bias.grad = torch.full_like(m.ref_score_head.net[0].bias,
                                                            float("inf"))
        m.ref_score_head.net[2].weight.grad = torch.ones_like(m.ref_score_head.net[2].weight)
        m.ref_score_head.net[2].bias.grad = torch.ones_like(m.ref_score_head.net[2].bias)
        r = self._report(m, self.CFG)
        self.assertGreater(r["ref_score_head"]["with_grad"], 0)  # 确实有正常的参数
        self.assertFalse(r["ref_score_head"]["ok"])  # 但仍必须失败
        self.assertFalse(r["ok"])

    def test_new_module_missing_grad_fails(self):
        """评分头出现"可训练但无梯度"必须失败 (断路)。"""
        m = self._model()
        for n, p in m.named_parameters():
            p.grad = None if n.startswith("ref_score_head.net.2") else torch.ones_like(p)
        r = self._report(m, self.CFG)
        self.assertEqual(r["ref_score_head"]["n_none_unused"], 2)  # weight + bias
        self.assertFalse(r["ok"])

    def test_all_zero_is_not_a_failure(self):
        """单 batch 下全零梯度只记录, 不判失败。"""
        m = self._model()
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        m.mask_head.bias.grad = torch.zeros_like(m.mask_head.bias)
        r = self._report(m, self.CFG)
        self.assertEqual(r["mask_head"]["n_all_zero"], 1)
        self.assertTrue(r["mask_head"]["ok"])

    def test_frozen_params_counted_as_frozen_not_unused(self):
        """冻结的 gd.bert 没有梯度属于预期: 归到 none_frozen, 不算断路, 整体仍 PASS。"""
        m = self._model()
        m.gd = nn.Module()
        m.gd.bert = nn.Linear(4, 4)  # 对应分组前缀 "gd.bert."
        for p in m.gd.bert.parameters():
            p.requires_grad_(False)
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        r = self._report(m, self.CFG)
        self.assertEqual(r["bert(frozen)"]["n_none_frozen"], 2)  # weight + bias
        self.assertEqual(r["bert(frozen)"]["n_none_unused"], 0)
        self.assertTrue(r["bert(frozen)"]["ok"])
        self.assertTrue(r["ok"])

    def test_module_with_no_grad_at_all_fails(self):
        m = self._model()
        for n, p in m.named_parameters():
            p.grad = None if n.startswith("pixel_decoder") else torch.ones_like(p)
        r = self._report(m, self.CFG)
        self.assertFalse(r["pixel_decoder"]["ok"])
        self.assertFalse(r["ok"])


class TestSmokeUpdateDiagnostics(unittest.TestCase):
    """服务器回归: 保留异常来源、阻止 FP32 非有限更新、统计真实更新。"""

    CFG = {"use_ref_score": False, "use_mmscope": False, "use_dn": False, "clip_max_norm": 1.0}

    def setUp(self):
        self.model = nn.Module()
        self.model.pixel_decoder = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.finish = TestGradReportLogic._load_fn(finish=True)

    def _run(self):
        return self.finish(self.model, self.optimizer, self.scaler, self.CFG)

    def _assert_unchanged(self):
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, self.before[n]), n)

    def test_finite_gradients_really_update(self):
        r = self._run()
        self.assertTrue(r["update_ok"])
        self.assertTrue(r["optimizer_step_called"])
        for n, p in self.model.named_parameters():
            self.assertFalse(torch.equal(p, self.before[n]), n)

    def test_nan_preserves_finite_sibling_and_blocks_fp32_update(self):
        self.model.pixel_decoder.weight.grad.fill_(float("nan"))
        r = self._run()
        self.assertEqual(r["grad_report"]["pixel_decoder"]["with_grad"], 1)
        self.assertEqual(r["post_clip_nonfinite"], ["pixel_decoder.weight"])
        self.assertFalse(r["clip_applied"])
        self.assertFalse(r["optimizer_step_called"])
        self.assertFalse(r["update_ok"])
        self._assert_unchanged()

    def test_finite_elements_with_overflowing_norm_block_update(self):
        for p in self.model.parameters():
            p.grad.fill_(1e30)
        r = self._run()
        self.assertEqual(r["grad_report"]["all_parameters"]["n_nonfinite"], 0)
        self.assertIn("clip_error", r)
        self.assertEqual(r["post_clip_nonfinite"], [])
        self.assertFalse(r["optimizer_step_called"])
        self.assertFalse(r["update_ok"])
        self._assert_unchanged()

    def test_scaler_skip_is_not_counted_as_optimizer_update(self):
        # 模拟 scaler 跳过更新, 只验证计数语义; 真实 CUDA AMP 由 smoke 覆盖。
        class SkippingScaler:
            scale = 65536.0

            def is_enabled(self):
                return True

            def get_scale(self):
                return self.scale

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                pass

            def update(self):
                self.scale /= 2

        self.scaler = SkippingScaler()
        self.model.pixel_decoder.weight.grad.fill_(float("inf"))
        r = self._run()
        self.assertFalse(r["optimizer_step_called"])
        self.assertTrue(r["scaler"]["step_skipped_due_to_nonfinite_grad"])
        self.assertEqual(r["scaler"]["scale_after"], 32768.0)
        self.assertFalse(r["update_ok"])
        self._assert_unchanged()

    def test_nonfinite_outside_named_groups_is_caught(self):
        self.model.extra = nn.Parameter(torch.ones(1))
        self.model.extra.grad = torch.full_like(self.model.extra, float("inf"))
        r = TestGradReportLogic._load_fn()(self.model, self.CFG)
        self.assertEqual(r["all_parameters"]["nonfinite"], ["extra"])
        self.assertFalse(r["ok"])


class _StubModel(nn.Module):
    """只为测试 init_model_weights 的键规则, 不是真模型。"""

    def __init__(self, mm=False, score=False, dn=False):
        super().__init__()
        self.use_mmscope = mm
        self.mask_head = nn.Linear(4, 4)
        if mm:
            self.pixel_decoder = nn.ModuleDict({"vssm": nn.Linear(8, 8)})
        else:
            self.pixel_decoder = nn.ModuleDict({"lateral4": nn.Linear(4, 4)})
        self.dn_generator = nn.Embedding(1, 4) if dn else None
        self.ref_score_head = RefScoreHead(4) if score else None


@unittest.skipIf(_ERR is not None, f"rrs.model 导入失败 (需服务器环境): {_ERR}")
class TestInitModelWeights(unittest.TestCase):

    def test_new_score_head_is_new_and_rest_kept(self):
        old = _StubModel()
        new = _StubModel(score=True)
        new.mask_head.load_state_dict(old.mask_head.state_dict())
        info = init_model_weights(new, old.state_dict(), {"use_hed": True})
        self.assertTrue(all(k.startswith(REF_SCORE_PREFIX) for k in info["new"]))
        self.assertTrue(info["new"])
        self.assertFalse(info["dropped"])
        self.assertFalse(info["pixel_decoder_replaced"])
        self.assertGreater(info["kept_count"], 0)

    def test_pixel_decoder_replaced_when_switching_to_mm(self):
        old = _StubModel(mm=False)
        new = _StubModel(mm=True, score=True)
        info = init_model_weights(new, old.state_dict(), {"use_hed": True, "use_mmscope": False})
        self.assertTrue(info["pixel_decoder_replaced"])
        self.assertTrue(all(k.startswith("pixel_decoder.") for k in info["dropped"]))
        self.assertTrue(any(k.startswith("pixel_decoder.") for k in info["new"]))
        self.assertTrue(any(k.startswith(REF_SCORE_PREFIX) for k in info["new"]))

    def test_same_pixel_decoder_type_keeps_weights(self):
        old, new = _StubModel(mm=True), _StubModel(mm=True, score=True)
        info = init_model_weights(new, old.state_dict(), {"use_hed": True, "use_mmscope": True})
        self.assertFalse(info["pixel_decoder_replaced"])
        self.assertFalse([k for k in info["new"] if k.startswith("pixel_decoder.")])

    # ---- 白名单按 源开关 × 目标开关 收紧 (第二轮修正点 6) ----
    def test_score_both_enabled_requires_complete_weights(self):
        """源、目标都启用评分头时, 评分头权重必须完整; 缺键必须报错, 不能当成"正常新增"。"""
        old, new = _StubModel(score=True), _StubModel(score=True)
        sd = old.state_dict()
        key = next(k for k in sd if k.startswith(REF_SCORE_PREFIX))
        broken = {k: v for k, v in sd.items() if k != key}
        with self.assertRaises(RuntimeError) as cm:
            init_model_weights(new, broken, {"use_hed": True, "use_ref_score": True})
        self.assertIn(key, str(cm.exception))
        # 完整时必须成功, 且评分头不出现在 new 里 (是保留的旧权重, 不是新建)
        info = init_model_weights(_StubModel(score=True), sd, {
            "use_hed": True,
            "use_ref_score": True
        })
        self.assertFalse([k for k in info["new"] if k.startswith(REF_SCORE_PREFIX)])
        self.assertIn("必须完整匹配", info["decisions"]["ref_score_head"])

    def test_score_source_off_target_on_allows_new(self):
        old, new = _StubModel(score=False), _StubModel(score=True)
        info = init_model_weights(new, old.state_dict(), {"use_hed": True, "use_ref_score": False})
        self.assertTrue([k for k in info["new"] if k.startswith(REF_SCORE_PREFIX)])
        self.assertIn("整体新建", info["decisions"]["ref_score_head"])

    def test_score_source_on_target_off_drops(self):
        old, new = _StubModel(score=True), _StubModel(score=False)
        info = init_model_weights(new, old.state_dict(), {"use_hed": True, "use_ref_score": True})
        self.assertTrue(all(k.startswith(REF_SCORE_PREFIX) for k in info["dropped"]))
        self.assertTrue(info["dropped"])
        self.assertIn("丢弃", info["decisions"]["ref_score_head"])

    def test_same_pixel_decoder_type_missing_key_raises(self):
        """像素支路类型相同时缺键也必须报错 (不因同属性名就放行)。"""
        old, new = _StubModel(mm=True), _StubModel(mm=True)
        sd = old.state_dict()
        key = next(k for k in sd if k.startswith("pixel_decoder."))
        with self.assertRaises(RuntimeError) as cm:
            init_model_weights(new, {
                k: v
                for k, v in sd.items() if k != key
            }, {
                "use_hed": True,
                "use_mmscope": True
            })
        self.assertIn(key, str(cm.exception))

    def test_source_flags_reported(self):
        info = init_model_weights(_StubModel(mm=True, score=True),
                                  _StubModel().state_dict(), {"use_hed": True})
        self.assertEqual(info["source_flags"], {"use_mmscope": False, "use_ref_score": False})

    def test_self_contradictory_checkpoint_raises(self):
        """cfg 说 use_ref_score=False 但权重里带评分头键 -> 报错, 不能按"源无"放行,
        否则目标开启评分头时会把真权重当成"整体新建"丢掉。"""
        sd = _StubModel(score=True).state_dict()
        with self.assertRaises(RuntimeError) as cm:
            init_model_weights(_StubModel(score=True), sd, {
                "use_hed": True,
                "use_ref_score": False
            })
        self.assertIn("自相矛盾", str(cm.exception))

    def test_shape_mismatch_raises(self):
        old = _StubModel()
        sd = old.state_dict()
        sd["mask_head.weight"] = torch.randn(5, 5)
        with self.assertRaises(RuntimeError):
            init_model_weights(_StubModel(score=True), sd, {"use_hed": True})

    def test_unexpected_key_raises(self):
        sd = _StubModel().state_dict()
        sd["something.unknown"] = torch.randn(2)
        with self.assertRaises(RuntimeError):
            init_model_weights(_StubModel(score=True), sd, {"use_hed": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
