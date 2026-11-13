"""论文关键逻辑：独立递推参考、逐组候选差异、恢复与概率融合顺序。"""
import unittest

import torch
import torch.nn.functional as F

from rrs.config import load_cfg
from rrs.evaluate import postprocess_hed, _restore_mask
from rrs.mcpe import MCPEPixelDecoder
from rrs.model_mode import resolve_model_mode
from rrs.selective_scan import affine_scan, SS2D


class TestSelectiveScan(unittest.TestCase):

    def test_states_and_gradients_against_sequential_recurrence(self):
        torch.manual_seed(24)
        # 131 crosses two chunk boundaries; coefficients include strong decay.
        a = torch.rand(2, 3, 131, dtype=torch.float64, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        actual = affine_scan(a, b)
        state = torch.zeros_like(b[..., 0])
        values = []
        for t in range(a.shape[-1]):
            state = a[..., t] * state + b[..., t]
            values.append(state)
        expected = torch.stack(values, -1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        weights = torch.randn_like(actual)
        grad_a, grad_b = torch.autograd.grad((actual * weights).sum(), (a, b))
        ref_a, ref_b = torch.autograd.grad((expected * weights).sum(), (a, b))
        torch.testing.assert_close(grad_a, ref_a, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(grad_b, ref_b, atol=1e-12, rtol=1e-12)

    def test_scan_is_input_dependent_and_all_core_parameters_train(self):
        torch.manual_seed(8)
        scan = SS2D(12)
        x = torch.randn(2, 12, 3, 7, requires_grad=True)
        y = scan(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        for name, parameter in scan.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ('A_logs', 'Ds', 'x_proj_weight', 'dt_projs_weight', 'dt_projs_bias'):
            self.assertGreater(getattr(scan, name).grad.abs().sum().item(), 0, name)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertFalse(torch.allclose(scan(x.detach()), scan(x.detach() + 1)))

    def test_mcpe_real_scan_odd_shapes_and_gradient_flow(self):
        torch.manual_seed(7)
        decoder = MCPEPixelDecoder(12, enc_dim=16, out_dim=16, channels=8, feat_proj_dim=16,
                                   gn_groups=2)
        for batch, h, w in ((1, 13, 21), (2, 7, 9), (1, 2, 2)):
            decoder.zero_grad(set_to_none=True)
            shallow = torch.randn(batch, 12, h, w, requires_grad=True)
            maps = [
                torch.randn(batch, 16, max(1, (h + factor - 1) // factor),
                            max(1, (w + factor - 1) // factor), requires_grad=True)
                for factor in (2, 4, 8)
            ]
            output = decoder(shallow, maps)
            self.assertEqual(output.shape, (batch, 16, h, w))
            output.square().mean().backward()
            for feature in (shallow, *maps):
                self.assertIsNotNone(feature.grad)
                self.assertTrue(torch.isfinite(feature.grad).all())
                self.assertGreater(feature.grad.abs().sum().item(), 0)
            for name, parameter in decoder.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class TestPaperInference(unittest.TestCase):

    def fixture(self):
        # Six different best queries. A common-index implementation cannot pass.
        ref = torch.full((6, 1, 6), -5.)
        for group in range(6):
            ref[group, 0, group] = 5.
        embed = torch.zeros(6, 1, 6, 1)
        for group in range(6):
            embed[group, 0, group, 0] = [-9., 1., 1., 1., 1., 1.][group]
        logits = torch.full((6, 1, 6, 256), -10.)
        logits[:, :, 0, 1] = 5.
        boxes = torch.rand(6, 1, 6, 4)
        out = dict(
            hed_all_logits=logits, hed_all_boxes=boxes, hed_all_ref_logits=ref,
            hed_mask_embeds=embed, hed_pix=torch.ones(1, 1, 2, 2),
            pad_mask=torch.zeros(1, 6, 8, dtype=torch.bool),
            tokenized=dict(input_ids=torch.tensor([[101, 2000, 2001, 1012, 102, 0]]),
                           attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]])))
        target = dict(size=torch.tensor([5, 7]), orig_size=torch.tensor([9, 11]))
        return out, [target]

    def test_independent_selection_and_probabilities_not_logits(self):
        out, targets = self.fixture()
        result = postprocess_hed(out, targets, use_ref_score=True)[0]
        self.assertEqual(result['queries'], list(range(6)))
        expected = torch.tensor([-9., 1., 1., 1., 1., 1.]).sigmoid().mean()
        torch.testing.assert_close(result['prob_mask'], torch.full((9, 11), expected))
        self.assertTrue(result['mask'].all())
        # Averaging these logits would yield -2/3, causing a false negative.
        self.assertLess(torch.tensor([-9., 1., 1., 1., 1., 1.]).mean().sigmoid(), .5)
        selected = out['hed_all_boxes'][torch.arange(6), 0, torch.arange(6)]
        self.assertEqual(len(result['selected_boxes']), 6)
        self.assertEqual(len(result['boxes']), 1)
        expected_cx = selected[:, 0].mean() * 11
        self.assertAlmostEqual((result['boxes'][0][0] + result['boxes'][0][2]) / 2,
                               expected_cx.item(), places=5)

    def test_restoration_happens_before_sigmoid(self):
        out, targets = self.fixture()
        out['hed_pix'] = torch.tensor([[[[-8., 1.], [4., -2.]]]])
        result = postprocess_hed(out, targets, use_ref_score=True)[0]
        expected = []
        wrong_order = []
        for group in range(6):
            logit = out['hed_mask_embeds'][group, 0, group, 0] * out['hed_pix']
            expected.append(_restore_mask(logit, (6, 8), 5, 7, 9, 11)[0].sigmoid())
            wrong_order.append(_restore_mask(logit.sigmoid(), (6, 8), 5, 7, 9, 11)[0])
        torch.testing.assert_close(result['prob_mask'], torch.stack(expected).mean(0))
        self.assertGreater((result['prob_mask'] - torch.stack(wrong_order).mean(0)).abs().max(),
                           .01)

    def test_tm_uses_each_groups_max_eligible_token(self):
        out, targets = self.fixture()
        out['hed_all_logits'].fill_(-10.)
        for group in range(6):
            # Two moderate tokens for q0; one stronger token for selected q.
            out['hed_all_logits'][group, 0, :, 0] = 100.  # CLS must be ignored
            out['hed_all_logits'][group, 0, 0, 1:3] = 1.
            out['hed_all_logits'][group, 0, group, 1] = 3.
        result = postprocess_hed(out, targets, use_ref_score=False)[0]
        self.assertEqual(result['queries'], list(range(6)))
        self.assertNotIn('token_scores', result)

    def test_missing_scores_and_invalid_mode_raise(self):
        out, targets = self.fixture()
        out.pop('hed_all_ref_logits')
        with self.assertRaises(ValueError):
            postprocess_hed(out, targets, use_ref_score=True)
        with self.assertRaises(NotImplementedError):
            postprocess_hed(out, targets, mode='multi')

    def test_last_group_ablation(self):
        out, targets = self.fixture()
        result = postprocess_hed(out, targets, mask_mode='last', use_ref_score=True)[0]
        torch.testing.assert_close(result['prob_mask'],
                                   torch.full((9, 11),
                                              torch.tensor(1.).sigmoid()))


class TestPaperConfig(unittest.TestCase):

    def test_full_model_settings_and_alias_overrides(self):
        cfg = resolve_model_mode(load_cfg('configs/mrenet_rislad.py'))
        self.assertTrue(all(cfg[k] for k in ('use_hed', 'use_mmscope', 'use_ref_score', 'use_dn')))
        self.assertEqual((cfg['epochs'], cfg['lr_drop'], cfg['batch_size'], cfg['grad_accum']),
                         (50, 30, 4, 1))
        self.assertEqual(cfg['hflip'], 0.)
        for option in ('use_rca=False', 'use_ref_score=False'):
            resolved = resolve_model_mode(load_cfg('configs/mrenet_rislad.py', [option]),
                                          opts=[option])
            self.assertFalse(resolved['use_ref_score'])
        for option in ('mcpe.channels=96', 'mm.channels=96'):
            resolved = resolve_model_mode(load_cfg('configs/mrenet_rislad.py', [option]),
                                          opts=[option])
            self.assertEqual(resolved['mm']['channels'], 96)


if __name__ == '__main__':
    unittest.main(verbosity=2)
