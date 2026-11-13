"""离线合成数据整链路检查；随机初始化模型，不验证预训练效果或数据集精度。

python tools/smoke_synthetic.py
使用真实 Swin-T、GroundingDINO 图文交互层和六个解码层；缩小 BERT、
查询数、encoder 深度、MCPE 宽度及输入尺寸，仅用于可执行性和梯度验证。
"""
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel, BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rrs.config import load_cfg
from rrs.criterion import build_criterion
from rrs.evaluate import postprocess_for_model
from rrs.model import GroundingDINORefSeg, build_groundingdino
from rrs.model_mode import resolve_model_mode
from groundingdino.util.slconfig import SLConfig


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    random.seed(7)
    cfg = resolve_model_mode(load_cfg(ROOT / 'configs/mrenet_rislad.py'))
    cfg['num_points'] = 32
    with tempfile.TemporaryDirectory(prefix='mrenet_synthetic_') as temporary:
        vocab = [f'unused{i}' for i in range(1020)]
        for index, token in {
                0: '[PAD]',
                100: '[UNK]',
                101: '[CLS]',
                102: '[SEP]',
                103: '[MASK]',
                104: 'the',
                105: 'car',
                106: 'left',
                1012: '.',
                1013: '?'
        }.items():
            vocab[index] = token
        path = Path(temporary) / 'vocab.txt'
        path.write_text('\n'.join(vocab), encoding='utf-8')
        tokenizer = BertTokenizer(vocab_file=str(path), do_lower_case=True)
        bert = BertModel(
            BertConfig(vocab_size=1020, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                       intermediate_size=64))
        args = SLConfig.fromfile(str(ROOT / cfg['gdino_cfg']))
        args.enc_layers = 1
        args.dim_feedforward = 256
        args.num_queries = 8
        args.use_checkpoint = False
        args.use_transformer_ckpt = False
        with patch('groundingdino.util.get_tokenlizer.get_tokenlizer', return_value=tokenizer), \
             patch('groundingdino.util.get_tokenlizer.get_pretrained_language_model', return_value=bert):
            gdino = build_groundingdino(args)
        model = GroundingDINORefSeg(gdino, mask_dim=32, use_hed=True, use_dn=True, dn_cfg={
            **cfg['dn'], 'num_dn_queries': 2
        }, use_mmscope=True, mm_cfg=dict(channels=8, feat_proj_dim=16, gn_groups=2),
                                    use_ref_score=True)
        # Match the paper's frozen language encoder.
        for parameter in model.gd.bert.parameters():
            parameter.requires_grad_(False)
        image = torch.randn(3, 64, 96)
        mask = torch.zeros(1, 64, 96)
        mask[:, 22:42, 32:64] = 1
        targets = [
            dict(boxes=torch.tensor([[.5, .5, 1 / 3, 20 / 64]]), masks=mask,
                 size=torch.tensor([64, 96]), orig_size=torch.tensor([64, 96]))
        ]
        model.train()
        output = model([image], ['the car left.'], targets=targets)
        loss, components, _ = build_criterion(cfg)(output, targets)
        if not torch.isfinite(loss):
            raise AssertionError('合成训练损失非有限')
        loss.backward()
        for prefix in ('pixel_decoder.', 'ref_score_head.', 'dn_generator.',
                       'gd.transformer.decoder.'):
            gradients = [
                p.grad for name, p in model.named_parameters()
                if name.startswith(prefix) and p.requires_grad
            ]
            assert gradients and any(g is not None and g.abs().sum() > 0 for g in gradients), prefix
            assert all(g is None or torch.isfinite(g).all() for g in gradients), prefix
        assert len([k for k in components if k.startswith('loss_ref_score')]) == 6
        assert len([k for k in components if k.startswith('loss_ce_dn')]) == 6
        model.eval()
        with torch.no_grad():
            inference = model([image], ['the car left.'])
            result = postprocess_for_model(model, inference, targets)[0]
        assert 'dn_outputs' not in inference
        assert len(result['queries']) == 6
        assert result['mask'].shape == (64, 96)
        report = dict(passed=True, device='cpu', torch=torch.__version__,
                      training_loss=float(loss.detach()), loss_terms=len(components),
                      ordinary_queries=8, dn_queries=output['dn_outputs']['pred_boxes'].shape[2],
                      selected_queries=result['queries'], output_shape=list(result['mask'].shape))
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
