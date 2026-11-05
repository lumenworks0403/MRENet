"""
B0: GroundingDINO (官方实现, 原始串行解码器) + 普通 FPN 像素支路 + 掩码预测头。
B0_HED (无DN结构版): 在 B0 基础上增加 HED 并行解码器结构。
  - 第1层解码器共享处理（串行, 与 B0 完全相同的权重和逻辑）
  - 第2~6层解码器并行，各自独立从第1层输出处理，分支间互不影响
  - 推理时对6层的分类概率（各层分别 sigmoid 后）和预测框分别求均值
  - 掩码在推理时对选定查询的6层逐层累加（避免一次保存全部层全部查询的掩码）
  - use_hed=False 时行为与原 B0 完全一致
B0_HED+MM / +Score (2026-10-16 新增, 两个开关互相独立):
  - use_mmscope=True: 像素支路换成基于 SegMAN 官方 MMSCopE 的 MMSCopEPixelDecoder
    (rrs/mmscope_pixel_decoder.py), 接口仍是 pix = pixel_decoder(c4, enc_maps);
    关闭时完全不实例化、不导入 triton / selective_scan。
  - use_ref_score=True: 新增共享 RefScoreHead (rrs/ref_score_head.py), 对六层普通查询
    各输出 (B,Q) 标量 logits; 推理时按六层 sigmoid 均值选 top1 查询 (evaluate.py),
    训练时用各层已有的 Hungarian 匹配结果做监督 (criterion.py)。只支持 use_hed=True。
  - 两者都关 => 与 2026-10-13 的 HED / HED+DN 行为逐元素相同。

B0_HED+DN (use_hed=True, use_dn=True, 2026-10-13 新增, 只影响训练):
  - 训练时按官方 CdnQueryGenerator 由 GT 生成 DN 查询 (rrs/dn.py), 排在普通查询之前;
  - 后五个分支以概率 τ (dn.reinit_rate) 重新生成自己的 DN, 否则继承共享层 DN
    (rrs/hed_decoder.py::hed_parallel_decode_dn);
  - 预测头之后立即拆出 DN, 掩码/匹配/汇总只用普通查询, DN 损失见 criterion.py;
  - 推理不生成 DN, 输出与 HED-structure/no-DN 同一路径。
  - use_hed=False 且 use_dn=True (B0+DN) 明确报错, 不支持。

复用的官方代码 (third_party/GroundingDINO, commit 856dde2):
  - groundingdino/models/GroundingDINO/groundingdino.py
  - groundingdino/models/GroundingDINO/transformer.py
  - groundingdino/models/GroundingDINO/bertwarper.py
HED 算法参考:
  - tmp/hed_reference/grounding_dino_layers_HED.py (GroundingDinoTransformerDecoder_parallel_15_DNQueryRand)
  - FT-FSOD.pdf 第3.2节, 图2, 公式5-7
  官方仓库 commit c68cea00fac282ec526d59bca55f27b3203daf37 (official_main.zip 注释),
  根项目 FT-FSOD-main/mmdet/... , 非 NTIRE 变体。本地快照 SHA256:
    grounding_dino_layers_HED.py d16bfda9d8509cb13c2dbe0cece64bb998a139f154ef95348a88738a2598dcbc
    grounding_dino_HED.py        b4ae3302066379a7d08927badf98ffb770b6607cc50289a37812db9e631b832b
    grounding_dino_head_HED.py   a7354e855f1dc38d54f0a85d4d3a4d6096e774079cef42eae0ea3155dd92eafe
"""
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_GD_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party",
                        "GroundingDINO")
if _GD_ROOT not in sys.path:
    sys.path.insert(0, _GD_ROOT)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from groundingdino.models.GroundingDINO import ms_deform_attn as _msda
    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map, )
    from groundingdino.models.GroundingDINO.groundingdino import build_groundingdino
    from groundingdino.models.GroundingDINO.utils import (
        gen_encoder_output_proposals,
        gen_sineembed_for_position,
    )
    from groundingdino.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
    from groundingdino.util.slconfig import SLConfig

from .hed_decoder import hed_parallel_decode as _hed_parallel_decode
from .hed_decoder import hed_parallel_decode_dn as _hed_parallel_decode_dn
from .hed_decoder import sample_branch_dns as _sample_branch_dns
from .dn import CdnQueryGenerator
from .ref_score_head import RefScoreHead
import random as _py_random  # 官方 pre_decoder 用 Python random 决定分支是否重新生成 DN

# 只有这些 dn 配置项会传给生成器 / 解码器; 其余 (如 loss_coef) 由 criterion 使用。
DN_GENERATOR_KEYS = ("label_book_size", "label_noise_scale", "box_noise_scale", "num_dn_queries")
# 旧 checkpoint (B0 或 HED-no-DN) 加载到 HED+DN 模型时, 只允许缺这些键 (新建的 DN 参数)。
DN_STATE_PREFIX = "dn_generator."
# 2026-10-16: 新增模块的 state_dict 前缀。
#   REF_SCORE_PREFIX   use_ref_score=True 时新建的评分头 (旧 checkpoint 必然没有)。
#   PIXEL_DECODER_PREFIX 两条像素支路共用同一属性名, 但参数结构完全不同:
#       切换 use_mmscope 时不能当成"同名同形状即兼容", 只允许走 --init-from 显式重新初始化。
REF_SCORE_PREFIX = "ref_score_head."
PIXEL_DECODER_PREFIX = "pixel_decoder."
# 只有这些 mm 配置项会传给 MMSCopEPixelDecoder; 其余 (若新增) 需显式加进来。
MM_DECODER_KEYS = ("channels", "feat_proj_dim", "norm", "gn_groups", "scan_fp32", "drop_path")


def _patch_deform_attn_if_no_cuda_op():
    """官方 ms_deform_attn 在 CUDA 张量上无条件调用 _C 扩展。
    若扩展未编译 (例如 Windows 无 MSVC), 改走同文件内的纯 PyTorch 实现
    multi_scale_deformable_attn_pytorch (grid_sample, 数学上等价, 可反传)。"""
    if hasattr(_msda, "_C"):
        return False
    _msda.MultiScaleDeformableAttnFunction.apply = staticmethod(
        lambda value, shapes, level_start, loc, attn, im2col: _msda.
        multi_scale_deformable_attn_pytorch(value, shapes, loc, attn))
    return True


DEFORM_ATTN_PYTORCH_FALLBACK = _patch_deform_attn_if_no_cuda_op()


class FPNPixelDecoder(nn.Module):
    """普通 FPN (Lin et al. 2017): 1/32,1/16,1/8 取 Feature Enhancer 输出的已融合文本的特征 (256 维),
    1/4 取 Swin stage-0 输出 (未经过 encoder)。自顶向下逐级上采样相加, 输出 1/4 分辨率像素特征。"""

    def __init__(self, c4_channels, dim=256, out_dim=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(3)])  # 1/8, 1/16, 1/32
        self.lateral4 = nn.Conv2d(c4_channels, dim, 1)
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim),
                          nn.ReLU(inplace=True)) for _ in range(4)
        ])
        self.out = nn.Conv2d(dim, out_dim, 1)

    def forward(self, c4, enc_feats):
        # enc_feats: [1/8, 1/16, 1/32]
        x = self.smooth[3](self.lateral[2](enc_feats[2]))
        for i in (1, 0):
            lat = self.lateral[i](enc_feats[i])
            x = self.smooth[i + 1](lat + F.interpolate(x, size=lat.shape[-2:], mode="nearest"))
        lat = self.lateral4(c4)
        x = self.smooth[0](lat + F.interpolate(x, size=lat.shape[-2:], mode="nearest"))
        return self.out(x)


class MaskHead(nn.Module):
    """Mask2Former 式点积掩码头: mask = MLP(query) · pixel_feature。"""

    def __init__(self, dim=256, pix_dim=256):
        super().__init__()
        self.mask_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, pix_dim),
        )

    def forward(self, hs, pix):
        # AMP 下为 fp16; 损失与匹配中的点采样均转 fp32 计算
        return torch.einsum("bqc,bchw->bqhw", self.mask_embed(hs), pix)


class GroundingDINORefSeg(nn.Module):
    """B0 + 可选 HED 并行解码器 + 可选 DN + 可选 MMSCopE 像素支路 / RefScoreHead。

    use_hed=False: 行为与原 B0 完全相同，参数加载路径不变。
    use_hed=True : HED，第1层共享，第2~6层并行。
    use_mmscope / use_ref_score: 见模块 docstring, 互相独立, 默认全部关闭。
    """

    def __init__(self, gdino, mask_dim=256, mask_aux_layers=True, use_hed=False, use_dn=False,
                 dn_cfg=None, use_mmscope=False, mm_cfg=None, use_ref_score=False):
        super().__init__()
        if use_dn and not use_hed:
            # 本阶段只迁移 HED 的 DN; B0 串行 decoder + DN 未实现, 不能静默换成 HED 结构。
            raise ValueError("本阶段不支持 B0+DN (use_hed=False, use_dn=True)")
        if use_ref_score and not use_hed:
            # 评分头的六层集成与 top1 选择只在 HED 路径实现; 不静默切换模型结构。
            raise ValueError("本阶段 use_ref_score=True 只支持 use_hed=True")
        self.gd = gdino
        self.mask_aux_layers = mask_aux_layers
        self.use_hed = use_hed
        self.use_dn = use_dn
        self.use_mmscope = use_mmscope
        self.use_ref_score = use_ref_score
        swin = self.gd.backbone[0]
        assert swin.out_indices == (1, 2, 3)
        assert self.gd.num_feature_levels == 4
        self._c4 = None
        c4_dim = swin.num_features[0]
        self.c4_norm = nn.LayerNorm(c4_dim)
        swin.layers[0].register_forward_hook(self._grab_c4)
        hidden = self.gd.hidden_dim
        self.mm_cfg = dict(mm_cfg or {})
        if use_mmscope:
            # 只有开启时才导入: 关闭时不触碰 triton / selective_scan, 旧环境照常可用。
            from .mmscope_pixel_decoder import MMSCopEPixelDecoder
            self.pixel_decoder = MMSCopEPixelDecoder(
                c4_dim, enc_dim=hidden, out_dim=mask_dim,
                **{k: self.mm_cfg[k]
                   for k in MM_DECODER_KEYS if k in self.mm_cfg})
        else:
            self.pixel_decoder = FPNPixelDecoder(c4_dim, hidden, mask_dim)
        self.mask_head = MaskHead(hidden, mask_dim)
        # DN 生成器必须在 FPN / MaskHead 之后创建: 它的 nn.Embedding 初始化会消耗随机数,
        # 放在后面才能保证 DN 开/关时 FPN / MaskHead 的初始化完全相同。
        # use_dn=False 时不创建, 不消耗任何随机数, state_dict 与原 HED-no-DN 相同。
        self.dn_cfg = dict(dn_cfg or {})
        if use_dn:
            self.dn_generator = CdnQueryGenerator(
                embed_dims=hidden, num_matching_queries=self.gd.num_queries,
                **{k: self.dn_cfg[k]
                   for k in DN_GENERATOR_KEYS if k in self.dn_cfg})
            self.dn_reinit_rate = float(self.dn_cfg.get("reinit_rate", 0.5))
            self.dn_branch_ref_mode = self.dn_cfg.get("branch_ref_mode", "sigmoid")
            assert 0.0 <= self.dn_reinit_rate <= 1.0, "reinit_rate (论文 τ) 必须在 [0,1]"
        else:
            self.dn_generator = None
        # 评分头最后创建: 关闭时不创建、不消耗随机数, 因此 use_ref_score=False 时
        # FPN / MaskHead / dn_generator 的初始化与 2026-10-13 版逐元素相同;
        # 开启时也不会改变上面任何模块的随机数流。
        self.ref_score_head = RefScoreHead(hidden) if use_ref_score else None

    def _grab_c4(self, module, inputs, output):
        x_out, H, W = output[0], output[1], output[2]
        self._c4 = self.c4_norm(x_out).view(-1, H, W, x_out.shape[-1]).permute(0, 3, 1,
                                                                               2).contiguous()

    def encode_text(self, captions, device):
        gd = self.gd
        tokenized = gd.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
        text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, gd.specical_tokens, gd.tokenizer)
        if text_self_attention_masks.shape[1] > gd.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :gd.max_text_len, :gd.
                                                                  max_text_len]
            position_ids = position_ids[:, :gd.max_text_len]
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                tokenized[k] = tokenized[k][:, :gd.max_text_len]
        if gd.sub_sentence_present:
            tok_enc = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tok_enc["attention_mask"] = text_self_attention_masks
            tok_enc["position_ids"] = position_ids
        else:
            tok_enc = tokenized
        bert_output = gd.bert(**tok_enc)
        encoded_text = gd.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()
        if encoded_text.shape[1] > gd.max_text_len:
            encoded_text = encoded_text[:, :gd.max_text_len]
            text_token_mask = text_token_mask[:, :gd.max_text_len]
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }
        return text_dict, tokenized

    def _build_srcs(self, samples, gd):
        """图像特征提取与多尺度 proj，返回 (srcs, masks, poss, c4)。"""
        self._c4 = None
        features, poss = gd.backbone(samples)
        c4 = self._c4
        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(gd.input_proj[l](src))
            masks.append(mask)
        for l in range(len(srcs), gd.num_feature_levels):
            src = gd.input_proj[l](
                features[-1].tensors) if l == len(features) else gd.input_proj[l](srcs[-1])
            mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            poss.append(gd.backbone[1](NestedTensor(src, mask)).to(src.dtype))
            srcs.append(src)
            masks.append(mask)
        return srcs, masks, poss, c4

    def forward(self, samples, captions, targets=None):
        """targets 只在 训练 且 use_dn=True 时使用 (生成 DN 查询); 其余情况忽略。
        eval / infer 不需要也不读取 targets。"""
        gd = self.gd
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        device = samples.tensors.device
        text_dict, tokenized = self.encode_text(captions, device)
        srcs, masks, poss, c4 = self._build_srcs(samples, gd)

        if not self.use_hed:
            return self._forward_b0(gd, srcs, masks, poss, c4, text_dict, tokenized, samples)
        dn_targets = None
        if self.use_dn and self.training:
            if targets is None:
                raise ValueError("HED+DN 训练前向需要 targets 来生成去噪查询")
            dn_targets = targets
        return self._forward_hed(gd, srcs, masks, poss, c4, text_dict, tokenized, samples,
                                 dn_targets)

    # ------------------------------------------------------------------
    # 原 B0 前向（use_hed=False，与原始实现完全一致）
    # ------------------------------------------------------------------
    def _forward_b0(self, gd, srcs, masks, poss, c4, text_dict, tokenized, samples):
        holder = {}
        h = gd.transformer.encoder.register_forward_hook(
            lambda m, i, o: holder.__setitem__("memory", o[0]))
        try:
            hs, reference, hs_enc, ref_enc, init_box_proposal = gd.transformer(
                srcs, masks, None, poss, None, None, text_dict)
        finally:
            h.remove()
        memory = holder["memory"]

        coords = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], gd.bbox_embed, hs):
            coords.append((layer_bbox_embed(layer_hs) + inverse_sigmoid(layer_ref_sig)).sigmoid())
        coords = torch.stack(coords)
        logits = torch.stack(
            [cls(layer_hs, text_dict) for cls, layer_hs in zip(gd.class_embed, hs)])

        enc_maps, start = [], 0
        for src in srcs[:3]:
            b, c, hh, ww = src.shape
            enc_maps.append(memory[:, start:start + hh * ww].transpose(1, 2).reshape(b, c, hh, ww))
            start += hh * ww
        pix = self.pixel_decoder(c4, enc_maps)

        mask_layers = range(len(hs)) if (self.mask_aux_layers and self.training) else [len(hs) - 1]
        mask_logits = {i: self.mask_head(hs[i], pix) for i in mask_layers}

        return {
            "pred_logits":
            logits[-1],
            "pred_boxes":
            coords[-1],
            "pred_masks":
            mask_logits[len(hs) - 1],
            "aux_outputs": [{
                "pred_logits": logits[i],
                "pred_boxes": coords[i],
                "pred_masks": mask_logits.get(i)
            } for i in range(len(hs) - 1)],
            "interm_outputs": {
                "pred_logits": gd.transformer.enc_out_class_embed(hs_enc[-1], text_dict),
                "pred_boxes": ref_enc[-1],
            },
            "text_token_mask":
            text_dict["text_token_mask"],
            "tokenized":
            tokenized,
            "pad_mask":
            samples.mask,
        }

    # ------------------------------------------------------------------
    # 只跑 encoder + 语言引导查询选择，不跑 decoder。
    # 逐行对应 third_party/.../transformer.py::Transformer.forward 的
    # "prepare input for encoder" / "Begin Encoder" / "End preparing tgt" 三段
    # (该文件第 221-359 行), 到 decoder 调用之前为止。
    # 抽出来的唯一原因: 官方 forward 把 encoder 和串行 decoder 绑在一起,
    # HED 需要在同一份 encoder 输出上换一种解码方式, 不能让 decoder 跑两遍。
    # encoder / enc_out_class_embed / tgt_embed 等权重与调用顺序均未改动。
    # ------------------------------------------------------------------
    def _encode_and_init_query(self, gd, srcs, masks, poss, text_dict):
        tf = gd.transformer
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poss)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if tf.num_feature_levels > 1 and tf.level_embed is not None:
                pos_embed = pos_embed + tf.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(pos_embed)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([tf.get_valid_ratio(m) for m in masks], 1)

        memory, memory_text = tf.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        # 与官方一致: encoder 增强后的文本特征写回 text_dict, 供 decoder 与 class_embed 使用
        text_dict["encoded_text"] = memory_text

        assert tf.two_stage_type == "standard", \
            f"HED 路径目前只支持 two_stage_type='standard' (当前 {tf.two_stage_type})"
        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        output_memory = tf.enc_output_norm(tf.enc_output(output_memory))
        enc_outputs_class_unselected = tf.enc_out_class_embed(output_memory, text_dict)
        topk_logits = enc_outputs_class_unselected.max(-1)[0]
        enc_outputs_coord_unselected = tf.enc_out_bbox_embed(output_memory) + output_proposals
        topk_proposals = torch.topk(topk_logits, tf.num_queries, dim=1)[1]

        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                               topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        refpoint_embed_ = refpoint_embed_undetach.detach()
        tgt_undetach = torch.gather(output_memory, 1,
                                    topk_proposals.unsqueeze(-1).repeat(1, 1, tf.d_model))
        # embed_init_tgt 分支必须与官方一致: SwinT-OGC 配置为 True, 用可学习的 tgt_embed
        if tf.embed_init_tgt:
            tgt_ = tf.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        else:
            tgt_ = tgt_undetach.detach()

        return dict(
            memory=memory,
            mask_flatten=mask_flatten,
            lvl_pos_embed_flatten=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            tgt=tgt_,
            refpoint_embed=refpoint_embed_,
            hs_enc=tgt_undetach.unsqueeze(0),
            ref_enc=refpoint_embed_undetach.sigmoid().unsqueeze(0),
        )

    # ------------------------------------------------------------------
    # HED 无DN结构版前向（use_hed=True）
    # NOTE: "掩码概率平均"是本项目对 FT-FSOD 的分割适配，原文无此设计
    # ------------------------------------------------------------------
    def _forward_hed(self, gd, srcs, masks, poss, c4, text_dict, tokenized, samples,
                     dn_targets=None):
        # ---- encoder + 查询初始化: 与 B0 相同的权重与顺序, 但不跑串行 decoder ----
        enc = self._encode_and_init_query(gd, srcs, masks, poss, text_dict)
        memory = enc["memory"]  # (B, sum(HW), 256)
        hs_enc = enc["hs_enc"]
        ref_enc = enc["ref_enc"]

        # memory 按 encoder 布局转成 [1/8, 1/16, 1/32] 特征图（FPN用）
        enc_maps, start = [], 0
        for src in srcs[:3]:
            b, c, hh, ww = src.shape
            enc_maps.append(memory[:, start:start + hh * ww].transpose(1, 2).reshape(b, c, hh, ww))
            start += hh * ww
        pix = self.pixel_decoder(c4, enc_maps)

        # ---- HED 并行解码 ----
        # tgt / refpoint_embed 由 _encode_and_init_query 按官方逻辑给出
        # (embed_init_tgt=True 时为可学习的 tgt_embed, 与原 B0 一致)。
        # decoder 接口为 nq-first, 故在此转置, 与 transformer.forward 调用 decoder 时的写法相同。
        dec_kwargs = dict(
            decoder=gd.transformer.decoder,
            tgt=enc["tgt"].transpose(0, 1),  # (Q, B, C)
            memory=memory.transpose(0, 1),  # (sum_HW, B, C)
            valid_ratios=enc["valid_ratios"],
            spatial_shapes=enc["spatial_shapes"],
            level_start_index=enc["level_start_index"],
            memory_key_padding_mask=enc["mask_flatten"],
            refpoints_unsigmoid=enc["refpoint_embed"].transpose(0, 1),  # (Q, B, 4) unsigmoid
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict[
                "text_token_mask"],  # True=pad (文本 key padding, 与 DN self-attn mask 无关)
            memory_pos=enc["lvl_pos_embed_flatten"].transpose(0, 1),
            gen_sineembed_for_position=gen_sineembed_for_position,
        )

        # ---- DN 查询: 只在 训练 且 use_dn 时由 GT 生成; 推理路径完全不碰 GT ----
        dn0 = None
        if dn_targets is not None:
            dn0 = self.dn_generator([t["boxes"] for t in dn_targets])  # 整个 batch 无目标时为 None

        if dn0 is None:
            hs_hed, refs_hed, input_refs_hed = _hed_parallel_decode(**dec_kwargs)
            head_refs_full = [refs_hed[0]] + [refs_hed[1]] * (len(hs_hed) - 1)
            num_dn, layer_dn = 0, None
        else:
            branch_dns = _sample_branch_dns(self.dn_generator, [t["boxes"] for t in dn_targets],
                                            len(gd.transformer.decoder.layers) - 1,
                                            self.dn_reinit_rate, _py_random)
            dec = _hed_parallel_decode_dn(dn0=dn0, branch_dns=branch_dns,
                                          branch_ref_mode=self.dn_branch_ref_mode, **dec_kwargs)
            hs_hed, input_refs_hed = dec["hs"], dec["input_refs"]
            head_refs_full, num_dn, layer_dn = dec["head_refs"], dec["num_dn"], dec["layer_dn"]
        # hs_hed: list[6×(B,Ndn+Q,C)] (Ndn=0 时即 no-DN 版); 各层前 num_dn 个是 DN, 其后是普通查询

        # ---- 6层分类 / 框预测 ----
        # 预测头参考框 head_refs 与 decoder 输入参考框 input_refs_hed 数值相同, 梯度不同:
        #   - decoder 内部用的是 detach 版 (input_refs_hed, 与 B0 transformer.py:726 一致);
        #   - 预测头用未 detach 版, 与 B0 一致: B0 model.py 用 reference[:-1],
        #     第 i 层框 = bbox_embed(hs_i) + inverse_sigmoid(第 i-1 层未 detach 的更新框),
        #     第 i 层框损失因此能回传到第 i-1 层的框偏移和输出 (look-forward-twice)。
        #   第0层: refs_hed[0] = 初始框 (本身来自已 detach 的 refpoint_embed, 同 B0)。
        #   第1~5层: refs_hed[1] = 共享层未 detach 的更新框; 五个分支实际输入都是它的数值,
        #   所以不能用 refs_hed[2..5] (那是其他分支的输出, 串行假设, 官方 head 在此处配错)。
        # 2026-10-13 修复: 原先这里直接用 input_refs_hed, 截断了五个并行层框损失
        # 通向共享层的梯度, 与 B0 和官方 (第2层) 都不一致。只影响训练, 不影响推理数值。
        # DN 版: head_refs_full 由 hed_parallel_decode_dn 给出, 每层 DN 部分是该层自己
        # 实际消费的 DN 参考框 (继承分支=共享层 DN 的更新框, 重新生成分支=该分支 DN 的框),
        # 普通部分仍是共享层未 detach 的更新框, 不与其他分支混用。
        head_refs = head_refs_full
        coords_all, logits_all = [], []
        for i, (layer_hs, layer_ref_sig) in enumerate(zip(hs_hed, head_refs)):
            coords_all.append(
                (gd.bbox_embed[i](layer_hs) + inverse_sigmoid(layer_ref_sig)).sigmoid())
            logits_all.append(gd.class_embed[i](layer_hs, text_dict))
        coords_all = torch.stack(coords_all)  # (6, B, Ndn+Q, 4)
        logits_all = torch.stack(logits_all)  # (6, B, Ndn+Q, 256)

        # ---- 先拆分 DN 与普通查询 (官方 DINOHead.split_outputs: 前 Ndn 个是 DN) ----
        # 之后的掩码、Hungarian 匹配、六层汇总只看普通查询; DN 只进入 out["dn_outputs"]。
        coords_hed, logits_hed = coords_all[:, :, num_dn:], logits_all[:, :, num_dn:]  # (6,B,Q,·)
        hs_hed = [h[:, num_dn:] for h in hs_hed]
        dn_outputs = None
        if num_dn > 0:
            dn_outputs = {
                "pred_logits": logits_all[:, :, :num_dn],  # (6, B, Ndn, 256)
                "pred_boxes": coords_all[:, :, :num_dn],  # (6, B, Ndn, 4)
                "layer_meta": [d.meta for d in layer_dn]
            }  # 第 i 层自己的 metadata
        elif self.use_dn and self.training:
            # 官方 forward_decoder: 无 GT 时让 label_embedding 以 0 系数进入图,
            # 防止分布式训练报"参数未参与反传"。数值不变。
            logits_hed = logits_hed + self.dn_generator.label_embedding.weight[0, 0] * 0.0

        # ---- 新增评分头: 只看剥离 DN 后的普通查询 (hs_hed 已在上面切掉前 num_dn 个) ----
        # 六层共用同一份权重 (self.ref_score_head 只有一个), 对每层分别调用。
        # 不 detach: 评分损失的梯度可以更新查询特征, 与 5.1 节要求一致。
        ref_logits_all = None
        if self.use_ref_score:
            ref_logits_all = torch.stack([self.ref_score_head(h) for h in hs_hed])  # (6, B, Q)

        # HED 汇总 (六层概率/框平均) 在 evaluate.py::postprocess_hed 中完成, 前向只输出各层结果。

        # ---- 掩码: 训练时各层独立监督，推理时对选定查询逐层累加 ----
        if self.training and self.mask_aux_layers:
            mask_logits = {i: self.mask_head(hs_hed[i], pix) for i in range(len(hs_hed))}
            hed_mask_embeds = None
        else:
            # 推理: 只算最后一层的全查询掩码 (保留"最后一层掩码"对照选项),
            # 另外输出六层的掩码嵌入 (6,B,Q,C) 与像素特征 pix,
            # 由 evaluate.py::postprocess_hed 只对选中的查询逐层计算掩码并平均,
            # 避免保存 6 层 × 全部查询的掩码。
            mask_logits = {len(hs_hed) - 1: self.mask_head(hs_hed[-1], pix)}
            hed_mask_embeds = torch.stack([self.mask_head.mask_embed(h) for h in hs_hed])

        # ---- 构造输出字典 ----
        # 训练时 pred_logits 必须是原始 logits（criterion 内部会 sigmoid）
        # 推理时的多层概率平均在 postprocess_hed 中通过 hed_all_logits 实现
        out = {
            "pred_logits":
            logits_hed[-1],  # 最后层 logits（训练用，criterion 兼容）
            "pred_boxes":
            coords_hed[-1],  # 最后层框（训练用）
            "pred_masks":
            mask_logits[len(hs_hed) - 1],
            "hed_all_logits":
            logits_hed,  # (6,B,Q,256) logits，用于推理时更精确的均值
            "hed_all_boxes":
            coords_hed,  # (6,B,Q,4)
            "hed_mask_embeds":
            hed_mask_embeds,  # 推理时 (6,B,Q,C)，训练时 None
            "hed_pix":
            None if hed_mask_embeds is None else pix,  # (B,C,H/4,W/4)
            "aux_outputs": [{
                "pred_logits": logits_hed[i],
                "pred_boxes": coords_hed[i],
                "pred_masks": mask_logits.get(i)
            } for i in range(len(hs_hed) - 1)],
            "interm_outputs": {
                "pred_logits": gd.transformer.enc_out_class_embed(hs_enc[-1], text_dict),
                "pred_boxes": ref_enc[-1],
            },
            "text_token_mask":
            text_dict["text_token_mask"],
            "tokenized":
            tokenized,
            "pad_mask":
            samples.mask,
        }
        if ref_logits_all is not None:
            # use_ref_score=True 时训练与推理前向都给出; 关闭时这几个键根本不存在。
            out["pred_ref_logits"] = ref_logits_all[-1]  # (B,Q) 最后层, 原始 logits
            out["hed_all_ref_logits"] = ref_logits_all  # (6,B,Q) 推理选 q* 用
            for i, aux in enumerate(out["aux_outputs"]):
                aux["pred_ref_logits"] = ref_logits_all[i]  # 第 i 层自己的 logits
        if dn_outputs is not None:
            out["dn_outputs"] = dn_outputs  # 只有训练+DN 才有; 推理与 no-DN 时不存在此键
        return out


def build_model(cfg_path, ckpt_path=None, bert_path="bert-base-uncased", device="cuda",
                use_checkpoint=None, use_hed=False, use_dn=False, dn_cfg=None, use_mmscope=False,
                mm_cfg=None, use_ref_score=False):
    """构建模型。

    use_hed=False: 原 B0 行为，与原始 checkpoint 完全兼容。
    use_hed=True : HED 无DN结构版，使用与 B0 相同的权重加载路径。
                   HED 结构不增加新参数，只改变解码器的调用方式，
                   因此所有预训练层（6层 decoder、bbox_embed、class_embed）均完整保留。
    use_dn=True  : HED+DN, 只新增 dn_generator.label_embedding.weight (N(0,1) 新初始化)。
                   预训练权重里的 label_enc.weight (IDEA 原版 DN 标签表) 不加载:
                   B0 构建的 groundingdino 不含该模块, 它一直作为 unexpected key 被忽略
                   (与原 B0 相同); 其行号是预训练数据的类别编号, 与本项目
                   "一条描述 = 类别 0" 的语义不对应, 见 docs/HED_DN接入说明_20261013.md。
    use_mmscope=True    : 像素支路换成 MMSCopEPixelDecoder, 全部参数新初始化
                   (本工程不加载 SegMAN 预训练权重; 预训练 GroundingDINO 里也没有这些键)。
    use_ref_score=True : 新增 ref_score_head.*, 全部新初始化。
    以上两个开关都不影响 gd.* 的加载路径。
    """
    if use_dn and not use_hed:
        raise ValueError("本阶段不支持 B0+DN (use_hed=False, use_dn=True)")
    if use_ref_score and not use_hed:
        raise ValueError("本阶段 use_ref_score=True 只支持 use_hed=True")
    args = SLConfig.fromfile(cfg_path)
    args.device = device
    args.text_encoder_type = bert_path
    if use_checkpoint is not None:
        args.use_checkpoint = use_checkpoint
        args.use_transformer_ckpt = use_checkpoint
    gdino = build_groundingdino(args)
    load_info = None
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        load_info = gdino.load_state_dict(sd, strict=False)
    model = GroundingDINORefSeg(gdino, use_hed=use_hed, use_dn=use_dn, dn_cfg=dn_cfg,
                                use_mmscope=use_mmscope, mm_cfg=mm_cfg, use_ref_score=use_ref_score)
    return model.to(device), load_info


def allowed_missing_keys(model):
    """把本项目旧 checkpoint (B0 / HED-no-DN) 加载进当前模型时, 允许缺失的键。
    只有 HED+DN 新建的 DN 参数; 其余任何缺失都视为错误。"""
    return sorted(k for k in model.state_dict() if k.startswith(DN_STATE_PREFIX))


def load_model_state(model, state_dict):
    """加载本项目训练出的 checkpoint (ck["model"])。用于 --resume / eval-only / infer。

    - 键完全匹配: 等价于 strict=True。
    - 旧 checkpoint 缺少 DN 参数: 只允许缺 allowed_missing_keys(model), 这些参数保留新初始化。
    - HED+DN 权重加载进 DN 关闭的模型 (用 use_dn=False 评估同一份权重): 只丢弃
      dn_generator.* —— 它们只在训练时生成 DN 查询, 推理不使用。显式过滤, 不靠 strict=False。
    - 任何其他缺失键或多余键: 抛错, 不静默吞掉。新增的 MM 像素支路与评分头在这里
      **不**享受任何豁免: 要把旧 B0_HED 权重迁到 MM/Score 新模型, 必须走 init_model_weights
      (tools/train.py --init-from), 由那条路径显式报告保留/新建的键。
    - 评分头与 DN 的处理为何不同: dn_generator 只在训练时生成 DN 查询, 推理不用, 所以
      "把 HED+DN 权重加载进 DN 关闭的模型做评估"是安全的, 可以丢弃。
      而 ref_score_head **决定推理的 top1 选择**, 把带评分头的权重加载进评分关闭的模型
      会静默换回 token 均分那套选择规则 —— 所以这里不丢弃、直接报 unexpected,
      要求调用方把模式说清楚 (cfg 里的 use_ref_score 由 resolve_model_mode 从 checkpoint 恢复)。
    返回实际缺失 (使用新初始化) 或被丢弃的 DN 键列表, 供日志记录。
    """
    allowed = set(allowed_missing_keys(model))
    dropped = []
    if getattr(model, "dn_generator", None) is None:
        dropped = sorted(k for k in state_dict if k.startswith(DN_STATE_PREFIX))
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(DN_STATE_PREFIX)}
    res = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in res.missing_keys if k not in allowed]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(
            f"checkpoint 与模型不匹配: missing={bad_missing} unexpected={res.unexpected_keys}")
    return list(res.missing_keys) + dropped


def init_model_weights(model, state_dict, old_cfg=None):
    """weights-only 初始化 (tools/train.py --init-from): 用旧 B0_HED checkpoint 的权重
    初始化开启了 MM / 评分头的新模型。只加载权重, 不恢复优化器 / epoch / best。

    缺键白名单按 **源 checkpoint 的开关 × 目标模型的开关** 逐模块判定, 不是无条件放行:

      评分头 (ref_score_head.*)            源关 / 目标开 -> 允许整体新建 (报告为 new);
                                           源开 / 目标开 -> **必须完整匹配**, 缺一个键就报错
                                                           (权重损坏或不完整不能当成"正常新增");
                                           源开 / 目标关 -> 丢弃并报告;
                                           源关 / 目标关 -> 无此模块。
      DN (dn_generator.*)                  沿用原有约定: 目标开则允许新建, 目标关则丢弃。
      像素支路 (pixel_decoder.*)           类型相同 -> 必须完整匹配;
                                           类型不同 (FPN <-> MMSCopE) -> 整条支路丢弃并新建。
                                           同属性名但结构不同, 同名同形状也不代表语义一致,
                                           因此不做逐键部分复用。
      MaskHead / c4_norm / gd.*            形状与语义未变, 必须完整匹配, 缺键报错。

    其余任何差异 (未在白名单内的缺键、多余键、同名形状不符) 一律 RuntimeError。
    源 checkpoint 的开关取自 old_cfg (tools/train.py 传入 checkpoint 里的 cfg);
    old_cfg 缺字段时按 False 处理 —— 那时的 checkpoint 只可能是 FPN / 无评分头。

    返回 dict: kept_count / kept_groups / new (新初始化) / dropped (被丢弃) / 各模块判定依据。
    """
    model_sd = model.state_dict()
    sd = dict(state_dict)
    old = dict(old_cfg or {})
    dropped = []

    def _drop(prefix):
        drop = sorted(k for k in sd if k.startswith(prefix))
        dropped.extend(drop)
        for k in drop:
            sd.pop(k)

    # ---- 逐模块判定: new_allowed 决定该前缀的缺键是否算"正常新建" ----
    old_score = bool(old.get("use_ref_score", False))
    new_score = getattr(model, "ref_score_head", None) is not None
    old_mm = bool(old.get("use_mmscope", False))
    new_mm = bool(getattr(model, "use_mmscope", False))
    new_dn = getattr(model, "dn_generator", None) is not None

    decisions = {}
    new_allowed_prefixes = []

    # 评分头
    if new_score and not old_score:
        decisions["ref_score_head"] = "源无 / 目标有 -> 整体新建"
        new_allowed_prefixes.append(REF_SCORE_PREFIX)
    elif new_score and old_score:
        decisions["ref_score_head"] = "源有 / 目标有 -> 必须完整匹配 (缺键报错)"
    elif old_score and not new_score:
        decisions["ref_score_head"] = "源有 / 目标无 -> 丢弃"
        _drop(REF_SCORE_PREFIX)
    else:
        decisions["ref_score_head"] = "两侧都无"
    # 一致性检查: cfg 说没有评分头, 但 state_dict 里带着这些键 -> checkpoint 自相矛盾, 直接报错,
    # 不按 old_score=False 当成"源无"去放行 (否则目标开启评分头时会误判成"整体新建"而丢掉真权重)。
    if not old_score and any(k.startswith(REF_SCORE_PREFIX) for k in sd):
        raise RuntimeError("--init-from 的 checkpoint 自相矛盾: cfg 里 use_ref_score=False, "
                           f"但权重里存在 {REF_SCORE_PREFIX}* 的键。请确认该 checkpoint 的来源与模式。")
    # DN (沿用原有兼容约定)
    if new_dn:
        decisions["dn_generator"] = "目标有 -> 缺则新建 (原 DN 约定)"
        new_allowed_prefixes.append(DN_STATE_PREFIX)
    else:
        decisions["dn_generator"] = "目标无 -> 丢弃"
        _drop(DN_STATE_PREFIX)
    # 像素支路
    replaced_pixel_decoder = old_mm != new_mm
    if replaced_pixel_decoder:
        decisions["pixel_decoder"] = (f"类型变化 (源 use_mmscope={old_mm} -> 目标 {new_mm}) "
                                      "-> 整条支路丢弃并新建")
        _drop(PIXEL_DECODER_PREFIX)
        new_allowed_prefixes.append(PIXEL_DECODER_PREFIX)
    else:
        decisions["pixel_decoder"] = (f"类型相同 (use_mmscope={new_mm}) -> 必须完整匹配 (缺键报错)")

    shape_bad = [
        f"{k}: ckpt{tuple(v.shape)} != model{tuple(model_sd[k].shape)}" for k, v in sd.items()
        if k in model_sd and hasattr(v, "shape") and tuple(v.shape) != tuple(model_sd[k].shape)
    ]
    if shape_bad:
        raise RuntimeError(f"--init-from 权重形状不匹配: {shape_bad}")

    res = model.load_state_dict(sd, strict=False)
    allowed_new = set(k for k in model_sd if any(k.startswith(p) for p in new_allowed_prefixes))
    bad_missing = [k for k in res.missing_keys if k not in allowed_new]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(
            "--init-from 与当前模型不兼容 (缺键白名单按源/目标开关判定, 见 decisions): "
            f"missing={bad_missing} unexpected={res.unexpected_keys} decisions={decisions}")
    kept = sorted(set(sd) & set(model_sd))
    # kept 有数千个 gd.* 键, 日志里只给数量和按前缀的分组计数; new / dropped 全量列出。
    return {
        "kept_count": len(kept),
        "kept_groups": _key_groups(kept),
        "new": sorted(res.missing_keys),
        "dropped": sorted(dropped),
        "pixel_decoder_replaced": replaced_pixel_decoder,
        "source_flags": {
            "use_mmscope": old_mm,
            "use_ref_score": old_score
        },
        "decisions": decisions
    }


def _key_groups(keys):
    """按顶层前缀统计键数, 供 --init-from 日志使用 (gd.backbone.0.xxx -> 'gd.backbone')。"""
    groups = {}
    for k in keys:
        parts = k.split(".")
        g = ".".join(parts[:2]) if parts[0] == "gd" else parts[0]
        groups[g] = groups.get(g, 0) + 1
    return dict(sorted(groups.items()))
