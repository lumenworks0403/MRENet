"""MRENet：GroundingDINO 编码器、MCPE 像素增强、SPED 并行解码和 RCA。
use_hed/use_mmscope/use_ref_score 为保留的接口名，分别对应 SPED/MCPE/RCA。"""
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
import random as _py_random

DN_GENERATOR_KEYS = ("label_book_size", "label_noise_scale", "box_noise_scale", "num_dn_queries")

DN_STATE_PREFIX = "dn_generator."

REF_SCORE_PREFIX = "ref_score_head."
PIXEL_DECODER_PREFIX = "pixel_decoder."

MM_DECODER_KEYS = ("channels", "feat_proj_dim", "norm", "gn_groups", "scan_fp32", "drop_path")


def _patch_deform_attn_if_no_cuda_op():
    """未编译 CUDA 扩展时使用 GroundingDINO 的可微 PyTorch deformable attention。"""
    if hasattr(_msda, "_C"):
        return False
    _msda.MultiScaleDeformableAttnFunction.apply = staticmethod(
        lambda value, shapes, level_start, loc, attn, im2col: _msda.
        multi_scale_deformable_attn_pytorch(value, shapes, loc, attn))
    return True


DEFORM_ATTN_PYTORCH_FALLBACK = _patch_deform_attn_if_no_cuda_op()


class FPNPixelDecoder(nn.Module):
    """基线 FPN：融合图文交互后的 1/8、1/16、1/32 特征和 1/4 浅层特征。"""

    def __init__(self, c4_channels, dim=256, out_dim=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(3)])
        self.lateral4 = nn.Conv2d(c4_channels, dim, 1)
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim),
                          nn.ReLU(inplace=True)) for _ in range(4)
        ])
        self.out = nn.Conv2d(dim, out_dim, 1)

    def forward(self, c4, enc_feats):

        x = self.smooth[3](self.lateral[2](enc_feats[2]))
        for i in (1, 0):
            lat = self.lateral[i](enc_feats[i])
            x = self.smooth[i + 1](lat + F.interpolate(x, size=lat.shape[-2:], mode="nearest"))
        lat = self.lateral4(c4)
        x = self.smooth[0](lat + F.interpolate(x, size=lat.shape[-2:], mode="nearest"))
        return self.out(x)


class MaskHead(nn.Module):
    """共享三层 MLP 生成掩码嵌入，与像素特征点积得到掩码 logits。"""

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

        return torch.einsum("bqc,bchw->bqhw", self.mask_embed(hs), pix)


class GroundingDINORefSeg(nn.Module):
    """保留预训练编码器与六个解码层参数；SPED 改为一层共享、五层独立并行。"""

    def __init__(self, gdino, mask_dim=256, mask_aux_layers=True, use_hed=False, use_dn=False,
                 dn_cfg=None, use_mmscope=False, mm_cfg=None, use_ref_score=False):
        super().__init__()
        if use_dn and not use_hed:

            raise ValueError("本阶段不支持 B0+DN (use_hed=False, use_dn=True)")
        if use_ref_score and not use_hed:

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

            from .mcpe import MCPEPixelDecoder
            self.pixel_decoder = MCPEPixelDecoder(
                c4_dim, enc_dim=hidden, out_dim=mask_dim,
                **{k: self.mm_cfg[k]
                   for k in MM_DECODER_KEYS if k in self.mm_cfg})
        else:
            self.pixel_decoder = FPNPixelDecoder(c4_dim, hidden, mask_dim)
        self.mask_head = MaskHead(hidden, mask_dim)

        self.dn_cfg = dict(dn_cfg or {})
        if use_dn:
            self.dn_generator = CdnQueryGenerator(
                embed_dims=hidden, num_matching_queries=self.gd.num_queries,
                **{k: self.dn_cfg[k]
                   for k in DN_GENERATOR_KEYS if k in self.dn_cfg})
            self.dn_reinit_rate = float(self.dn_cfg.get("reinit_rate", 0.5))
            self.dn_branch_ref_mode = self.dn_cfg.get("branch_ref_mode", "sigmoid")
            assert 0.0 <= self.dn_reinit_rate <= 1.0, "分支再生概率 p 必须在 [0,1]"
        else:
            self.dn_generator = None

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
        """构建图像多尺度特征、padding mask、位置编码及浅层细节特征。"""
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
        ''
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

    def _forward_hed(self, gd, srcs, masks, poss, c4, text_dict, tokenized, samples,
                     dn_targets=None):

        enc = self._encode_and_init_query(gd, srcs, masks, poss, text_dict)
        memory = enc["memory"]
        hs_enc = enc["hs_enc"]
        ref_enc = enc["ref_enc"]

        enc_maps, start = [], 0
        for src in srcs[:3]:
            b, c, hh, ww = src.shape
            enc_maps.append(memory[:, start:start + hh * ww].transpose(1, 2).reshape(b, c, hh, ww))
            start += hh * ww
        pix = self.pixel_decoder(c4, enc_maps)

        dec_kwargs = dict(
            decoder=gd.transformer.decoder,
            tgt=enc["tgt"].transpose(0, 1),
            memory=memory.transpose(0, 1),
            valid_ratios=enc["valid_ratios"],
            spatial_shapes=enc["spatial_shapes"],
            level_start_index=enc["level_start_index"],
            memory_key_padding_mask=enc["mask_flatten"],
            refpoints_unsigmoid=enc["refpoint_embed"].transpose(0, 1),
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            memory_pos=enc["lvl_pos_embed_flatten"].transpose(0, 1),
            gen_sineembed_for_position=gen_sineembed_for_position,
        )

        dn0 = None
        if dn_targets is not None:
            dn0 = self.dn_generator([t["boxes"] for t in dn_targets])

        if dn0 is None:
            hs_hed, refs_hed, input_refs_hed = _hed_parallel_decode(**dec_kwargs)
            # 五个分支都使用共享层参考框，不能按串行假设使用 refs_hed[:-1]。
            # 解码层消费 detach 的框；预测头保留共享框的梯度路径。
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

        head_refs = head_refs_full
        coords_all, logits_all = [], []
        for i, (layer_hs, layer_ref_sig) in enumerate(zip(hs_hed, head_refs)):
            coords_all.append(
                (gd.bbox_embed[i](layer_hs) + inverse_sigmoid(layer_ref_sig)).sigmoid())
            logits_all.append(gd.class_embed[i](layer_hs, text_dict))
        coords_all = torch.stack(coords_all)
        logits_all = torch.stack(logits_all)

        # 先剥离 DN，RCA、掩码预测及普通 Hungarian 匹配只接收普通查询。
        coords_hed, logits_hed = coords_all[:, :, num_dn:], logits_all[:, :, num_dn:]
        hs_hed = [h[:, num_dn:] for h in hs_hed]
        dn_outputs = None
        if num_dn > 0:
            dn_outputs = {
                "pred_logits": logits_all[:, :, :num_dn],
                "pred_boxes": coords_all[:, :, :num_dn],
                "layer_meta": [d.meta for d in layer_dn]
            }
        elif self.use_dn and self.training:

            logits_hed = logits_hed + self.dn_generator.label_embedding.weight[0, 0] * 0.0

        ref_logits_all = None
        if self.use_ref_score:
            ref_logits_all = torch.stack([self.ref_score_head(h) for h in hs_hed])

        if self.training and self.mask_aux_layers:
            mask_logits = {i: self.mask_head(hs_hed[i], pix) for i in range(len(hs_hed))}
            hed_mask_embeds = None
        else:

            mask_logits = {len(hs_hed) - 1: self.mask_head(hs_hed[-1], pix)}
            hed_mask_embeds = torch.stack([self.mask_head.mask_embed(h) for h in hs_hed])

        out = {
            "pred_logits":
            logits_hed[-1],
            "pred_boxes":
            coords_hed[-1],
            "pred_masks":
            mask_logits[len(hs_hed) - 1],
            "hed_all_logits":
            logits_hed,
            "hed_all_boxes":
            coords_hed,
            "hed_mask_embeds":
            hed_mask_embeds,
            "hed_pix":
            None if hed_mask_embeds is None else pix,
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

            out["pred_ref_logits"] = ref_logits_all[-1]
            out["hed_all_ref_logits"] = ref_logits_all
            for i, aux in enumerate(out["aux_outputs"]):
                aux["pred_ref_logits"] = ref_logits_all[i]
        if dn_outputs is not None:
            out["dn_outputs"] = dn_outputs
        return out


def build_model(cfg_path, ckpt_path=None, bert_path="bert-base-uncased", device="cuda",
                use_checkpoint=None, use_hed=False, use_dn=False, dn_cfg=None, use_mmscope=False,
                mm_cfg=None, use_ref_score=False):
    """加载 GroundingDINO 预训练权重并构建分割扩展。新 MCPE 和 RCA 随机初始化。"""
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
    """同架构加载时只允许新建训练专用的 DN 标签嵌入。"""
    return sorted(k for k in model.state_dict() if k.startswith(DN_STATE_PREFIX))


def load_model_state(model, state_dict):
    """严格检查模型参数；仅允许缺少或丢弃训练专用 DN 参数。"""
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
    """--init-from 只初始化权重；像素模块类型改变时整体新建，其他模块按源/目标模式严格检查。"""
    model_sd = model.state_dict()
    sd = dict(state_dict)
    old = dict(old_cfg or {})
    dropped = []

    def _drop(prefix):
        drop = sorted(k for k in sd if k.startswith(prefix))
        dropped.extend(drop)
        for k in drop:
            sd.pop(k)

    old_score = bool(old.get("use_ref_score", False))
    new_score = getattr(model, "ref_score_head", None) is not None
    old_mm = bool(old.get("use_mmscope", False))
    new_mm = bool(getattr(model, "use_mmscope", False))
    new_dn = getattr(model, "dn_generator", None) is not None

    decisions = {}
    new_allowed_prefixes = []

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

    if not old_score and any(k.startswith(REF_SCORE_PREFIX) for k in sd):
        raise RuntimeError("--init-from 的 checkpoint 自相矛盾: cfg 里 use_ref_score=False, "
                           f"但权重里存在 {REF_SCORE_PREFIX}* 的键。请确认该 checkpoint 的来源与模式。")

    if new_dn:
        decisions["dn_generator"] = "目标有 -> 缺则新建 (原 DN 约定)"
        new_allowed_prefixes.append(DN_STATE_PREFIX)
    else:
        decisions["dn_generator"] = "目标无 -> 丢弃"
        _drop(DN_STATE_PREFIX)

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
    """按顶层参数前缀统计加载数量。"""
    groups = {}
    for k in keys:
        parts = k.split(".")
        g = ".".join(parts[:2]) if parts[0] == "gd" else parts[0]
        groups[g] = groups.get(g, 0) + 1
    return dict(sorted(groups.items()))
