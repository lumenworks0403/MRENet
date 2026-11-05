"""MMSCopE 像素支路 (B0_HED 可选, use_mmscope=True 时替换 FPNPixelDecoder)。

接口与原 FPNPixelDecoder 完全相同: ``pix = pixel_decoder(c4, enc_maps)``
  c4:       (B, C4, H4, W4)  Swin stage-0 的 1/4 浅层特征 (本工程 hook + c4_norm 得到)
  enc_maps: [ (B,D,H8,W8), (B,D,H16,W16), (B,D,H32,W32) ]  图文融合后的 encoder memory
  返回 pix: (B, mask_dim, H4, W4)  —— 空间尺寸与 c4 完全一致, 通道匹配现有 MaskHead

官方来源 (完整版本信息见 third_party/SegMAN/VERSION.md):
  SegMAN, commit 9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8
  segmentation/mmseg/models/decode_heads/segman_decoder.py
    SHA256 a163365fc0bd1a43b09ba4bbecbdd706b97b7bc775e32671be6225db50a095b5
  算法说明: SegMAN.pdf 第 3.3 节 "Mamba-based multi-scale context extraction" 与图 3(d)。

官方 → 本文件 迁移对应 (详见 docs/MMSCopE与评分头接入说明_20261016.md):
  SegMANDecoder.forward_mlp_decoder  -> MMSCopEPixelDecoder.forward_mlp_decoder   (按原语句顺序改写)
  SegMANDecoder.forward_winssm       -> MMSCopEPixelDecoder.forward_winssm        (按原语句顺序改写)
  SegMANDecoder.__init__ 的模块表     -> MMSCopEPixelDecoder.__init__
  VSSMBlock / VSSM(SS2D) / FFN       -> third_party/SegMAN/segman/mmscope_core.py (官方代码抽取)
  CrossScanTriton / CrossMergeTriton -> third_party/SegMAN/segman/csm_triton.py   (原样复制)
  SegMANDecoder.cls_seg              -> 不迁移: 本项目取语义分类之前的特征, 不要类别 logits

官方层级名与本工程分辨率的对应 (官方 c4 = 最深层, 本工程 c4 = 1/4 浅层, 不要混淆):
  官方 F2/c2 (1/8)  = enc_maps[0]      官方 F3/c3 (1/16) = enc_maps[1]
  官方 F4/c4 (1/32) = enc_maps[2]      官方 c1 (1/4, 未被 MMSCopE 使用) ≠ 本工程 c4

本项目新增的适配 (不属于官方算法, 文档中逐条记录):
  A1 1/4 细节融合: 官方输出在 1/8 直接接分类头; 这里把上下文特征对齐到 c4 的实际尺寸,
     与 1/4 浅层特征做与原 FPNPixelDecoder 相同的 "1x1 lateral + 相加 + 3x3 GN ReLU + 1x1 out" 融合。
  A2 ConvModule 替换: 官方 ConvModule(norm_cfg=SyncBN) -> 本文件 conv_norm_act, 顺序
     conv(bias=False) -> norm -> ReLU 与官方一致; 默认把 SyncBN 换成 GroupNorm, 原因:
     本工程 batch_size=1~2 单卡, 且官方 image_pool 在 1x1 特征上做 BN, batch=1 时
     PyTorch 直接报 "Expected more than 1 value per channel when training"。GroupNorm 与
     现有 FPNPixelDecoder 的 GroupNorm(32, dim) 约定一致。norm="bn" 可切回 BatchNorm 对照
     (batch=1 会报错, 这是 BN 本身的限制)。初始化按 mmcv ConvModule 默认复刻:
     kaiming_normal_(fan_out, relu) + norm weight=1 / bias=0。
  A3 尺寸 pad: Pixel Unshuffle 要求 1/8 特征的 H、W 能被 4 整除 (800x1333 输入时 W8=167 不满足)。
     只在本模块内部把 1/8 特征 replicate-pad 到 4 的倍数, 算完后裁回原尺寸,
     不改变任何对外的尺寸与坐标恢复语义。
  A4 scan_fp32: 官方只在 FP32 下训练; selective_scan_cuda_oflex 要求 u/delta/B/C dtype 相同,
     而 autocast 下 LayerNorm 产出 fp32、conv1d 产出 fp16, 会触发 TORCH_CHECK 失败。
     因此默认只在 VSSMBlock 调用处关闭 autocast 并转 fp32 (扫描在 1/32 分辨率, 开销小),
     其余卷积仍按外部 AMP 执行。不把整个模型改成 FP32。
  A5 内部通道宽度与官方权重无关: 本工程不加载 SegMAN 预训练权重, 新支路全部随机初始化。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_SEGMAN_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "third_party", "SegMAN")
if _SEGMAN_ROOT not in sys.path:
    sys.path.insert(0, _SEGMAN_ROOT)

from segman.mmscope_core import VSSMBlock  # noqa: E402

# 内部宽度等默认值集中在 rrs/model_mode.py::MM_DEFAULTS (那里不依赖 torch, 便于无 torch 的
# 模式测试导入), 依据见该处注释与 docs/MMSCopE与评分头接入说明_20261016.md。
# channels=128 取官方 SegMAN-T 解码器宽度 (segman_t_ade.py), 扫描通道为 3*channels=384;
# feat_proj_dim=256 与 enc_maps 的 D 相同。SS2D 的 d_state 不开放配置: 官方 VSSMBlock
# 不向 token_mixer 传 d_state, 固定用 VSSM 的默认值 1。
from .model_mode import MM_NORMS  # noqa: E402


def conv_norm_act(in_ch, out_ch, kernel_size=1, stride=1, padding=0, norm="gn", gn_groups=32,
                  act=True):
    """替代官方 ConvModule(norm_cfg=SyncBN, act_cfg=ReLU)。
    顺序与 ConvModule 默认 order=('conv','norm','act') 一致; 有 norm 时 conv 不带 bias
    (ConvModule bias='auto')。初始化复刻 mmcv ConvModule.init_weights 的默认行为。"""
    if norm == "gn":
        assert out_ch % gn_groups == 0, f"GroupNorm({gn_groups}) 要求通道数可整除, 当前 {out_ch}"
        norm_layer = nn.GroupNorm(gn_groups, out_ch)
    elif norm == "bn":
        norm_layer = nn.BatchNorm2d(out_ch)
    else:
        raise ValueError(f"mm.norm 必须是 {MM_NORMS} 之一, 当前 {norm!r}")
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    nn.init.constant_(norm_layer.weight, 1.0)
    nn.init.constant_(norm_layer.bias, 0.0)
    layers = [conv, norm_layer]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class StageProj(nn.Module):
    """官方 segman_decoder.py::MLP (SegFormer 式逐位置线性投影), 无 norm / 激活。"""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 1).reshape(n, -1, h, w)


class MMSCopEPixelDecoder(nn.Module):
    """官方 SegMAN 解码器的 MMSCopE 特征路径 + 本工程 1/4 细节融合。

    参数:
        c4_channels: 1/4 浅层特征 (本工程 c4) 的通道数
        enc_dim:     enc_maps 的通道数 (GroundingDINO hidden_dim, 256)
        out_dim:     输出通道 = mask_dim, 必须与 MaskHead 一致
        channels / feat_proj_dim / norm / gn_groups / scan_fp32 / drop_path: 见 MM_DEFAULTS
    """

    # 官方 forward_winssm 的三条路径都把 1/8 特征降到 1/32, 需要 1/8 尺寸能被 4 整除 (适配 A3)
    PAD_MULTIPLE = 4

    def __init__(self, c4_channels, enc_dim=256, out_dim=256, channels=128, feat_proj_dim=256,
                 norm="gn", gn_groups=32, scan_fp32=True, drop_path=0.0):
        super().__init__()
        self.embed_dim = channels
        self.feat_proj_dim = feat_proj_dim
        self.scan_fp32 = bool(scan_fp32)
        cna = lambda i, o, **kw: conv_norm_act(i, o, norm=norm, gn_groups=gn_groups, **kw
                                               )  # noqa: E731

        # ---- 官方 forward_mlp_decoder: 1/8、1/16、1/32 投影 + 融合到 1/8 ----
        # 官方 linear_c4/c3/c2 对应最深->较浅; 这里按分辨率一一对应 enc_maps[2]/[1]/[0]。
        self.linear_c32 = StageProj(enc_dim, feat_proj_dim)  # 官方 linear_c4
        self.linear_c16 = StageProj(enc_dim, feat_proj_dim)  # 官方 linear_c3
        self.linear_c8 = StageProj(enc_dim, feat_proj_dim)  # 官方 linear_c2
        self.linear_fuse = cna(feat_proj_dim * 3, channels)

        # ---- 官方 forward_winssm 的模块表 ----
        self.conv_downsample_2 = cna(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv_downsample_4 = cna(channels, channels * 4, kernel_size=5, stride=4, padding=1)
        self.reduce_channels = nn.ModuleList(
            [cna(channels * 4 * (2**i), channels) for i in range(3)])
        vssm_dim = channels * 3
        self.vssm = VSSMBlock(embed_dim=vssm_dim, expansion_ratio=1, channel_split=False,
                              drop_path=drop_path)
        self.short_path = cna(channels, channels)
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), cna(channels, channels))
        self.proj_out = cna(vssm_dim, feat_proj_dim)
        feat_concat_dim = channels * (2 + 3) + feat_proj_dim * 3
        self.cat = cna(feat_concat_dim, channels)

        # ---- 本项目适配 A1: 对齐到 c4 (1/4) 并与浅层特征融合 ----
        # 与 FPNPixelDecoder 末段同构: lateral(1x1, 带 bias) + 相加 + smooth(3x3, bias=False,
        # GroupNorm, ReLU) + out(1x1)。norm="gn" 时 cna 与原 FPN 的 smooth 逐层一致;
        # norm="bn" 时跟随 mm.norm (batch=1 会受 BN 本身限制)。
        self.lateral4 = nn.Conv2d(c4_channels, channels, 1)
        self.smooth4 = cna(channels, channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(channels, out_dim, 1)

    # ------------------------------------------------------------------
    # 官方 SegMANDecoder.forward_mlp_decoder (按原语句顺序改写; 官方 inputs[1] -> 本工程 1/8 特征)
    # 未做与官方的数值对照, 见 docs/MMSCopE与评分头接入说明_20261016.md 第 8.1 节
    # ------------------------------------------------------------------
    def forward_mlp_decoder(self, enc_maps):
        c8, c16, c32 = enc_maps
        _c32 = self.linear_c32(c32)
        _c16 = self.linear_c16(c16)
        _c8 = self.linear_c8(c8)

        size8 = c8.shape[2:]
        _c32 = F.interpolate(_c32, size=size8, mode="bilinear", align_corners=False).contiguous()
        _c16 = F.interpolate(_c16, size=size8, mode="bilinear", align_corners=False).contiguous()

        _c = self.linear_fuse(torch.cat([_c32, _c16, _c8], dim=1))
        return _c, _c8, _c16, _c32

    def _scan(self, multi_x):
        """官方 self.vssm(multi_x)。适配 A4: 默认在 fp32 下执行真实 SS2D 扫描。
        device_type 跟随输入, 便于 CPU 上做接线测试 (真实算子只能在 CUDA 上跑)。"""
        if not self.scan_fp32:
            return self.vssm(multi_x)
        with torch.autocast(device_type=multi_x.device.type, enabled=False):
            return self.vssm(multi_x.float())

    # ------------------------------------------------------------------
    # 官方 SegMANDecoder.forward_winssm (按原语句顺序改写, 未做数值对照)
    # 适配 A3: 仅在降采样/Pixel Unshuffle/扫描路径上 pad 到 4 的倍数, 算完裁回。
    # ------------------------------------------------------------------
    def forward_winssm(self, x, c8, c16, c32):
        out = [
            self.short_path(x),
            F.interpolate(self.image_pool(x), size=x.shape[2:], mode="bilinear",
                          align_corners=False).contiguous()
        ]

        h, w = x.shape[2:]
        m = self.PAD_MULTIPLE
        ph, pw = (-h) % m, (-w) % m
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x

        x_2 = self.conv_downsample_2(xp)  # 1/2 resolution
        x_4 = self.conv_downsample_4(xp)  # 1/4 resolution

        # unshuffle all features to size 1/4 resolution
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(xp, downscale_factor=4)

        # reduce channels
        x_unshuffle = self.reduce_channels[2](x_unshuffle)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_4 = self.reduce_channels[0](x_4)

        multi_x = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1)

        _out = self._scan(multi_x).to(x.dtype)

        _out = F.interpolate(_out, size=xp.shape[2:], mode="bilinear", align_corners=False)
        if ph or pw:
            _out = _out[..., :h, :w].contiguous()

        _out_ = self.proj_out(_out)
        c8 = c8 + _out_
        c16 = c16 + _out_
        c32 = c32 + _out_

        out += [_out, c8, c16, c32]

        return self.cat(torch.cat(out, dim=1))

    def forward(self, c4, enc_feats):
        """c4: (B,C4,H4,W4) 1/4 浅层特征; enc_feats: [1/8, 1/16, 1/32] 图文融合特征。"""
        x, c8, c16, c32 = self.forward_mlp_decoder(enc_feats)
        ctx = self.forward_winssm(x, c8, c16, c32)  # (B, channels, H8, W8)
        # 适配 A1: 上采样到 c4 的实际尺寸 (不用 scale_factor), 与浅层细节相加后整合
        lat = self.lateral4(c4)
        ctx = F.interpolate(ctx, size=lat.shape[-2:], mode="nearest")
        return self.out(self.smooth4(lat + ctx))
