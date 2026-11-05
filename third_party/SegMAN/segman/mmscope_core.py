"""SegMAN MMSCopE 所需的官方计算单元 (从官方 segman_decoder.py 抽取, 去掉 MMSeg 注册器依赖)。

来源 (见 third_party/SegMAN/VERSION.md):
  https://github.com/yunxiangfu2001/SegMAN
  commit 9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8
  segmentation/mmseg/models/decode_heads/segman_decoder.py
    SHA256 a163365fc0bd1a43b09ba4bbecbdd706b97b7bc775e32671be6225db50a095b5
  原文件完整副本保留在同目录 segman_decoder_official.py (仅作对照, 不被导入)。
  许可: Apache-2.0, 见 third_party/SegMAN/LICENSE。
  文件内 NVIDIA 版权声明随被抽取的代码一并保留 (见下)。

抽取的内容 (逐类对应官方同名类, 行为未改):
  SelectiveScanOflex  <- segman_decoder.py L366-384   (selective_scan_cuda_oflex 封装)
  LayerScale          <- L430-449
  LayerNorm2d         <- L461-469
  VSSM                <- L484-655   (SS2D, 官方 MMSCopE 实际调用的扫描模块)
  FFN                 <- L737-762
  VSSMBlock           <- L765-814   (token_mixer=VSSM, channel_mixer=FFN)
  CrossScanTriton / CrossMergeTriton <- 同目录 csm_triton.py (原样复制, 未改)

未抽取 (MMSCopE 路径不调用): RoPE / Attention (natten) / EfficientScan / EfficientMerge /
  cross_scan_continuous / get_continuous_paths / MLP / SegMANDecoder 本体 / fvcore flops 工具。
  SegMAN 编码器与 LASS 同样未抽取。

与官方的差异 (仅工程适配, 不改数值):
  1. `import selective_scan_cuda_oflex` 改为延迟导入: 缺少该扩展时在构造 VSSM 时抛出
     带安装命令的异常, 不退回任何近似实现 (本项目要求禁止假算子)。
  2. timm 的 DropPath 导入做了 timm.models.layers / timm.layers 两种路径兼容 (新版 timm 移动了位置)。
  3. 删除了官方文件中与 MMSCopE 无关的 import (mmcv / mmseg / natten / fvcore / einops.rearrange 等)。
     einops.repeat 仍为官方 A_log_init / D_init 所用, 保持原样。
"""
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# (上面的声明来自官方 segman_decoder.py 文件头, 随抽取的代码保留。)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from einops import repeat
    _EINOPS_ERR = None
except Exception as e:  # einops 未安装 (原 B0 环境不需要它)
    repeat = None
    _EINOPS_ERR = e

try:  # 新版 timm 把 DropPath 移到了 timm.layers
    from timm.models.layers import DropPath
    _TIMM_ERR = None
except Exception:  # pragma: no cover - 取决于服务器 timm 版本
    try:
        from timm.layers import DropPath
        _TIMM_ERR = None
    except Exception as e:
        DropPath = None
        _TIMM_ERR = e

try:
    from .csm_triton import CrossMergeTriton, CrossScanTriton
    _TRITON_ERR = None
except Exception as e:  # triton 缺失 / 不支持该平台 (例如 Windows)
    CrossScanTriton = CrossMergeTriton = None
    _TRITON_ERR = e

try:
    import selective_scan_cuda_oflex
    _OFLEX_ERR = None
except Exception as e:  # 扩展未编译
    selective_scan_cuda_oflex = None
    _OFLEX_ERR = e

INSTALL_HINT = (
    "MMSCopE 需要真实的 SS2D 算子, 当前环境缺少依赖:\n"
    "  {missing}\n"
    "请在 Linux GPU 服务器上安装 (见 scripts/setup_mmscope.sh):\n"
    "  pip install einops\n"
    "  cd third_party/SegMAN/kernels/selective_scan && pip install --no-build-isolation .\n"
    "  (triton 随 Linux 版 PyTorch 一并安装; Windows 无 triton, 无法启用 MMSCopE)\n"
    "本项目禁止用卷积/池化/注意力/Identity 冒充 selective scan, 因此这里直接报错。"
)


def require_scan_ops():
    """缺少真实算子时报错, 不做任何静默退化。VSSM 构造时调用。"""
    missing = []
    if selective_scan_cuda_oflex is None:
        missing.append(f"selective_scan_cuda_oflex: {_OFLEX_ERR!r}")
    if CrossScanTriton is None:
        missing.append(f"triton (csm_triton): {_TRITON_ERR!r}")
    if repeat is None:
        missing.append(f"einops: {_EINOPS_ERR!r}")
    if DropPath is None:
        missing.append(f"timm: {_TIMM_ERR!r}")
    if missing:
        raise ImportError(INSTALL_HINT.format(missing="\n  ".join(missing)))


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False,
                nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias,
                                                      delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()

        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias

        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class VSSM(nn.Module):
    """官方 segman_decoder.py::VSSM (SS2D)。MMSCopE 的扫描模块本体, 逐行照搬。"""

    def __init__(
        self,
        d_model=96,
        d_state=1,
        expansion_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        k_groups=4,
        **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        require_scan_ops()   # 适配: 缺少真实算子时立刻报错 (官方为模块级 import)
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.expansion_ratio = expansion_ratio
        if self.expansion_ratio != 1:
            self.xproj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True)  # (K * D)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True,
                        nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus,
                                        nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)

    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)

    def forward(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        xs = self._cross_scan(x)
        if self.expansion_ratio != 1:
            xs = self.xproj(xs.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2).contiguous()
        xs = xs.reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)

        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys = self._selective_scan(xs, dts, As, Bs,
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)

        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)

        if self.expansion_ratio != 1:
            y = self.yproj(y.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()

        if to_dtype:
            y = y.to(x.dtype)

        return y


class FFN(nn.Module):
    """官方 segman_decoder.py::FFN (VSSMBlock 的 channel mixer)。"""

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
    ):
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):

        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class VSSMBlock(nn.Module):
    """官方 segman_decoder.py::VSSMBlock。SegMANDecoder.forward_winssm 里的 self.vssm 就是它
    (以 embed_dim=3*channels, expansion_ratio=1, channel_split=False 构造)。"""

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2,
                 expansion_ratio=1,
                 channel_split=False,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 drop_path=0,
                 layerscale=False,
                 layer_init_values=1e-6,
                 token_mixer=VSSM,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d,
                 dropout=0.1):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.token_mixer = token_mixer(d_model=embed_dim,
                                       k_groups=4,
                                       expansion_ratio=expansion_ratio,
                                       channel_split=channel_split,)
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.mlp = channel_mixer(embed_dim, embed_dim * 4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x):
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))

        return x
