"""MCPE 的可微二维选择性状态空间扫描，无额外 CUDA 扩展。

四个序列为行优先、列优先及其逆序。每个方向使用输入相关的 delta/B/C，
递推 h[t]=exp(delta[t]*A)*h[t-1]+delta[t]*B[t]*u[t]，y[t]=C[t]*h[t]+D*u[t]。
状态维数取 1；用分块仿射前缀扫描求解递推，避免累乘除法造成数值溢出。
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def affine_scan(a, b, chunk_size=64):
    """返回零初态递推的全部状态；a/b 形状相同，最后一维为时间。

    块内合成仿射变换 (a,b)∘(c,d)=(a*c,b+a*d)，块间传递最后状态。
    无输入相关的原地写入；autograd 可穿过全部时间步和块边界。
    """
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] == 0:
        raise ValueError("a/b 必须形状相同且时间维非空")
    if chunk_size < 1:
        raise ValueError("chunk_size 必须大于 0")
    state = torch.zeros_like(b[..., 0])
    outputs = []
    for start in range(0, a.shape[-1], chunk_size):
        ac = a[..., start:start + chunk_size]
        bc = b[..., start:start + chunk_size]
        offset = 1
        while offset < ac.shape[-1]:
            next_b = torch.cat(
                (bc[..., :offset], bc[..., offset:] + ac[..., offset:] * bc[..., :-offset]), -1)
            next_a = torch.cat((ac[..., :offset], ac[..., offset:] * ac[..., :-offset]), -1)
            ac, bc = next_a, next_b
            offset *= 2
        values = bc + ac * state.unsqueeze(-1)
        state = values[..., -1]
        outputs.append(values)
    return torch.cat(outputs, -1)


class SS2D(nn.Module):
    """四方向选择性扫描。通道保持不变，不以卷积或注意力替代状态递推。"""

    def __init__(self, dim, dt_rank=None):
        super().__init__()
        rank = dt_rank or math.ceil(dim / 16)
        self.dim, self.rank = dim, rank
        self.in_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.x_proj_weight = nn.Parameter(torch.empty(4, rank + 2, dim))
        self.dt_projs_weight = nn.Parameter(torch.empty(4, dim, rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, dim))
        self.A_logs = nn.Parameter(torch.zeros(4, dim))
        self.Ds = nn.Parameter(torch.ones(4, dim))
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.uniform_(self.dt_projs_weight, -rank**-0.5, rank**-0.5)
        with torch.no_grad():
            dt = torch.exp(torch.empty(4, dim).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_projs_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        batch, channels, height, width = x.shape
        u = F.silu(self.local(self.in_proj(x)))
        row = u.flatten(2)
        col = u.transpose(2, 3).contiguous().flatten(2)
        seq = torch.stack((row, col, row.flip(-1), col.flip(-1)), dim=1)
        projected = torch.einsum('bkcl,krc->bkrl', seq, self.x_proj_weight)
        dt_low, bcoef, ccoef = torch.split(projected, (self.rank, 1, 1), dim=2)
        delta = F.softplus(
            torch.einsum('bkrl,kcr->bkcl', dt_low, self.dt_projs_weight) +
            self.dt_projs_bias[None, :, :, None])
        decay = torch.exp(-delta * self.A_logs.exp()[None, :, :, None])
        drive = delta * bcoef * seq
        states = affine_scan(decay, drive)
        y = ccoef * states + self.Ds[None, :, :, None] * seq
        forward_row = y[:, 0].reshape(batch, channels, height, width)
        forward_col = y[:, 1].reshape(batch, channels, width, height).transpose(2, 3)
        backward_row = y[:, 2].flip(-1).reshape(batch, channels, height, width)
        backward_col = y[:, 3].flip(-1).reshape(batch, channels, width, height).transpose(2, 3)
        merged = forward_row + forward_col + backward_row + backward_col
        merged = self.out_norm(merged.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return self.out_proj(merged)


class ScanBlock(nn.Module):
    """深度卷积位置编码 + 归一化/SS2D 残差 + 归一化/FFN 残差。"""

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.drop_path = float(drop_path)
        self.position = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = SS2D(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Conv2d(dim * 4, dim, 1))

    @staticmethod
    def _norm(x, layer):
        return layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def _drop(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = 1 - self.drop_path
        mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x):
        x = x + self.position(x)
        x = x + self._drop(self.token_mixer(self._norm(x, self.norm1)))
        return x + self._drop(self.ffn(self._norm(x, self.norm2)))
