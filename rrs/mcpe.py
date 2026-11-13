"""MCPE：方法第 3.3 节公式 (1)-(5)，输出供六组候选共享的像素特征。"""
import torch
from torch import nn
import torch.nn.functional as F

from .selective_scan import ScanBlock


def conv_norm_act(in_ch, out_ch, kernel_size=1, stride=1, padding=0, norm='gn', gn_groups=32,
                  act=True):
    if norm == 'gn':
        if out_ch % gn_groups or out_ch // gn_groups < 2:
            raise ValueError('GroupNorm 每组至少需要两个通道，以支持 batch=1 的全局池化分支')
        normalization = nn.GroupNorm(gn_groups, out_ch)
    elif norm == 'bn':
        normalization = nn.BatchNorm2d(out_ch)
    else:
        raise ValueError("norm 必须为 'gn' 或 'bn'")
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    return nn.Sequential(conv, normalization, *([nn.ReLU(inplace=True)] if act else []))


class StageProj(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class MCPEPixelDecoder(nn.Module):
    """独立投影、三路重编码、联合扫描、上下文重注入和浅层融合。

    默认各 stage 投影为 256 维，三路各 128 维，扫描输入 384 维，
    六路聚合输入 1408 维，最终输出 256 维、分辨率与 shallow 相同。
    """
    PAD_MULTIPLE = 4

    def __init__(self, c4_channels, enc_dim=256, out_dim=256, channels=128, feat_proj_dim=256,
                 norm='gn', gn_groups=32, scan_fp32=True, drop_path=0.0):
        super().__init__()
        self.scan_fp32 = bool(scan_fp32)

        def cna(i, o, **kwargs):
            return conv_norm_act(i, o, norm=norm, gn_groups=gn_groups, **kwargs)

        self.linear_c8 = StageProj(enc_dim, feat_proj_dim)
        self.linear_c16 = StageProj(enc_dim, feat_proj_dim)
        self.linear_c32 = StageProj(enc_dim, feat_proj_dim)
        self.linear_fuse = cna(3 * feat_proj_dim, channels)
        self.conv_downsample_2 = cna(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv_downsample_4 = cna(channels, channels * 4, kernel_size=5, stride=4, padding=1)
        self.reduce_channels = nn.ModuleList(
            [cna(channels * factor, channels) for factor in (16, 8, 4)])
        self.vssm = ScanBlock(3 * channels, drop_path=drop_path)
        self.short_path = cna(channels, channels)
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), cna(channels, channels))
        self.proj_out = cna(channels * 3, feat_proj_dim)
        self.cat = cna(channels * 5 + feat_proj_dim * 3, channels)
        self.lateral4 = nn.Conv2d(c4_channels, channels, 1)
        self.smooth4 = cna(channels, channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(channels, out_dim, 1)

    def forward_mlp_decoder(self, enc_maps):
        c8, c16, c32 = enc_maps
        a8 = self.linear_c8(c8)
        a16 = F.interpolate(self.linear_c16(c16), size=c8.shape[-2:], mode='bilinear',
                            align_corners=False)
        a32 = F.interpolate(self.linear_c32(c32), size=c8.shape[-2:], mode='bilinear',
                            align_corners=False)
        feature = self.linear_fuse(torch.cat((a32, a16, a8), dim=1))
        return feature, a8, a16, a32

    def _scan(self, z):
        if not self.scan_fp32:
            return self.vssm(z)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.vssm(z.float())

    def forward_winssm(self, feature, a8, a16, a32):
        height, width = feature.shape[-2:]
        ph, pw = (-height) % 4, (-width) % 4
        padded = F.pad(feature, (0, pw, 0, ph), mode='replicate') if ph or pw else feature
        paths = (F.pixel_unshuffle(padded, 4), F.pixel_unshuffle(self.conv_downsample_2(padded),
                                                                 2), self.conv_downsample_4(padded))
        z = torch.cat([projection(path) for projection, path in zip(self.reduce_channels, paths)],
                      dim=1)
        u = self._scan(z).to(feature.dtype)
        u = F.interpolate(u, size=padded.shape[-2:], mode='bilinear', align_corners=False)
        u = u[..., :height, :width].contiguous()
        injected = self.proj_out(u)
        global_feature = F.interpolate(self.image_pool(feature), size=(height, width),
                                       mode='bilinear', align_corners=False)
        return self.cat(
            torch.cat((self.short_path(feature), global_feature, u, a8 + injected, a16 + injected,
                       a32 + injected), dim=1))

    def forward(self, shallow, enc_feats):
        feature, a8, a16, a32 = self.forward_mlp_decoder(enc_feats)
        context = self.forward_winssm(feature, a8, a16, a32)
        context = F.interpolate(context, size=shallow.shape[-2:], mode='nearest')
        return self.out(self.smooth4(self.lateral4(shallow) + context))
