import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from ultralytics_mod.nn.modules.coordatt import EnhancedCoordAttV2


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = kernel_size // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class SPDConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1*4, c2, k, s, k//2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.bn(self.conv(x)))


class AA_SPDConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        # AA-SPD 的抗混叠预滤波
        # self.pre_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 均值池化
        weights = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]).float() / 16.0
        self.pre_blur = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False)
        self.pre_blur.weight.data = weights.repeat(c1, 1, 1, 1)  # 深度可分
        self.pre_blur.weight.requires_grad = False  # 固定权重，轻量
        self.unshuffle = nn.PixelUnshuffle(2)

        # PhaseNorm 的分相位 BN
        self.bn_phase = nn.ModuleList([nn.BatchNorm2d(c1) for _ in range(4)])

        self.conv = nn.Conv2d(c1*4, c2, k, s, k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # x = self.pre_pool(x)
        x = self.pre_blur(x)
        x = self.unshuffle(x)  # [B, 4C, H/2, W/2]
        # 插入分相位归一化
        chunks = [bn(t) for bn, t in zip(self.bn_phase, torch.chunk(x, 4, dim=1))]
        x = torch.cat(chunks, dim=1)
        return self.act(self.bn(self.conv(x)))


class UIB(nn.Module):
    """改进的 UIB 模块，支持 Partial Residual"""
    def __init__(self, inp, oup, start_dw_kernel_size=3, middle_dw_kernel_size=3,
                 middle_dw_downsample=False, stride=1, expand_ratio=2):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.add = inp == oup

        expand_filters = int(inp * expand_ratio)

        # 起始深度卷积
        if start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp)

        # 扩展阶段
        self._expand = conv_2d(inp, expand_filters, 1)

        # 中间深度卷积
        if middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
                                      

        # 投影阶段
        self._proj = conv_2d(expand_filters, oup, 1)

        # SE
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(inp, inp // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // 16, inp, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        x0 = x

        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        x = self._proj(x)

        if self.add:
            b, c, _, _ = x0.size()
            g = self.avg_pool(x0).view(b, c)  # 全局平均池化并展平
            g = self.gate(g).view(b, c, 1, 1)  # 生成权重
            x = g * (x0 + x) + (1 - g) * x  # 残差逻辑

        return x


class MultiBranchUIB(nn.Module):
    """串联多层 UIB + Partial Residual + 拼接压缩 + CoordAtt"""
    def __init__(self, inp, oup, n=2, e=0.5, kernel_size=[[3, 3, 2], [3, 3, 2]]):
        super().__init__()
        self.c = int(oup * e)  # hidden channels
        self.cv1 = conv_2d(inp, 2 * self.c, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.submodules = nn.ModuleList(
            UIB(self.c, self.c,
                start_dw_kernel_size=kernel_size[i][0],
                middle_dw_kernel_size=kernel_size[i][1],
                expand_ratio=kernel_size[i][2])
            for i in range(n)
        )
        self.cv2 = conv_2d((3 + n) * self.c, oup, 1)
        self.encooradatt = EnhancedCoordAttV2(oup, oup)

    def forward(self, x):
        x = self.cv1(x)
        y = list(x.chunk(2, 1))
        y.append(self.maxpool(y[0]))
        for m in self.submodules:
            y.append(m(y[-1]))
        x = torch.cat(y, 1)
        x = self.cv2(x)
        x = self.encooradatt(x)
        return x
