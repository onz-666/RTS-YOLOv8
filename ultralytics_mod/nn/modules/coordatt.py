import torch
import torch.nn as nn

import math


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直池化
        mip = max(8, inp // reduction)  # 中间通道数
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.ReLU6()  # 与 MobileNetV4 一致
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).transpose(2, 3)  # [N, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)
        # 动态计算分割大小
        split_h = h
        split_w = y.size(2) - h  # 剩余部分（通常为 W）
        x_h, x_w = torch.split(y, [split_h, split_w], dim=2)
        x_w = x_w.transpose(2, 3)
        a_h = self.conv_h(x_h).sigmoid()  # [N, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [N, C, 1, W]
        out = identity * a_h * a_w
        return out
    

class EnhancedCoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32, dilation=2):
        super(EnhancedCoordAtt, self).__init__()
        # 多尺度池化：支持不同池化核大小以扩展感受野
        self.pool_h = nn.ModuleList([
            nn.AdaptiveAvgPool2d((None, 1)),  # 水平池化
            nn.AdaptiveAvgPool2d((None, 2))   # 多尺度水平池化
        ])
        self.pool_w = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, None)),  # 垂直池化
            nn.AdaptiveAvgPool2d((2, None))   # 多尺度垂直池化
        ])
        mip = max(8, inp // reduction)  # 中间通道数
        # 空洞卷积替换 1x1 卷积，扩展感受野
        self.conv1 = nn.Conv2d(inp * 3, mip, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # 使用 h-swish 替代 ReLU6，提升表达能力
        # ECA 模块增强通道交互
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Conv1d(1, 1, 1),  # 1D Conv on channels
            nn.Sigmoid()
        )
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 多尺度池化
        x_h = [pool(x) for pool in self.pool_h]  # [N, C, H, 1], [N, C, H, 2]
        x_w = [pool(x).transpose(2, 3) for pool in self.pool_w]  # [N, C, W, 1], [N, C, W, 2]

        x_h_1 = x_h[1][:, :, :, 0:1]  # [N, C, H, 1]，左侧区域
        x_h_2 = x_h[1][:, :, :, 1:2]  # [N, C, H, 1]，右侧区域
        x_w_1 = x_w[1][:, :, :, 0:1]
        x_w_2 = x_w[1][:, :, :, 1:2]

        # 融合多尺度特征
        x_h = torch.cat([x_h[0], x_h_1, x_h_2], dim=1)  # [N, 3C, H, 1]
        x_w = torch.cat([x_w[0], x_w_1, x_w_2], dim=1)  # [N, 3C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [N, 3C, H+W, 1]
        
        # 空洞卷积处理
        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)
        
        # SE 模块增强通道交互
        se_weight = self.se(y)
        y = y * se_weight
        
        # 动态分割
        split_h = h
        split_w = y.size(2) - h
        x_h, x_w = torch.split(y, [split_h, split_w], dim=2)
        x_w = x_w.transpose(2, 3)
        
        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # [N, oup, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [N, oup, 1, W]
        
        # 应用注意力权重
        out = identity * a_h * a_w
        return out


class EnhancedCoordAttV2(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(EnhancedCoordAttV2, self).__init__()
        # 多尺度池化
        self.pool_h = nn.ModuleList([
            nn.AdaptiveAvgPool2d((None, 2)),  # 水平池化
            nn.AdaptiveMaxPool2d((None, 1))   # 多尺度水平池化
        ])
        self.pool_w = nn.ModuleList([
            nn.AdaptiveAvgPool2d((2, None)),  # 垂直池化
            nn.AdaptiveMaxPool2d((1, None))   # 多尺度垂直池化
        ])

        mip = max(8, inp // reduction)  # 中间通道数

        self.conv1 = nn.Conv2d(inp * 3, mip, kernel_size=1)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        # ECA 模块增强通道交互
        t = int(abs((math.log2(mip) / 2.0) + 1))
        k = t if t % 2 == 1 else t + 1
        k = max(1, k)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.eca_sigmoid = nn.Sigmoid()

        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)

    def forward(self, x):
        n, c, h, w = x.size()
        
        x_h = [pool(x) for pool in self.pool_h]  # [N, C, H, 2], [N, C, H, 1]
        x_w = [pool(x).transpose(2, 3) for pool in self.pool_w]  # [N, C, W, 2], [N, C, W, 1]

        # 融合多尺度特征
        x_h = torch.cat([x_h[0][:,:,:,:1], x_h[0][:,:,:,1:], x_h[1]], dim=1)  # [N, 3C, H, 1]
        x_w = torch.cat([x_w[0][:,:,:,:1], x_w[0][:,:,:,1:], x_w[1]], dim=1)  # [N, 3C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [N, 3C, H+W, 1]
        
        y = self.act(self.bn(self.conv1(y)))
        
        # ECA
        y_gap = nn.functional.adaptive_avg_pool2d(y, 1)  # (B, mip, 1, 1)
        y_gap = y_gap.view(n, 1, -1)                     # (B, 1, mip)
        y_eca = self.eca_conv(y_gap)                     # (B, 1, mip)
        y_eca = self.eca_sigmoid(y_eca).view(n, -1, 1, 1) # (B, mip, 1, 1)
        y = y * y_eca
        
        # 分割特征
        split_h = h
        split_w = y.size(2) - h
        x_h, x_w = torch.split(y, [split_h, split_w], dim=2)
        x_w = x_w.transpose(2, 3)
        
        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # [N, oup, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [N, oup, 1, W]
        
        # 应用注意力权重 用se增强通道交互
        out = x * a_h * a_w
        return out