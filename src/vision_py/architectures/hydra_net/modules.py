import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):

    def __init__(
            self, in_channels, out_channels, k_size=3,
            stride=1, groups=1, padding=None, h_swish=False
        ):
        super().__init__()

        if padding == None:
            if k_size % 2 == 1:
                padding = k_size // 2
            else:
                padding = (k_size + 1) // 2 

        self.conv = nn.Conv2d(in_channels, out_channels, k_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

        if h_swish:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ECABlock(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        temp = int(abs((math.log2(channels) + b) / gamma)) 
        k_size = temp if temp % 2 else temp + 1
        k_size = max(k_size, 3)
        padding = k_size // 2

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):

    def __init__(self, in_channels, skip_connection=True, expansion=0.5):
        super().__init__()

        intermediate_channels = int(in_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(intermediate_channels, in_channels, kernel_size=3)
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.skip_connection else out

class CSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks=1, skip_connection=True):
        super().__init__()

        intermediate_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, 2 * intermediate_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            Bottleneck(intermediate_channels, skip_connection) for _ in range(n_blocks)
        )
        self.cv2 = ConvBNSiLU((2 + n_blocks) * intermediate_channels, out_channels, k_size=1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
            
        return self.cv2(torch.cat(y, dim=1))

class DownsampleCSP(nn.Module):
 
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()

        self.downsample = ConvBNSiLU(in_channels, out_channels, k_size=3, stride=2)
        self.csp = CSPBlock(out_channels, out_channels, n_blocks=n)
        self.eca = ECABlock(out_channels)
 
    def forward(self, x):
        x = self.downsample(x)
        x = self.csp(x)
        x = self.eca(x)
        return x

class WeightedFusion(nn.Module):

    def __init__(self, num_inputs, channels, eps=1e-4):
        super().__init__()

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.fuse_conv = ConvBNSiLU(channels, channels, k_size=3)

    def forward(self, feature_maps):
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)

        out = sum(w[i] * feature_maps[i] for i in range(len(feature_maps)))
        return self.fuse_conv(out)