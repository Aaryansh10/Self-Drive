import torch.nn as nn
from .modules import ConvBNSiLU, DownsampleCSP

class Backbone(nn.Module):

    def __init__(self, in_channels, n_channels=(32, 64, 128, 256, 512), depths=(1, 2, 3, 1)):
        """
        widths: (stem, C2, C3, C4, C5)
        depths: number of CSP bottlenecks in (C1, C2, C3, C4)
        """

        super().__init__()

        n0, n1, n2, n3, n4 = n_channels
        d1, d2, d3, d4 = depths

        # Stem: stride 2, e.g. 960x544 -> 480x272
        self.stem = ConvBNSiLU(in_channels, n0, k_size=3, stride=2)

        # Stage1 / C2: stride 4 total, 480x272 -> 240x136 (feature not exposed to neck)
        self.stage1 = DownsampleCSP(n0, n1, n_blocks=d1)

        # Stage2 / C3: stride 8 total, 240x136 -> 120x68
        self.stage2 = DownsampleCSP(n1, n2, n_blocks=d2)

        # Stage3 / C4: stride 16 total, 120x68 -> 60x34
        self.stage3 = DownsampleCSP(n2, n3, n_blocks=d3)
 
        # Stage4 / C5: stride 32 total, 60x34 -> 30x17
        self.stage4 = DownsampleCSP(n3, n4, n_blocks=d4)
 
        self.out_channels = (n2, n3, n4)  # channels of C3, C4, C5

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5
