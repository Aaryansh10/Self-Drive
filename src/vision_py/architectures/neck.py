import torch.nn as nn
import torch.nn.functional as F
from .modules import ConvBNSiLU, WeightedFusion

class Neck(nn.Module):

    def __init__(self, in_channels, out_channels=128):
        """
        in_channels: (C3_channels, C4_channels, C5_channels) from the backbone
        out_channels: unified channel width for P3/P4/P5
        """

        super().__init__()

        C3_channels, C4_channels, C5_channels = in_channels

        # lateral 1x1 convs to unify channel width
        self.lat_c3 = ConvBNSiLU(C3_channels, out_channels, k_size=1)
        self.lat_c4 = ConvBNSiLU(C4_channels, out_channels, k_size=1)
        self.lat_c5 = ConvBNSiLU(C5_channels, out_channels, k_size=1)

        # ---- FPN (top-down) ----
        self.fuse_p4_topdown = WeightedFusion(2, out_channels)
        self.fuse_p3_out = WeightedFusion(2, out_channels)

        # ---- PANet (bottom-up) ----
        self.down_p3 = ConvBNSiLU(out_channels, out_channels, k_size=3, stride=2)  # P3_out -> P4 resolution
        self.down_p4 = ConvBNSiLU(out_channels, out_channels, k_size=3, stride=2)  # P4_out -> P5 resolution
        self.fuse_p4_out = WeightedFusion(2, out_channels)
        self.fuse_p5_out = WeightedFusion(2, out_channels)

        self.out_channels = out_channels

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, c3, c4, c5):
        p5 = self.lat_c5(c5)
        p4_lat = self.lat_c4(c4)
        p3_lat = self.lat_c3(c3)
 
        # top-down (FPN)
        p4_td = self.fuse_p4_topdown([p4_lat, self._upsample_to(p5, p4_lat)])
        p3_out = self.fuse_p3_out([p3_lat, self._upsample_to(p4_td, p3_lat)])
 
        # bottom-up (PANet)
        p4_out = self.fuse_p4_out([p4_td, self.down_p3(p3_out)])
        p5_out = self.fuse_p5_out([p5, self.down_p4(p4_out)])
 
        return p3_out, p4_out, p5_out  # strides 8, 16, 32
