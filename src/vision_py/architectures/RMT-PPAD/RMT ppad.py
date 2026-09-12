"""
Phase 3 — Master Wrapper for RMT-PPAD

Pipeline:
    RMTPPADFeatureExtractor (Backbone -> proj -> AIFI -> CCFM) -> split into 2 branches
        detection branch:     3x GCA (one per scale) -> RTDETRHead
        segmentation branch:  3x GCA (one per scale) -> SegmentationHead

Phase 1 (rmt_ppad_phase1.py — rename import below to match your actual filename):
    - RMTPPADFeatureExtractor(x) -> [p3, p4, p5]   ResNet18Backbone -> 1x1 channel
      projection -> AIFI (on S5 only) -> CCFM, all bundled into one module. All
      three output scales share `hidden_dim` channels (default 256).

Phase 2, still assumed (adjust import path as needed):
    - GCA(channels) -> module; GCA_instance(x) -> x  gated cross-scale/channel attention,
                                                     one instantiated per scale per branch
"""

import torch
import torch.nn as nn

from RMT_ppad_phase_1 import RMTPPADFeatureExtractor  # Phase 1 — rename to your actual filename
from GCA import GCA                                   # Phase 2
from Task_heads import RTDETRHead, SegmentationHead


class RMTPPAD(nn.Module):
    def __init__(self, hidden_dim=256, pretrained_backbone=False,
                 aifi_heads=8, aifi_layers=1,
                 scale_channels=(256, 256, 256), d_model=256):
        super().__init__()

        self.trunk = RMTPPADFeatureExtractor(
            hidden_dim=hidden_dim,
            pretrained_backbone=pretrained_backbone,
            aifi_heads=aifi_heads,
            aifi_layers=aifi_layers,
        )

        # 3 GCA modules per branch, one per fused scale (p3, p4, p5)
        self.det_gca = nn.ModuleList([GCA(c) for c in scale_channels])
        self.seg_gca = nn.ModuleList([GCA(c) for c in scale_channels])

        self.det_head = RTDETRHead(d_model=d_model)
        self.seg_head = SegmentationHead()

        # fuse the 3 GCA'd scales into one map per branch before the heads.
        # 1x1 convs bring every scale to a common channel dim, then we
        # upsample to the highest-res scale and sum.
        self.det_fuse = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in scale_channels])
        self.seg_fuse = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in scale_channels])

    def _merge(self, fs, convs):
        # fs: list of 3 (B, C, H, W) maps, high-res -> low-res (p3, p4, p5)
        t0 = fs[0].shape[-2:]
        out = 0
        for f, cv in zip(fs, convs):
            x = cv(f)
            if x.shape[-2:] != t0:
                x = nn.functional.interpolate(x, size=t0, mode="bilinear", align_corners=False)
            out = out + x
        return out

    def forward(self, im):
        p3, p4, p5 = self.trunk(im)
        ps = [p3, p4, p5]

        # --- detection branch ---
        dps = [g(p) for g, p in zip(self.det_gca, ps)]
        df = self._merge(dps, self.det_fuse)
        det_out = self.det_head(df)

        # --- segmentation branch ---
        sps = [g(p) for g, p in zip(self.seg_gca, ps)]
        sf = self._merge(sps, self.seg_fuse)
        seg_out = self.seg_head(sf)

        return {"detection": det_out, "segmentation": seg_out}

    # --- convenience accessors for the gradient-routing training loop ---
    def shared_parameters(self):
        return list(self.trunk.parameters())

    def detection_parameters(self):
        return list(self.det_gca.parameters()) + list(self.det_fuse.parameters()) + list(self.det_head.parameters())

    def segmentation_parameters(self):
        return list(self.seg_gca.parameters()) + list(self.seg_fuse.parameters()) + list(self.seg_head.parameters())