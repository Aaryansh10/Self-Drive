"""
The three HydraNet task heads.

1. SegmentationHead      - P3 -> bilinear upsample + conv, twice -> per-pixel
                            segmentation logits at full input resolution.
2. DetectionHead         - anchor-free, decoupled cls / reg / centerness
                            branches, weight-shared across P3/P4/P5 (FCOS /
                            YOLOX style) so obstacle detection stays cheap.
3. SignAuthenticityHead  - ROIAlign on a proposed stop-sign box followed by a
                            tiny conv classifier -> real vs fake logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .modules import ConvBNSiLU
from .utils import generate_grid_points, decode_ltrb

class SegmentationHead(nn.Module):
    """
    Upsamples the (stride-8) P3 feature map back towards input resolution
    using bilinear interpolation + regular convolution at each step, ending
    in a per-pixel classification map (lane lines / stop lines / background).
    """

    def __init__(self, in_ch, num_classes=4, mid_ch=64):
        super().__init__()

        # stride 8 -> stride 4
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, k_size=3)

        # stride 4 -> stride 2
        self.conv2 = ConvBNSiLU(mid_ch, mid_ch, k_size=3)

        # stride 2 -> stride 1
        self.conv3 = ConvBNSiLU(mid_ch, mid_ch // 2, k_size=3)

        self.classifier = nn.Conv2d(mid_ch // 2, num_classes, kernel_size=1)

    def forward(self, p3, out_size):
        """out_size: (H, W) of the original input image."""
        x = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv3(x)

        logits = self.classifier(x)
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

        return logits  # (B, num_classes, H, W)


class _ScaleExp(nn.Module):
    """Learnable per-level scalar applied before exp(), FCOS-style, so the
    shared regression tower can still predict different box scales at each
    pyramid level."""

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class DetectionHead(nn.Module):
    """
    Anchor-free, decoupled object-detection head, weight-shared across all
    FPN/PAN levels (P3/P4/P5) to keep parameter count and compute low.

    - classification branch: predicts per-class probability at every point
    - regression branch: predicts (l, t, r, b) distances to the box edges
    - centerness branch: branches off the regression tower, predicts how
      close a point is to the true object center (down-weights low-quality
      boxes far from the object center at inference/NMS time)
    """

    def __init__(self, in_ch, num_classes, strides=(8, 16, 32), mid_ch=96, num_convs=2):
        super().__init__()

        self.num_classes = num_classes
        self.strides = strides

        self.cls_tower = nn.Sequential(
            *[ConvBNSiLU(in_ch if i == 0 else mid_ch, mid_ch, k_size=3) for i in range(num_convs)]
        )
        self.reg_tower = nn.Sequential(
            *[ConvBNSiLU(in_ch if i == 0 else mid_ch, mid_ch, k_size=3) for i in range(num_convs)]
        )

        self.cls_pred = nn.Conv2d(mid_ch, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(mid_ch, 4, kernel_size=1)
        self.centerness_pred = nn.Conv2d(mid_ch, 1, kernel_size=1)

        # one learnable scale per pyramid level, since the tower is shared
        self.scales = nn.ModuleList([_ScaleExp(1.0) for _ in strides])

        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob)).item()
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(self, feats):
        """feats: (P3, P4, P5). Returns per-level raw predictions plus
        decoded boxes in input-pixel space."""

        cls_outs, reg_outs, ctr_outs, points_all, strides_all = [], [], [], [], []

        for level, (feat, stride, scale) in enumerate(zip(feats, self.strides, self.scales)):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_logits = self.cls_pred(cls_feat)              # (B, C, H, W)
            reg_raw = scale(self.reg_pred(reg_feat))            # (B, 4, H, W)
            reg_dist = F.relu(reg_raw) * stride                 # distances in pixels, >= 0
            centerness = self.centerness_pred(reg_feat)          # (B, 1, H, W)

            b, _, h, w = cls_logits.shape
            points = generate_grid_points(h, w, stride, feat.device, feat.dtype)  # (H*W, 2)

            cls_outs.append(cls_logits.permute(0, 2, 3, 1).reshape(b, h * w, self.num_classes))
            reg_outs.append(reg_dist.permute(0, 2, 3, 1).reshape(b, h * w, 4))
            ctr_outs.append(centerness.permute(0, 2, 3, 1).reshape(b, h * w, 1))
            points_all.append(points)
            strides_all.append(torch.full((h * w,), stride, device=feat.device, dtype=feat.dtype))

        cls_all = torch.cat(cls_outs, dim=1)          # (B, N, num_classes)
        reg_all = torch.cat(reg_outs, dim=1)           # (B, N, 4) ltrb distances (pixels)
        ctr_all = torch.cat(ctr_outs, dim=1)           # (B, N, 1)
        points_all = torch.cat(points_all, dim=0)      # (N, 2)

        boxes = decode_ltrb(points_all.unsqueeze(0), reg_all)  # (B, N, 4) xyxy pixels

        return {
            "cls_logits": cls_all,
            "reg_dist": reg_all,
            "centerness": ctr_all,
            "boxes": boxes,
            "points": points_all,
        }


class SignAuthenticityHead(nn.Module):
    """
    Given a stop-sign box proposed by the DetectionHead, crop a fixed-size
    feature patch from the neck with ROIAlign, then run a tiny classifier
    (a couple of conv layers + global average pool + a linear layer) to
    decide real vs. fake (e.g. a sticker/graffiti-altered or adversarial
    stop sign).
    """

    def __init__(self, in_ch, roi_size=7, mid_ch=64, num_classes=2, feat_stride=8):
        super().__init__()
        self.roi_size = roi_size
        self.feat_stride = feat_stride  # which neck level (by stride) to pool from
        self.conv1 = ConvBNSiLU(in_ch, mid_ch, k_size=3)
        self.conv2 = ConvBNSiLU(mid_ch, mid_ch, k_size=3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mid_ch, num_classes)

    def forward(self, feat_map, boxes):
        """
        feat_map: (B, C, h, w) neck feature map with stride == self.feat_stride
                  (P3 by default: fine detail is what distinguishes a
                  genuine printed sign from a tampered/fake one)
        boxes: list of length B, each a (K_i, 4) tensor of xyxy boxes in
               original input-pixel space (stop-sign proposals for that image)
        returns: (sum_i K_i, num_classes) logits, in the same flattened
                 order as torch.cat(boxes)
        """

        spatial_scale = 1.0 / self.feat_stride
        pooled = roi_align(
            feat_map, boxes, output_size=self.roi_size,
            spatial_scale=spatial_scale, aligned=True,
        )  # (sum_i K_i, C, roi, roi)

        if pooled.shape[0] == 0:
            return pooled.new_zeros((0, self.fc.out_features))
        
        x = self.conv1(pooled)
        x = self.conv2(x)
        x = self.gap(x).flatten(1)

        return self.fc(x)