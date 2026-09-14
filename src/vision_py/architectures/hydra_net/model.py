import torch
import torch.nn as nn

from .backbone import Backbone
from .neck import Neck
from .heads import SegmentationHead, DetectionHead, SignAuthenticityHead


class HydraNet(nn.Module):
    def __init__(
        self,
        input_size=(544, 960),          # (H, W)
        backbone_widths=(32, 64, 128, 256, 512),
        backbone_depths=(1, 2, 3, 1),
        neck_ch=128,
        num_seg_classes=4,             
        num_obj_classes=5,              
        sign_roi_stride=8,              # pool sign-authenticity ROIs from P3
    ):
        super().__init__()
        self.input_size = input_size

        self.backbone = Backbone(in_channels=3, n_channels=backbone_widths, depths=backbone_depths)
        self.neck = Neck(self.backbone.out_channels, out_channels=neck_ch)

        self.seg_head = SegmentationHead(neck_ch, num_classes=num_seg_classes)
        self.det_head = DetectionHead(neck_ch, num_classes=num_obj_classes, strides=(8, 16, 32))
        self.sign_head = SignAuthenticityHead(neck_ch, feat_stride=sign_roi_stride)

    def forward(self, images, sign_boxes=None):
        """
        images: (B, 3, H, W) already resized/normalized to `self.input_size`
        sign_boxes: optional list[len B] of (K_i, 4) xyxy boxes (pixel space)
                    for stop-sign candidates to run through the authenticity
                    head. During training these are matched GT stop-sign
                    boxes; during inference they come from the detection
                    head's own stop-sign predictions after NMS.

        returns a dict with keys: 'lane_logits', 'detection', and
        (only if sign_boxes was given) 'sign_logits'.
        """
        h, w = images.shape[-2], images.shape[-1]
        c3, c4, c5 = self.backbone(images)
        p3, p4, p5 = self.neck(c3, c4, c5)

        seg_logits = self.seg_head(p3, out_size=(h, w))
        detection = self.det_head((p3, p4, p5))

        out = {"lane_logits": seg_logits, "detection": detection}

        if sign_boxes is not None:
            out["sign_logits"] = self.sign_head(p3, sign_boxes)

        return out

    @torch.no_grad()
    def count_parameters(self):
        return {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "neck": sum(p.numel() for p in self.neck.parameters()),
            "seg_head": sum(p.numel() for p in self.seg_head.parameters()),
            "det_head": sum(p.numel() for p in self.det_head.parameters()),
            "sign_head": sum(p.numel() for p in self.sign_head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }