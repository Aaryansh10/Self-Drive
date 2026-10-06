"""
Helper functions for the anchor-free detection head: grid-point generation and
center+distance -> xyxy box decoding (FCOS/YOLOX style).
"""
import torch

def generate_grid_points(feat_h, feat_w, stride, device, dtype=torch.float32):
    """Return (H*W, 2) tensor of (x, y) pixel-space coordinates for the
    center of every cell of a feature map at the given stride."""
    ys, xs = torch.meshgrid(
        torch.arange(feat_h, device=device, dtype=dtype),
        torch.arange(feat_w, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = (xs + 0.5) * stride
    ys = (ys + 0.5) * stride
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # (H*W, 2)


def decode_ltrb(points, ltrb):
    """
    points: (N, 2) grid center coordinates (x, y) in pixel space
    ltrb:   (..., N, 4) predicted distances (left, top, right, bottom), >= 0
    returns (..., N, 4) boxes in xyxy pixel space
    """
    x = points[..., 0]
    y = points[..., 1]
    l, t, r, b = ltrb[..., 0], ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]
    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b
    return torch.stack([x1, y1, x2, y2], dim=-1)