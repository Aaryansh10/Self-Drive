import cv2
import numpy as np
import torch
import torch.nn.functional as F

def _build_lut():
    """lut[step][code] -> True if a foreground pixel with this 8-neighbour code is deletable.
    Bit order (P2..P9 clockwise): N=1, NE=2, E=4, SE=8, S=16, SW=32, W=64, NW=128."""
    lut = torch.zeros(2, 256, dtype=torch.bool)
    for code in range(256):
        n = [(code >> i) & 1 for i in range(8)]
        N, NE, E, SE, S, SW, W, NW = n
        B = sum(n)
        A = sum(1 for i in range(8) if n[i] == 0 and n[(i + 1) % 8] == 1)
        if 2 <= B <= 6 and A == 1:
            lut[0, code] = not (N and E and S) and not (E and S and W)
            lut[1, code] = not (N and E and W) and not (N and S and W)
    return lut


_W = torch.tensor([[128, 1, 2],
                   [64,  0, 4],
                   [32, 16, 8]], dtype=torch.float32).view(1, 1, 3, 3)
_LUT = _build_lut()

# Staircase corners: exactly two neighbours that are orthogonal to each other
# (N+E, E+S, S+W, W+N). The two neighbours are already diagonally connected,
# so the centre pixel is redundant. Zhang-Suen keeps these (A == 2).
_CORNERS = (1 | 4, 4 | 16, 16 | 64, 64 | 1)

_cache = {}


def _consts(device):
    if device not in _cache:
        _cache[device] = (_W.to(device), _LUT.to(device))
    return _cache[device]


@torch.no_grad()
def zhang_suen_thinning(masks: torch.Tensor, max_iter: int = 64, check_every: int = 4):
    """
    masks: (N, H, W) bool tensor on any device (N masks thinned in one batch).
    max_iter: must exceed half the thickest line in pixels (each iteration peels 1px
              from each side). Early-exits once nothing changes, so a high cap is free.
    Returns (N, H, W) bool skeleton, 1px wide (8-connected).
    """
    w, lut = _consts(masks.device)
    img = masks.unsqueeze(1).float()                      # (N,1,H,W)
    changed = torch.zeros((), dtype=torch.bool, device=masks.device)

    for it in range(max_iter):
        for step in (0, 1):
            code = F.conv2d(img, w, padding=1).long()     # neighbour byte per pixel
            delete = lut[step][code] & (img > 0)
            img = img * (~delete)
            changed |= delete.any()
        if (it + 1) % check_every == 0:                   # one GPU sync every few iters
            if not bool(changed):
                break
            changed.zero_()

    # Remove leftover staircase pixels. One orientation per pass, so two adjacent
    # corner pixels are never deleted simultaneously (which could break the line).
    for _ in range(2):
        for target in _CORNERS:
            code = F.conv2d(img, w, padding=1).long()
            img = img * (~((code == target) & (img > 0)))

    return img.squeeze(1) > 0


def order_skeleton_pixels(skel: torch.Tensor, min_len: int = 15):
    """
    skel: (H, W) bool skeleton. Returns (v, u) pixel indices ordered as polylines:
    each connected component is contiguous and runs near -> far (bottom of image first);
    components are ordered left -> right. This is the ordering the centerline code
    (split_lines / resample_line) assumes. torch.nonzero() instead returns row-major
    order, which interleaves the left and right lines row by row.
    Components shorter than min_len pixels (specks) are dropped.
    """
    dev = skel.device
    sk = skel.cpu().numpy().astype(np.uint8)
    n, labels = cv2.connectedComponents(sk, connectivity=8)
    v, u = np.nonzero(labels)
    if v.size == 0:
        e = torch.empty(0, dtype=torch.int64, device=dev)
        return e, e.clone()
    lab = labels[v, u]
    counts = np.bincount(lab, minlength=n)
    keep = counts[lab] >= min_len
    v, u, lab = v[keep], u[keep], lab[keep]
    mean_u = np.bincount(lab, weights=u, minlength=n) / np.maximum(counts, 1)
    # lexsort: last key is primary
    order = np.lexsort((u, -v, lab, mean_u[lab]))
    to_t = lambda a: torch.from_numpy(a[order].astype(np.int64)).to(dev)
    return to_t(v), to_t(u)