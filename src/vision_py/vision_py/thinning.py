import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Lookup tables
#
# The neighbourhood code now includes the CENTRE pixel as bit 8 (value 256), so a
# single gather returns the final "keep" multiplier (1.0 = keep, 0.0 = delete) and
# the old `& (img > 0)` / `~` / compare steps disappear. Per sub-iteration that is
# 4 kernels (conv, cast, gather, mul) instead of ~9.
#
# Neighbour bit order (P2..P9 clockwise): N=1, NE=2, E=4, SE=8, S=16, SW=32, W=64, NW=128.
# ----------------------------------------------------------------------------
def _build_luts_np():
    keep = np.ones((2, 512), dtype=np.float32)
    for code9 in range(512):
        centre = (code9 >> 8) & 1
        code = code9 & 0xFF
        if not centre:
            continue
        n = [(code >> i) & 1 for i in range(8)]
        N, NE, E, SE, S, SW, W, NW = n
        B = sum(n)
        A = sum(1 for i in range(8) if n[i] == 0 and n[(i + 1) % 8] == 1)
        if 2 <= B <= 6 and A == 1:
            if not (N and E and S) and not (E and S and W):
                keep[0, code9] = 0.0
            if not (N and E and W) and not (N and S and W):
                keep[1, code9] = 0.0
    return keep


# Staircase corners: exactly two neighbours that are orthogonal to each other
# (N+E, E+S, S+W, W+N). The two neighbours are already diagonally connected,
# so the centre pixel is redundant. Zhang-Suen keeps these (A == 2).
_CORNERS = (1 | 4, 4 | 16, 16 | 64, 64 | 1)


def _build_corner_luts_np():
    keep = np.ones((len(_CORNERS), 512), dtype=np.float32)
    for t, target in enumerate(_CORNERS):
        keep[t, 256 | target] = 0.0
    return keep


_W_NP = np.array([[128, 1, 2],
                  [64, 256, 4],
                  [32, 16, 8]], dtype=np.float32)
_KEEP_NP = _build_luts_np()
_CORNER_NP = _build_corner_luts_np()

_cache = {}


def _consts(device):
    """Returns (weights, (keep_step0, keep_step1), [keep_corner_t...], dtype) on `device`.
    fp16 on CUDA halves memory traffic; every value involved (<= 511) is exact in fp16."""
    key = str(device)
    if key not in _cache:
        dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        w = torch.from_numpy(_W_NP).view(1, 1, 3, 3).to(device=device, dtype=dtype)
        steps = tuple(torch.from_numpy(_KEEP_NP[i]).to(device=device, dtype=dtype) for i in range(2))
        corners = [torch.from_numpy(_CORNER_NP[i]).to(device=device, dtype=dtype) for i in range(len(_CORNERS))]
        _cache[key] = (w, steps, corners, dtype)
    return _cache[key]


@torch.no_grad()
def _thin_conv(masks: torch.Tensor, max_iter: int = 64, check_every: int = 4):
    """Conv + LUT implementation (portable fallback; memory-bound on large masks)."""
    w, (keep0, keep1), corner_luts, dtype = _consts(masks.device)
    steps = (keep0, keep1)
    img = (masks.unsqueeze(1) > 0).to(dtype)               # (N,1,H,W), fresh tensor -> safe to mutate
    prev = torch.count_nonzero(img)

    for it in range(max_iter):
        for lut in steps:
            code = F.conv2d(img, w, padding=1).long()      # 9-bit neighbourhood incl. centre
            img.mul_(lut[code])                            # gather keep-multiplier, apply
        if (it + 1) % check_every == 0:                    # one GPU sync every few iters
            cur = torch.count_nonzero(img)
            if bool(cur == prev):                          # deletions only shrink the count
                break
            prev = cur

    # Remove leftover staircase pixels. One orientation per pass, so two adjacent
    # corner pixels are never deleted simultaneously (which could break the line).
    for _ in range(2):
        for lut in corner_luts:
            code = F.conv2d(img, w, padding=1).long()
            img.mul_(lut[code])

    return img.squeeze(1) > 0


# ----------------------------------------------------------------------------
# Fused implementation
#
# The conv/LUT version moves ~25 MB per pass (fp16 conv, int64 code, gather) and 1-channel
# cuDNN convs are slow, so ~40 passes cost ~30 ms even with few kernels. Here the Zhang-Suen
# conditions are computed directly from 8 shifted views of a uint8 image with plain
# elementwise ops; torch.compile fuses each sub-iteration into ONE Triton kernel that reads
# 1 byte/pixel and writes 1 byte/pixel.
# ----------------------------------------------------------------------------
def _nbrs(img):
    """img: (N,H,W) uint8 0/1. Returns N, NE, E, SE, S, SW, W, NW views (zero padded)."""
    p = F.pad(img, (1, 1, 1, 1))
    return (p[:, :-2, 1:-1], p[:, :-2, 2:], p[:, 1:-1, 2:], p[:, 2:, 2:],
            p[:, 2:, 1:-1], p[:, 2:, :-2], p[:, 1:-1, :-2], p[:, :-2, :-2])


def _zs_both(img):
    """One full Zhang-Suen iteration (sub-step 0 then sub-step 1)."""
    for step in (0, 1):
        N, NE, E, SE, S, SW, W, NW = _nbrs(img)
        B = N + NE + E + SE + S + SW + W + NW
        A = (((1 - N) & NE) + ((1 - NE) & E) + ((1 - E) & SE) + ((1 - SE) & S)
             + ((1 - S) & SW) + ((1 - SW) & W) + ((1 - W) & NW) + ((1 - NW) & N))
        if step == 0:
            ok = ((1 - (N & E & S)) & (1 - (E & S & W))) == 1
        else:
            ok = ((1 - (N & E & W)) & (1 - (N & S & W))) == 1
        delete = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & ok
        img = img * (~delete)
    return img


def _corners(img):
    """Staircase-corner removal: 2 passes x 4 orientations, sequential (same as the LUT version)."""
    for _ in range(2):
        for pair in ((0, 2), (2, 4), (4, 6), (6, 0)):        # N+E, E+S, S+W, W+N
            nb = _nbrs(img)
            B = nb[0] + nb[1] + nb[2] + nb[3] + nb[4] + nb[5] + nb[6] + nb[7]
            delete = (img == 1) & (B == 2) & (nb[pair[0]] == 1) & (nb[pair[1]] == 1)
            img = img * (~delete)
    return img


_fused = None   # (compiled_iter, compiled_corners) once enable_fused_thinning() succeeds


@torch.no_grad()
def _thin_fused(masks, max_iter=64, check_every=4):
    it_fn, corner_fn = _fused
    img = (masks > 0).to(torch.uint8)
    prev = torch.count_nonzero(img)
    for it in range(max_iter):
        img = it_fn(img)
        if (it + 1) % check_every == 0:
            cur = torch.count_nonzero(img)
            if bool(cur == prev):
                break
            prev = cur
    return corner_fn(img) > 0


def enable_fused_thinning(device, hw, log=print):
    """Compile the fused kernels (takes ~10-40 s once, at startup), then verify them against the
    conv implementation on a synthetic mask with thick vertical, horizontal and diagonal bands.
    Any failure (no Triton, no C compiler, mismatch) leaves the conv path active. Returns bool."""
    global _fused
    _fused = None
    if torch.device(device).type != "cuda":
        return False
    try:
        import time
        H, W = hw
        t0 = time.perf_counter()
        fns = (torch.compile(_zs_both, dynamic=False), torch.compile(_corners, dynamic=False))
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        m = torch.zeros(2, H, W, dtype=torch.bool, device=device)
        m[0] = ((xx - 300).abs() < 14) | (((xx - 0.8 * yy - 450).abs() < 10) & (yy > 100)) | ((yy - 400).abs() < 7)
        m[1] = ((xx - 700).abs() < 6) | ((xx + 0.5 * yy - 900).abs() < 12) | (((xx - 500) ** 2 + (yy - 300) ** 2).sqrt() < 40)
        ref = _thin_conv(m)
        _fused = fns
        out = _thin_fused(m)
        torch.cuda.synchronize()
        if not torch.equal(out, ref):
            raise RuntimeError(f"fused result differs from conv result in {int((out != ref).sum())} px")
        torch.cuda.synchronize(); t1 = time.perf_counter()
        for _ in range(10):
            _thin_fused(m)
        torch.cuda.synchronize(); t_f = (time.perf_counter() - t1) / 10 * 1e3
        t2 = time.perf_counter()
        for _ in range(10):
            _thin_conv(m)
        torch.cuda.synchronize(); t_c = (time.perf_counter() - t2) / 10 * 1e3
        log(f"[thinning] fused path active (compile {t1 - t0:.1f}s): fused {t_f:.2f} ms vs conv {t_c:.2f} ms")
        if t_f >= t_c:
            _fused = None
            log("[thinning] fused path not faster; using conv path")
        return _fused is not None
    except Exception as e:                                  # noqa: BLE001
        _fused = None
        log(f"[thinning] fused path disabled, using conv path: {e}")
        return False


def zhang_suen_thinning(masks: torch.Tensor, max_iter: int = 64, check_every: int = 4):
    """
    masks: (N, H, W) bool/uint8 tensor on any device (N masks thinned in one batch).
    max_iter: must exceed half the thickest line in pixels (each iteration peels 1px
              from each side). Early-exits once nothing changes, so a high cap is free.
    Returns (N, H, W) bool skeleton, 1px wide (8-connected). Output is identical to the
    original implementation. Uses the fused kernels if enable_fused_thinning() succeeded.
    """
    if _fused is not None and masks.is_cuda:
        return _thin_fused(masks, max_iter, check_every)
    return _thin_conv(masks, max_iter, check_every)


def order_skeleton_pixels(skel, min_len: int = 15):
    """
    skel: (H, W) bool/uint8 skeleton as a NUMPY array (transfer both skeletons to the CPU
    once, then call this per mask; this avoids a GPU sync per call).
    Returns numpy int64 (v, u) pixel indices ordered as polylines:
    each connected component is contiguous and runs near -> far (bottom of image first);
    components are ordered left -> right. This is the ordering the centerline code
    (split_lines / resample_line) assumes. Components shorter than min_len pixels
    (specks) are dropped.
    """
    if isinstance(skel, torch.Tensor):
        skel = skel.cpu().numpy()
    sk = skel.astype(np.uint8, copy=False)
    n, labels = cv2.connectedComponents(sk, connectivity=8)
    v, u = np.nonzero(sk)                                  # sparse: only skeleton pixels
    if v.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e.copy()
    lab = labels[v, u]
    counts = np.bincount(lab, minlength=n)
    keep = counts[lab] >= min_len
    v, u, lab = v[keep], u[keep], lab[keep]
    mean_u = np.bincount(lab, weights=u, minlength=n) / np.maximum(counts, 1)
    # lexsort: last key is primary
    order = np.lexsort((u, -v, lab, mean_u[lab]))
    return v[order].astype(np.int64), u[order].astype(np.int64)