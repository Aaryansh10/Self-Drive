import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_norm_cache = {}


def _norm_consts(device):
    """(x/255 - mean)/std folded into one multiply-add: x * scale - bias."""
    key = str(device)
    if key not in _norm_cache:
        mean = torch.tensor(_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        _norm_cache[key] = (1.0 / (255.0 * std), mean / std)
    return _norm_cache[key]


def preprocess(img_bgr, device='cuda', use_fp16=False, channels_last=False):
    """Uploads the raw uint8 BGR frame (1.5 MB) and does BGR->RGB, float conversion and
    normalisation on the GPU. The old version did this in float32 on the CPU (~9 ms)
    and uploaded 6 MB."""
    scale, bias = _norm_consts(device)
    t = torch.from_numpy(np.ascontiguousarray(img_bgr)).to(device)    # (H, W, 3) uint8
    x = t.permute(2, 0, 1).unsqueeze(0).flip(1)                       # (1, 3, H, W), RGB
    x = x.float().mul_(scale).sub_(bias)
    if use_fp16:
        x = x.half()
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    return x.contiguous(memory_format=fmt)


def load_weights(model, weights):
    if not weights:
        print("No weights given!")
        return

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)

    def is_state_dict(d):
        return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())

    state = None
    if isinstance(ckpt, torch.nn.Module):
        state = ckpt.state_dict()
    elif is_state_dict(ckpt):
        state = ckpt
    elif isinstance(ckpt, dict):
        print(f"Checkpoint top-level keys: {list(ckpt.keys())}")
        preferred = (
            "ema", "ema_state_dict", "model_state_dict",
            "state_dict", "model_state", "model", "net", "weights"
        )
        for key in list(preferred) + list(ckpt.keys()):
            v = ckpt.get(key)
            if isinstance(v, torch.nn.Module):
                v = v.state_dict()
            if is_state_dict(v):
                print(f"Using checkpoint entry '{key}'")
                state = v
                break
    if state is None:
        raise RuntimeError("Could not find a state_dict in the checkpoint")

    # strip DataParallel / DDP prefix if present
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    msg = f"Loaded weights from {weights} (missing={len(missing)}, unexpected={len(unexpected)})"
    if missing or unexpected:
        print(msg + f" e.g. missing={missing[:3]} unexpected={unexpected[:3]}")
    else:
        print(msg)


_EMPTY = (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64))


@torch.no_grad()
def postprocess_detections(det, scale_xy, frame_wh, score_thresh, nms_iou, max_detections):
    """FCOS-style: score = sqrt(cls_prob * centerness_prob), class-wise NMS.
    Returns (frame_boxes (n,4), scores (n,), labels (n,)) as numpy arrays.
    (model_boxes is no longer returned; nothing used it.) One GPU->CPU transfer."""
    class_logits = det["cls_logits"][0].float().sigmoid()  # (N, C)
    centreness = det["centerness"][0].float().sigmoid()    # (N, 1)
    boxes = det["boxes"][0].float()                        # (N, 4) xyxy, input px
    num_classes = class_logits.shape[1]

    flat = (class_logits * centreness).sqrt().flatten()
    idx = (flat > score_thresh).nonzero(as_tuple=True)[0]  # the one unavoidable sync (dynamic shape)
    if idx.numel() == 0:
        return _EMPTY

    if idx.numel() > 1000:  # pre-NMS top-k
        idx = idx[flat[idx].topk(1000).indices]

    anchor_idx = idx // num_classes
    labels = idx % num_classes
    sel_boxes, sel_scores = boxes[anchor_idx], flat[idx]

    k = batched_nms(sel_boxes, sel_scores, labels, nms_iou)[:max_detections]
    sel_boxes, sel_scores, labels = sel_boxes[k], sel_scores[k], labels[k]

    sx, sy = scale_xy
    fw, fh = frame_wh
    fb = torch.empty_like(sel_boxes)
    fb[:, [0, 2]] = (sel_boxes[:, [0, 2]] * sx).clamp(0, fw - 1)
    fb[:, [1, 3]] = (sel_boxes[:, [1, 3]] * sy).clamp(0, fh - 1)

    # drop boxes that collapsed after clamping; filter on the CPU so there is no extra sync
    pack = torch.cat([fb, sel_scores[:, None], labels[:, None].float()], dim=1).cpu().numpy()
    valid = ((pack[:, 2] - pack[:, 0]) > 2) & ((pack[:, 3] - pack[:, 1]) > 2)
    pack = pack[valid]
    if pack.shape[0] == 0:
        return _EMPTY
    return pack[:, :4], pack[:, 4], pack[:, 5].astype(np.int64)


# ----------------------------------------------------------------------------
# Mask cleaning (CPU / OpenCV)
#
# The GPU version needed ~100+ tiny kernels and a host sync per label-propagation
# iteration, for each of 3 masks. A 544x960 uint8 mask is only 0.5 MB, so one transfer
# plus OpenCV's connected-components is far cheaper. Semantics are identical:
# open -> close -> drop 8-connected components with area < min_area or
# perimeter < min_perimeter (perimeter = pixels removed by a 3x3 erosion).
# Borders behave the same as max_pool2d's implicit -inf padding (no erosion at the edge).
# ----------------------------------------------------------------------------
_se_cache = {}


def _se(k):
    if k not in _se_cache:
        _se_cache[k] = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return _se_cache[k]


def clean_mask_np(mask, open_k=3, close_k=5, min_area=150, min_perimeter=50):
    """mask: (H, W) bool or uint8 numpy array. Returns (H, W) bool numpy array."""
    m = mask.view(np.uint8) if mask.dtype == np.bool_ else (mask > 0).astype(np.uint8)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)

    if open_k > 1:                       # removes specks smaller than the kernel
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _se(open_k))
    if close_k > 1:                      # bridges small gaps in dashed or worn markings
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _se(close_k))
    if not m.any():
        return np.zeros(m.shape, dtype=bool)

    # connectedComponents is ~5x faster than ...WithStats, and everything below only touches
    # foreground pixels (a few % of the frame) instead of the full 544x960 label map.
    n, labels = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, dtype=bool)

    fg = m.view(np.bool_)                                      # m is strictly 0/1
    lab_fg = labels[fg]
    area = np.bincount(lab_fg, minlength=n)

    boundary = (m - cv2.erode(m, _se(3))).view(np.bool_)       # m >= erode(m), so no underflow
    perim = np.bincount(labels[boundary], minlength=n)

    keep = (area >= min_area) & (perim >= min_perimeter)
    keep[0] = False
    out = np.zeros(m.shape, dtype=bool)
    out[fg] = keep[lab_fg]
    return out