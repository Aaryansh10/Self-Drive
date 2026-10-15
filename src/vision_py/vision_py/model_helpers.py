import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

def preprocess(
        img_bgr, device='cuda', use_fp16=False,
        mean=np.array([0.485, 0.456, 0.406] ,dtype=np.float32).reshape(1, 1, 3), 
        std=np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3),
    ):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = (img_rgb.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
    x = x.to(device)
    return x.half() if use_fp16 else x

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
            "state_dict","model_state", "model", "net", "weights"
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

@torch.no_grad()
def postprocess_detections(det, scale_xy, frame_wh, score_thresh, nms_iou, max_detections):
    class_logits = det["cls_logits"][0].float().sigmoid()  # (N, C)
    centreness = det["centerness"][0].float().sigmoid()  # (N, 1)
    boxes = det["boxes"][0].float()  # (N, 4) xyxy, input px
    num_classes = class_logits.shape[1]

    def empty():
        return (
            boxes.new_zeros((0, 4)),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    scores = (class_logits * centreness).sqrt()  # (N, C)
    flat = scores.flatten()
    idx = (flat > score_thresh).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return empty()

    if idx.numel() > 1000: # pre-NMS top-k
        idx = idx[flat[idx].topk(1000).indices]

    anchor_idx = idx // num_classes
    labels = idx % num_classes
    selected_boxes, selected_scores = boxes[anchor_idx], flat[idx]

    k = batched_nms(selected_boxes, selected_scores, labels, nms_iou)[: max_detections]
    model_boxes, selected_scores, labels = selected_boxes[k], selected_scores[k], labels[k]

    # rescale to original frame (separate copy; model_boxes stays in model space)
    sx, sy = scale_xy
    fw, fh = frame_wh
    frame_boxes = model_boxes.clone()
    frame_boxes[:, [0, 2]] = (frame_boxes[:, [0, 2]] * sx).clamp(0, fw - 1)
    frame_boxes[:, [1, 3]] = (frame_boxes[:, [1, 3]] * sy).clamp(0, fh - 1)

    # drop boxes that collapsed after clamping
    valid = ((frame_boxes[:, 2] - frame_boxes[:, 0]) > 2) & ((frame_boxes[:, 3] - frame_boxes[:, 1]) > 2)
    if not valid.any():
        return empty()

    model_boxes = model_boxes[valid]
    frame_boxes = frame_boxes[valid]
    selected_scores = selected_scores[valid]
    labels = labels[valid]

    return (
        model_boxes,
        frame_boxes.cpu().numpy(),
        selected_scores.cpu().numpy(),
        labels.cpu().numpy(),
    )

def dilate(mask, k):
    return F.max_pool2d(mask, k, stride=1, padding=k // 2)

def erode(mask, k):
    return -F.max_pool2d(-mask, k, stride=1, padding=k // 2)

def label_components(mask: torch.Tensor, max_iters: int = 48) -> torch.Tensor:
    """
    mask: (H, W) bool on GPU. Returns a flat float32 label map (H*W,), 0 = background.
    Max-label propagation over 8-neighbours plus pointer jumping, which converges in
    far fewer iterations than plain propagation, even on long lanes.
    """
    H, W = mask.shape
    fg = mask.reshape(-1).float()
    labels = torch.arange(1, H * W + 1, device=mask.device, dtype=torch.float32) * fg
    zero = labels.new_zeros(1)
    for _ in range(max_iters):
        prev = labels
        nb = F.max_pool2d(labels.view(1, 1, H, W), 3, 1, 1).view(-1) * fg
        # jump: replace each label by the label of the pixel whose id it holds
        labels = torch.cat([zero, nb])[nb.long()]
        if torch.equal(labels, prev):
            break
    return labels

def clean_mask(
    mask: torch.Tensor,
    open_k: int = 3,
    close_k: int = 5,
    min_area: int = 120,
    min_perimeter: int = 40,
) -> torch.Tensor:
    """
    mask: (H, W) bool on GPU, full resolution.
    open -> close -> drop connected components with area < min_area or perimeter < min_perimeter.
    """
    H, W = mask.shape
    m = mask[None, None].float()

    if open_k > 1:                       # removes specks smaller than the kernel
        m = dilate(erode(m, open_k), open_k)
    if close_k > 1:                      # bridges small gaps in dashed or worn markings
        m = erode(dilate(m, close_k), close_k)

    mb = m[0, 0] > 0.5
    if not mb.any():
        return mb

    labels = label_components(mb).long()
    n = H * W + 1

    boundary = mb & ~(erode(mb[None, None].float(), 3)[0, 0] > 0.5)
    area = torch.bincount(labels, minlength=n)
    perim = torch.bincount(labels[boundary.reshape(-1)], minlength=n)

    keep = (area >= min_area) & (perim >= min_perimeter)
    keep[0] = False
    return keep[labels].view(H, W)
