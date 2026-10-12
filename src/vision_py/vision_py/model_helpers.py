import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

def preprocess(self, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
    x = x.to(self.device)
    return x.half() if self.use_fp16 else x

def load_weights(self, weights):
    if not weights:
        self.get_logger().warn("No weights given!")
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
        self.get_logger().info(f"Checkpoint top-level keys: {list(ckpt.keys())}")
        preferred = (
            "ema", "ema_state_dict", "model_state_dict", 
            "state_dict","model_state", "model", "net", "weights"
        )
        for key in list(preferred) + list(ckpt.keys()):
            v = ckpt.get(key)
            if isinstance(v, torch.nn.Module):
                v = v.state_dict()
            if is_state_dict(v):
                self.get_logger().info(f"Using checkpoint entry '{key}'")
                state = v
                break
    if state is None:
        raise RuntimeError("Could not find a state_dict in the checkpoint")

    # strip DataParallel / DDP prefix if present
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = self.model.load_state_dict(state, strict=False)
    msg = f"Loaded weights from {weights} (missing={len(missing)}, unexpected={len(unexpected)})"
    if missing or unexpected:
        self.get_logger().error(msg + f" e.g. missing={missing[:3]} unexpected={unexpected[:3]}")
    else:
        self.get_logger().info(msg)

@torch.no_grad()
def postprocess_detections(self, det, scale_xy, frame_wh):
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
    idx = (flat > self.score_thresh).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return empty()

    if idx.numel() > 1000: # pre-NMS top-k
        idx = idx[flat[idx].topk(1000).indices]

    anchor_idx = idx // num_classes
    labels = idx % num_classes
    selected_boxes, selected_scores = boxes[anchor_idx], flat[idx]

    k = batched_nms(selected_boxes, selected_scores, labels, self.nms_iou)[: self.max_detections]
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