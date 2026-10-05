"""
Loss functions for HydraNet pretraining:
  - sigmoid_focal_loss: classification
  - giou_loss: box regression
  - MultiTaskLoss: learned-uncertainty combination of per-task losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    """logits, targets: (N, num_classes). targets are one-hot, all-zero for negatives."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def giou_loss(pred_boxes, target_boxes, reduction="mean", eps=1e-7):
    """pred_boxes, target_boxes: (M, 4) xyxy pixel coords, positive samples only."""
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0  # zero loss, keeps the autograd graph valid

    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    union = pred_area + target_area - inter + eps
    iou = inter / union

    cx1, cy1 = torch.min(px1, tx1), torch.min(py1, ty1)
    cx2, cy2 = torch.max(px2, tx2), torch.max(py2, ty2)
    enclosing = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps

    giou = iou - (enclosing - union) / enclosing
    loss = 1.0 - giou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class MultiTaskLoss(nn.Module):
    """
    Learned-uncertainty multi-task loss combination. Each task gets a
    learned log-variance parameter; the optimizer itself down-weights
    noisier/harder tasks over training instead of you hand-tuning fixed
    scalar weights.
    """

    def __init__(self, task_names=("seg", "cls", "reg", "ctr")):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses: dict):
        """losses: dict[task_name] -> scalar loss tensor for this step."""
        total = 0.0
        weighted = {}
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * losses[name] + self.log_vars[i]
            weighted[name] = weighted_loss
            total = total + weighted_loss
        return total, weighted