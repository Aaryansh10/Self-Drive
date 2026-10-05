"""
FCOS-style target assignment for HydraNet's anchor-free DetectionHead.

DetectionHead.forward() concatenates predictions from all pyramid levels
(P3, P4, P5) into single (B, N, ...) tensors, in level order. This module
reproduces that exact point layout so you can score raw model outputs
against these targets directly, without re-deriving level boundaries
yourself.

For every grid point, decide:
  - whether it's a positive sample (falls inside some GT box, at a scale
    appropriate for its pyramid level) or negative
  - which GT box (if any) it's responsible for -- ties broken by smallest
    box area, standard FCOS practice
  - the classification target (one-hot for positives, all-zero for negatives)
  - the regression target (l, t, r, b distances to the assigned box, pixels)
  - the centerness target (how close the point is to the box's center)
"""
import torch


def build_strides_per_point(img_h, img_w, strides=(8, 16, 32)):
    """(N,) tensor giving the stride of the level each concatenated point
    belongs to, matching DetectionHead's own concatenation order."""
    parts = []
    for s in strides:
        h, w = img_h // s, img_w // s
        parts.append(torch.full((h * w,), float(s)))
    return torch.cat(parts, dim=0)


def assign_targets(points, strides_per_point, gt_boxes, gt_labels, num_classes,
                    scale_ranges=((0, 64), (64, 128), (128, float("inf")))):
    """
    points:            (N, 2) xy grid centers, pixel space
    strides_per_point: (N,) stride of the level each point belongs to
    gt_boxes:          (K, 4) xyxy GT boxes for this image, pixel space
    gt_labels:         (K,) GT class ids for this image
    num_classes:       number of object classes
    scale_ranges:      per-level max(l,t,r,b) range a point may regress,
                        ordered to match the ascending unique strides
                        present in strides_per_point (default matches
                        HydraNet's default strides=(8, 16, 32))

    returns:
      cls_target: (N, num_classes) one-hot, all-zero for negatives
      reg_target: (N, 4) ltrb distances, zero for negatives
      ctr_target: (N,) centerness in [0, 1], zero for negatives
      pos_mask:   (N,) bool, True where the point is a positive sample
    """
    n = points.shape[0]
    device = points.device
    cls_target = torch.zeros((n, num_classes), device=device)
    reg_target = torch.zeros((n, 4), device=device)
    ctr_target = torch.zeros((n,), device=device)
    pos_mask = torch.zeros((n,), dtype=torch.bool, device=device)

    k = gt_boxes.shape[0]
    if k == 0:
        return cls_target, reg_target, ctr_target, pos_mask

    unique_strides = sorted(set(strides_per_point.tolist()))
    stride_to_range = {s: scale_ranges[i] for i, s in enumerate(unique_strides)}
    point_low = torch.tensor([stride_to_range[s.item()][0] for s in strides_per_point], device=device)
    point_high = torch.tensor([stride_to_range[s.item()][1] for s in strides_per_point], device=device)

    xs = points[:, 0].unsqueeze(1)  # (N, 1)
    ys = points[:, 1].unsqueeze(1)  # (N, 1)
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]  # (K,)

    l = xs - x1.unsqueeze(0)   # (N, K)
    t = ys - y1.unsqueeze(0)
    r = x2.unsqueeze(0) - xs
    b = y2.unsqueeze(0) - ys
    ltrb = torch.stack([l, t, r, b], dim=-1)  # (N, K, 4)

    inside_box = ltrb.min(dim=-1).values > 0
    max_dist = ltrb.max(dim=-1).values
    in_scale_range = (max_dist >= point_low.unsqueeze(1)) & (max_dist < point_high.unsqueeze(1))
    candidate = inside_box & in_scale_range  # (N, K)

    areas = (x2 - x1) * (y2 - y1)  # (K,)
    areas_expanded = areas.unsqueeze(0).expand(n, k).clone()
    areas_expanded[~candidate] = float("inf")
    min_area, best_box_idx = areas_expanded.min(dim=1)  # (N,), (N,)

    pos_mask = torch.isfinite(min_area)
    if pos_mask.sum() == 0:
        return cls_target, reg_target, ctr_target, pos_mask

    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    matched_box_idx = best_box_idx[pos_idx]

    reg_target[pos_idx] = ltrb[pos_idx, matched_box_idx]
    matched_labels = gt_labels[matched_box_idx]
    cls_target[pos_idx, matched_labels] = 1.0

    l_pos = reg_target[pos_idx, 0]
    t_pos = reg_target[pos_idx, 1]
    r_pos = reg_target[pos_idx, 2]
    b_pos = reg_target[pos_idx, 3]
    ctr = torch.sqrt(
        (torch.min(l_pos, r_pos) / torch.clamp(torch.max(l_pos, r_pos), min=1e-6))
        * (torch.min(t_pos, b_pos) / torch.clamp(torch.max(t_pos, b_pos), min=1e-6))
    )
    ctr_target[pos_idx] = ctr

    return cls_target, reg_target, ctr_target, pos_mask


def assign_targets_batch(points, strides_per_point, gt_boxes_list, gt_labels_list,
                          num_classes, scale_ranges=((0, 64), (64, 128), (128, float("inf")))):
    """Per-image assign_targets, stacked into a batch dimension."""
    cls_targets, reg_targets, ctr_targets, pos_masks = [], [], [], []
    for boxes, labels in zip(gt_boxes_list, gt_labels_list):
        c, r, ct, p = assign_targets(points, strides_per_point, boxes, labels, num_classes, scale_ranges)
        cls_targets.append(c)
        reg_targets.append(r)
        ctr_targets.append(ct)
        pos_masks.append(p)
    return torch.stack(cls_targets), torch.stack(reg_targets), torch.stack(ctr_targets), torch.stack(pos_masks)