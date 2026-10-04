"""
Shared evaluation helpers for HydraNet (used by finetune_sim.py and Evaluate_sim.py).

  decode()          raw detection-head output of one image -> boxes / scores / class ids after NMS
  SegAccumulator    confusion matrix -> per-class IoU
  DetAccumulator    per-class AP@IoU, and precision / recall at a score threshold
  evaluate_model()  runs a model over a DataLoader and returns all of the above
"""
import numpy as np
import torch
from torchvision.ops import batched_nms


@torch.no_grad()
def decode(det, i, h, w, score_thr, nms_iou=0.6, pre_k=1000, max_det=100):
    """Detection-head outputs of image i -> (boxes xyxy px, scores, class ids)."""
    cls_prob = torch.sigmoid(det["cls_logits"][i].float())      # (N, C)
    ctr = torch.sigmoid(det["centerness"][i].float())           # (N, 1)
    scores = torch.sqrt(cls_prob * ctr)                         # (N, C)
    boxes = det["boxes"][i].float()                             # (N, 4)
    num_c = scores.shape[1]

    flat = scores.flatten()
    top_s, top_i = flat.topk(min(pre_k, flat.numel()))
    keep = top_s > score_thr
    top_s, top_i = top_s[keep], top_i[keep]
    if top_s.numel() == 0:
        return boxes.new_zeros((0, 4)), top_s, top_i

    pt = top_i // num_c
    cl = top_i % num_c
    b = boxes[pt].clone()
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)

    k = batched_nms(b, top_s, cl, nms_iou)[:max_det]
    return b[k], top_s[k], cl[k]


def _iou_1_to_n(box, boxes):
    ix = np.maximum(0.0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]))
    iy = np.maximum(0.0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))
    inter = ix * iy
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


class SegAccumulator:
    def __init__(self, num_classes):
        self.n = num_classes
        self.conf = torch.zeros(num_classes * num_classes, dtype=torch.long)

    def add(self, pred, gt):
        idx = (gt.long() * self.n + pred.long()).flatten()
        self.conf += torch.bincount(idx, minlength=self.n * self.n).cpu()

    def matrix(self):
        return self.conf.reshape(self.n, self.n).numpy()

    def iou(self):
        """per-class IoU; NaN where the class never occurs in GT or predictions"""
        c = self.matrix().astype(np.float64)
        out = np.full(self.n, np.nan)
        for k in range(self.n):
            union = c[k, :].sum() + c[:, k].sum() - c[k, k]
            if union > 0:
                out[k] = c[k, k] / union
        return out


class DetAccumulator:
    def __init__(self, num_classes):
        self.n = num_classes
        self.dets = [[] for _ in range(num_classes)]   # (img_idx, score, box)
        self.gts = [{} for _ in range(num_classes)]    # img_idx -> [boxes]

    def add(self, img_idx, boxes, scores, labels, gt_boxes, gt_labels):
        for b, s, c in zip(boxes, scores, labels):
            self.dets[int(c)].append((img_idx, float(s), b))
        for gb, gl in zip(gt_boxes, gt_labels):
            self.gts[int(gl)].setdefault(img_idx, []).append(gb)

    def n_gt(self, c):
        return sum(len(v) for v in self.gts[c].values())

    def _match(self, c, iou_thr):
        """greedy matching in descending score order -> (sorted dets, tp flags)"""
        dets = sorted(self.dets[c], key=lambda d: -d[1])
        gts = {k: np.asarray(v, dtype=np.float64) for k, v in self.gts[c].items()}
        used = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
        tp = np.zeros(len(dets))
        for j, (img, _, box) in enumerate(dets):
            if img not in gts:
                continue
            ious = _iou_1_to_n(np.asarray(box, dtype=np.float64), gts[img])
            m = int(ious.argmax())
            if ious[m] >= iou_thr and not used[img][m]:
                tp[j] = 1
                used[img][m] = True
        return dets, tp

    def ap(self, c, iou_thr=0.5):
        n_gt = self.n_gt(c)
        if n_gt == 0:
            return float("nan")
        dets, tp = self._match(c, iou_thr)
        if not dets:
            return 0.0
        fp = 1 - tp
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / n_gt
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

    def prf(self, c, iou_thr=0.5, score_thr=0.3):
        """(precision, recall, tp, fp, n_gt) counting only detections with score >= score_thr"""
        n_gt = self.n_gt(c)
        dets, tp = self._match(c, iou_thr)
        scores = np.array([d[1] for d in dets])
        sel = scores >= score_thr
        tpn = int(tp[sel].sum())
        fpn = int(sel.sum() - tpn)
        prec = tpn / (tpn + fpn) if (tpn + fpn) else float("nan")
        rec = tpn / n_gt if n_gt else float("nan")
        return prec, rec, tpn, fpn, n_gt


@torch.no_grad()
def evaluate_model(model, loader, device, num_seg, num_obj, img_h, img_w,
                   map_score_thr=0.05, max_images=None, on_batch=None):
    """
    Returns dict with seg_iou (list, per class), fg_miou (mean over classes 1..), ap50 / ap75
    (per object class, NaN if class absent), map50, map75 and the raw accumulators.
    on_batch(first_idx, images, pred_seg, out, batch) is an optional hook (used for visualisations).
    """
    was_training = model.training
    model.eval()
    use_amp = device.type == "cuda"
    seg, det = SegAccumulator(num_seg), DetAccumulator(num_obj)
    seen = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(images)
        pred_seg = out["lane_logits"].argmax(1)
        seg.add(pred_seg, batch["seg_masks"].to(device))

        for i in range(images.shape[0]):
            b, s, c = decode(out["detection"], i, img_h, img_w, map_score_thr)
            det.add(seen + i, b.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy(),
                    batch["boxes"][i].numpy(), batch["labels"][i].numpy())

        if on_batch is not None:
            on_batch(seen, images, pred_seg, out, batch)
        seen += images.shape[0]
        if max_images and seen >= max_images:
            break

    if was_training:
        model.train()

    iou = seg.iou()
    ap50 = [det.ap(c, 0.5) for c in range(num_obj)]
    ap75 = [det.ap(c, 0.75) for c in range(num_obj)]
    return {
        "seg_iou": iou.tolist(),
        "fg_miou": float(np.nanmean(iou[1:])) if not np.all(np.isnan(iou[1:])) else 0.0,
        "ap50": ap50, "ap75": ap75,
        "map50": float(np.nanmean(ap50)) if not np.all(np.isnan(ap50)) else 0.0,
        "map75": float(np.nanmean(ap75)) if not np.all(np.isnan(ap75)) else 0.0,
        "seg": seg, "det": det, "n_images": seen,
    }