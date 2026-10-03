"""
Evaluate a pretrained HydraNet checkpoint on a BDD100K split.

Run from the hydra_net folder (same way as training):

  PYTHONPATH=.:.. python3 evaluate_bdd.py \
      --bdd_root ~/Manas/Self-Drive/src/Data/bdd100k_images_100k/100k/test \
      --ckpt bdd_pretrained_base.pt --split test

Outputs:
  - Segmentation IoU (background / drivable)   [only if the split has labels]
  - Detection AP@0.5 per class + mAP@0.5       [only if the split has labels]
  - Overlay images (boxes + drivable area) saved to --out_dir (always)
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms, box_iou

from hydra_net.model import build_hydranet
from training_scripts.bdd_dataset_loader import (
    BDDDataset, bdd_collate_fn, BDD_DET_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
)


@torch.no_grad()
def decode(det, i, h, w, score_thr, nms_iou=0.6, pre_k=1000, max_det=100):
    """Raw head outputs of image i -> (boxes, scores, class_ids) after NMS."""
    cls_prob = torch.sigmoid(det["cls_logits"][i].float())      # (N, C)
    ctr = torch.sigmoid(det["centerness"][i].float())           # (N, 1)
    scores = torch.sqrt(cls_prob * ctr)                         # (N, C)
    boxes = det["boxes"][i].float()                             # (N, 4) xyxy px
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


def ap_for_class(dets, gts, iou_thr=0.5):
    """dets: list of (img_idx, score, [x1,y1,x2,y2]); gts: {img_idx: [[x1,y1,x2,y2], ...]}"""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return None
    if not dets:
        return 0.0

    dets = sorted(dets, key=lambda d: -d[1])
    gt_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in gts.items()}
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))

    for j, (img, _, box) in enumerate(dets):
        if img not in gt_t:
            fp[j] = 1
            continue
        ious = box_iou(torch.tensor([box], dtype=torch.float32), gt_t[img])[0].numpy()
        m = int(ious.argmax())
        if ious[m] >= iou_thr and not used[img][m]:
            tp[j] = 1
            used[img][m] = True
        else:
            fp[j] = 1

    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)

    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def save_vis(image_t, pred_seg, boxes, scores, labels, path):
    img = image_t.cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    seg = pred_seg.cpu().numpy()
    overlay = img.copy()
    overlay[seg == 1] = (0, 200, 0)          # drivable area -> green
    overlay[seg == 2] = (0, 0, 255)          # class 2 (unused for now)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    for b, s, c in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, f"{BDD_DET_CLASSES[int(c)]} {s:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_root", required=True)
    ap.add_argument("--ckpt", default="bdd_pretrained_base.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--img_h", type=int, default=544)
    ap.add_argument("--img_w", type=int, default=960)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--map_score_thr", type=float, default=0.05)
    ap.add_argument("--vis_score_thr", type=float, default=0.3)
    ap.add_argument("--vis_n", type=int, default=30)
    ap.add_argument("--out_dir", default="eval_out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    num_seg = ck["num_seg_classes"]
    num_obj = ck["num_obj_classes"]
    model = build_hydranet(ck["model_variant"], input_size=(args.img_h, args.img_w),
                           num_seg_classes=num_seg, num_obj_classes=num_obj)
    model.load_state_dict(ck["full_state_dict"])
    model.to(device).eval()
    print(f"Loaded {args.ckpt} (variant '{ck['model_variant']}', epoch {ck.get('epoch', '?')})")

    ds = BDDDataset(args.bdd_root, split=args.split, img_size=(args.img_h, args.img_w), augment=False)
    has_labels = ds.labels_dir.exists() and ds.drivable_dir.exists()
    if not has_labels:
        print(f"[info] no labels found for split '{args.split}' -> only saving visualizations, no metrics.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=bdd_collate_fn, pin_memory=True)

    conf = torch.zeros(num_seg * num_seg, dtype=torch.long, device=device)
    det_store = {c: [] for c in range(num_obj)}
    gt_store = {c: {} for c in range(num_obj)}
    seen, t0 = 0, time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
            det = out["detection"]
            pred_seg = out["lane_logits"].argmax(1)
            B = images.shape[0]

            if has_labels:
                gt_seg = batch["seg_masks"].to(device)
                conf += torch.bincount((gt_seg * num_seg + pred_seg).flatten(),
                                       minlength=num_seg * num_seg)

            for i in range(B):
                img_idx = seen + i
                if has_labels:
                    b, s, c = decode(det, i, args.img_h, args.img_w, args.map_score_thr)
                    for bb, ss, cc in zip(b.tolist(), s.tolist(), c.tolist()):
                        det_store[cc].append((img_idx, ss, bb))
                    for gb, gl in zip(batch["boxes"][i].tolist(), batch["labels"][i].tolist()):
                        gt_store[gl].setdefault(img_idx, []).append(gb)

                if img_idx < args.vis_n:
                    vb, vs, vc = decode(det, i, args.img_h, args.img_w, args.vis_score_thr)
                    name = os.path.splitext(batch["names"][i])[0]
                    save_vis(images[i], pred_seg[i], vb, vs, vc,
                             os.path.join(args.out_dir, f"{name}_pred.jpg"))

            seen += B
            if args.max_images and seen >= args.max_images:
                break

    print(f"Processed {seen} images in {(time.time() - t0) / 60:.1f} min")
    print(f"Visualizations saved in: {args.out_dir}/")

    if not has_labels:
        return

    conf = conf.reshape(num_seg, num_seg).cpu().numpy()
    seg_names = ["background", "drivable", "class2 (unused)"]
    print("\nSegmentation IoU:")
    ious = []
    for c in range(num_seg):
        gt_c = conf[c, :].sum()
        if gt_c == 0:
            continue
        iou = conf[c, c] / (gt_c + conf[:, c].sum() - conf[c, c])
        ious.append(iou)
        print(f"  {seg_names[c]:<16} {iou:.4f}")
    print(f"  mIoU             {np.mean(ious):.4f}")
    print(f"  pixel accuracy   {np.trace(conf) / conf.sum():.4f}")

    print("\nDetection AP@0.5:")
    aps = []
    for c in range(num_obj):
        a = ap_for_class(det_store[c], gt_store[c])
        if a is None:
            print(f"  {BDD_DET_CLASSES[c]:<14} (no GT)")
            continue
        aps.append(a)
        print(f"  {BDD_DET_CLASSES[c]:<14} {a:.4f}")
    print(f"  mAP@0.5        {np.mean(aps):.4f}")


if __name__ == "__main__":
    main()