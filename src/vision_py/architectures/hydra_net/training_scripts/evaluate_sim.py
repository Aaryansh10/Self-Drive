"""
Evaluate a sim-finetuned HydraNet checkpoint on a labeled sim split.

Outputs per image:
  1) <name>_pred_img.jpg  : Image with predicted (orange) & GT (green) bounding boxes.
  2) <name>_pred_seg.jpg  : Pure segmentation mask on a black background.

Run from the hydra_net folder:

  PYTHONPATH=.:.. python3 training_scripts/evaluate_sim.py \
      --root ~/Manas/Self-Drive/src/Data/Sim_Data --ckpt sim_finetuned.pt --split test
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import sys
from pathlib import Path

script_path = Path(__file__).resolve()
hydra_net_dir = script_path.parents[1]         # points to .../hydra_net
architectures_dir = script_path.parents[2]     # points to .../architectures
vision_py_dir = script_path.parents[3]         # points to .../vision_py

for p in [hydra_net_dir, architectures_dir, vision_py_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from architectures.model import MODEL_VARIANTS, build_hydranet
from training_scripts.sim_dataset_loader import (
    SimDataset, sim_collate_fn, SIM_SEG_CLASSES, SIM_OBJ_CLASSES,
)
from training_scripts.bdd_dataset_loader import IMAGENET_MEAN, IMAGENET_STD
from training_scripts.eval_utils import evaluate_model, decode

# Color mappings for pure segmentation output (RGB)
SEG_COLORS = {
    1: (255, 255, 255),  # white_lane -> White
    2: (255, 220, 0),    # yellow_lane -> Yellow
    3: (255, 40, 40)     # stop_line -> Red
}


def draw_predictions(image_t, pred_seg, pboxes, pscores, plabels, gboxes, glabels, base_path):
    # -------------------------------------------------------------------------
    # 1. Box Detection Image (Original frame + Bounding Boxes)
    # -------------------------------------------------------------------------
    img = image_t.cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    pil_det = Image.fromarray(img)
    d = ImageDraw.Draw(pil_det)

    # Ground-truth boxes (Green)
    for b, l in zip(gboxes, glabels):
        d.rectangle([float(v) for v in b], outline=(0, 255, 0), width=2)

    # Predicted boxes (Orange)
    for b, s, l in zip(pboxes, pscores, plabels):
        d.rectangle([float(v) for v in b], outline=(255, 140, 0), width=2)
        d.text((float(b[0]) + 2, max(0.0, float(b[1]) - 11)),
               f"{SIM_OBJ_CLASSES[int(l)]} {float(s):.2f}", fill=(255, 255, 0))

    pil_det.save(f"{base_path}_pred_img.jpg")

    # -------------------------------------------------------------------------
    # 2. Pure Segmentation Image (Black background with color masks)
    # -------------------------------------------------------------------------
    seg = pred_seg.cpu().numpy()
    h, w = seg.shape
    seg_canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Black canvas

    for cid, col in SEG_COLORS.items():
        seg_canvas[seg == cid] = col

    pil_seg = Image.fromarray(seg_canvas)
    pil_seg.save(f"{base_path}_pred_seg.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", default="sim_finetuned.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--img_h", type=int, default=544)
    ap.add_argument("--img_w", type=int, default=960)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--vis_n", type=int, default=30)
    ap.add_argument("--out_dir", default="eval_sim_out_deep")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    num_seg, num_obj = ck["num_seg_classes"], ck["num_obj_classes"]
    model = build_hydranet(ck["model_variant"], input_size=(args.img_h, args.img_w),
                           num_seg_classes=num_seg, num_obj_classes=num_obj)
    model.load_state_dict(ck["full_state_dict"])
    model.to(device).eval()
    print(f"Loaded {args.ckpt} (variant '{ck['model_variant']}', epoch {ck.get('epoch', '?')})")

    ds = SimDataset(args.root, args.split, img_size=(args.img_h, args.img_w), augment=False,
                    val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=sim_collate_fn, pin_memory=True)

    def on_batch(first, images, pred_seg, out, batch):
        for i in range(images.shape[0]):
            if first + i >= args.vis_n:
                break
            b, s, c = decode(out["detection"], i, args.img_h, args.img_w, args.score_thr)
            name = os.path.splitext(batch["names"][i])[0]
            base_path = os.path.join(args.out_dir, f"{name}")
            draw_predictions(images[i], pred_seg[i], b.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy(),
                             batch["boxes"][i].numpy(), batch["labels"][i].numpy(), base_path)

    m = evaluate_model(model, loader, device, num_seg, num_obj, args.img_h, args.img_w,
                       max_images=args.max_images, on_batch=on_batch)

    print(f"\n=== {args.split} split, {m['n_images']} images ===")
    print("\nSegmentation IoU:")
    for i, n in enumerate(SIM_SEG_CLASSES):
        print(f"  {n:12s} {m['seg_iou'][i]:.4f}")
    print(f"  mIoU (3 line classes) {m['fg_miou']:.4f}")

    print(f"\nDetection (AP over IoU thresholds; precision/recall at score >= {args.score_thr}, IoU 0.5):")
    print(f"  {'class':11s} {'#GT':>5s} {'AP50':>7s} {'AP75':>7s} {'prec':>7s} {'recall':>7s}")
    det = m["det"]
    for c, n in enumerate(SIM_OBJ_CLASSES):
        prec, rec, tp, fp, n_gt = det.prf(c, 0.5, args.score_thr)
        print(f"  {n:11s} {n_gt:5d} {m['ap50'][c]:7.4f} {m['ap75'][c]:7.4f} {prec:7.3f} {rec:7.3f}")
    print(f"  {'mean':11s} {'':5s} {m['map50']:7.4f} {m['map75']:7.4f}")

    print("\nSegmentation confusion matrix (rows = ground truth, columns = prediction), % of GT pixels:")
    conf = m["seg"].matrix().astype(np.float64)
    print("  " + " " * 12 + "".join(f"{n[:9]:>10s}" for n in SIM_SEG_CLASSES))
    for i, n in enumerate(SIM_SEG_CLASSES):
        row = 100.0 * conf[i] / max(1.0, conf[i].sum())
        print(f"  {n:12s}" + "".join(f"{v:10.2f}" for v in row))

    print(f"\nPrediction images saved in {args.out_dir}/")


if __name__ == "__main__":
    main()