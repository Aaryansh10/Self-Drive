"""
Pretrain HydraNet's backbone + neck + segmentation/detection heads on
BDD100K. The sign-authenticity head is not trained here.
"""
import argparse
import math
import time

import torch
from torch.utils.data import DataLoader

from vision_py.architectures.hydra_net.model_small import HydraNet
from training_scripts.bdd_dataset_loader import BDDDataset, bdd_collate_fn, BDD_DET_CLASSES
from training_scripts.target_assigner import build_strides_per_point, assign_targets_batch
from training_scripts.pretrain_losses import sigmoid_focal_loss, giou_loss, MultiTaskLoss

NUM_SEG_CLASSES = 3  # proxy classes: background, drivable area, lane marking


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def _reg_targets_to_xyxy(points, pos_mask, reg_target):
    """Convert per-positive ltrb distance targets back to xyxy boxes, in
    the same layout as the model's own decoded det['boxes'], so GIoU can
    compare like-for-like."""
    pts_pos = points.unsqueeze(0).expand(pos_mask.shape[0], -1, -1)[pos_mask]
    l, t, r, b = reg_target[pos_mask].unbind(-1)
    x, y = pts_pos[:, 0], pts_pos[:, 1]
    return torch.stack([x - l, y - t, x + r, y + b], dim=-1)


def compute_losses(out, seg_masks, gt_boxes, gt_labels, points, strides_per_point, num_obj_classes):
    det = out["detection"]
    cls_t, reg_t, ctr_t, pos_mask = assign_targets_batch(
        points, strides_per_point, gt_boxes, gt_labels, num_obj_classes
    )

    seg_loss = torch.nn.functional.cross_entropy(out["lane_logits"], seg_masks)

    num_pos = max(1, pos_mask.sum().item())
    cls_loss = sigmoid_focal_loss(det["cls_logits"], cls_t, reduction="sum") / num_pos

    if pos_mask.any():
        target_xyxy = _reg_targets_to_xyxy(points, pos_mask, reg_t)
        reg_loss = giou_loss(det["boxes"][pos_mask], target_xyxy, reduction="mean")
        ctr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            det["centerness"].squeeze(-1)[pos_mask], ctr_t[pos_mask]
        )
    else:
        reg_loss = det["boxes"].sum() * 0.0
        ctr_loss = det["centerness"].sum() * 0.0

    return {"seg": seg_loss, "cls": cls_loss, "reg": reg_loss, "ctr": ctr_loss}


def train_one_epoch(model, loader, optimizer, mt_loss, device, strides_per_point,
                     num_obj_classes, scaler, base_lr, step, total_steps, warmup_steps, log_every=50):
    model.train()
    running = {"seg": 0.0, "cls": 0.0, "reg": 0.0, "ctr": 0.0, "total": 0.0}
    n_batches = 0
    use_amp = device.type == "cuda"

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        seg_masks = batch["seg_masks"].to(device, non_blocking=True)
        gt_boxes = [b.to(device) for b in batch["boxes"]]
        gt_labels = [l.to(device) for l in batch["labels"]]

        lr = cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            out = model(images)
            points = out["detection"]["points"]
            losses = compute_losses(out, seg_masks, gt_boxes, gt_labels, points, strides_per_point, num_obj_classes)
            total_loss, _ = mt_loss(losses)

        if scaler is not None and use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        for k in ("seg", "cls", "reg", "ctr"):
            running[k] += losses[k].item()
        running["total"] += total_loss.item()
        n_batches += 1
        step += 1

        if n_batches % log_every == 0:
            print(f"  step {n_batches}/{len(loader)} | lr {lr:.2e} | "
                  f"total {running['total']/n_batches:.4f} seg {running['seg']/n_batches:.4f} "
                  f"cls {running['cls']/n_batches:.4f} reg {running['reg']/n_batches:.4f} "
                  f"ctr {running['ctr']/n_batches:.4f}")

    return {k: v / max(1, n_batches) for k, v in running.items()}, step


@torch.no_grad()
def validate(model, loader, device, strides_per_point, num_obj_classes, mt_loss):
    model.eval()
    running = {"seg": 0.0, "cls": 0.0, "reg": 0.0, "ctr": 0.0, "total": 0.0}
    n_batches = 0
    for batch in loader:
        images = batch["images"].to(device)
        seg_masks = batch["seg_masks"].to(device)
        gt_boxes = [b.to(device) for b in batch["boxes"]]
        gt_labels = [l.to(device) for l in batch["labels"]]

        out = model(images)
        points = out["detection"]["points"]
        losses = compute_losses(out, seg_masks, gt_boxes, gt_labels, points, strides_per_point, num_obj_classes)
        total_loss, _ = mt_loss(losses)

        for k in ("seg", "cls", "reg", "ctr"):
            running[k] += losses[k].item()
        running["total"] += total_loss.item()
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in running.items()}


class EarlyStopper:
    """Stops training once validation loss stops improving by at least
    `min_delta` for `patience` consecutive epochs. Tracks the best value
    seen so the caller can decide whether the current epoch produced a
    new best checkpoint."""

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad_epochs = 0

    def step(self, value):
        """Returns (is_new_best, should_stop)."""
        if value < self.best - self.min_delta:
            self.best = value
            self.num_bad_epochs = 0
            return True, False

        self.num_bad_epochs += 1
        should_stop = self.patience is not None and self.num_bad_epochs >= self.patience
        return False, should_stop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd_root", type=str, required=True)
    parser.add_argument("--img_h", type=int, default=544)
    parser.add_argument("--img_w", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="bdd_pretrained.pt")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop training after this many consecutive epochs with no "
                              "validation-loss improvement (> --min_delta). Set to 0 to disable.")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                         help="Minimum decrease in val total loss to count as an improvement.")
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable train-time data augmentation (random crop/flip/color jitter).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_obj_classes = len(BDD_DET_CLASSES)

    train_set = BDDDataset(args.bdd_root, split="train", img_size=(args.img_h, args.img_w),
                            augment=not args.no_augment)
    val_set = BDDDataset(args.bdd_root, split="val", img_size=(args.img_h, args.img_w),
                          augment=False)
    print(f"train images: {len(train_set)}, val images: {len(val_set)} "
          f"(augmentation {'off' if args.no_augment else 'on'})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=bdd_collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=bdd_collate_fn)

    model = HydraNet(
        input_size=(args.img_h, args.img_w),
        num_seg_classes=NUM_SEG_CLASSES,
        num_obj_classes=num_obj_classes,
    ).to(device)

    strides_per_point = build_strides_per_point(args.img_h, args.img_w, model.det_head.strides).to(device)

    mt_loss = MultiTaskLoss(task_names=("seg", "cls", "reg", "ctr")).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mt_loss.parameters()), lr=args.lr, weight_decay=0.05
    )
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    total_steps = args.epochs * len(train_loader)
    step = 0

    patience = args.patience if args.patience > 0 else None
    early_stopper = EarlyStopper(patience=patience, min_delta=args.min_delta)

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics, step = train_one_epoch(
            model, train_loader, optimizer, mt_loss, device, strides_per_point,
            num_obj_classes, scaler, args.lr, step, total_steps, args.warmup_steps,
        )
        val_metrics = validate(model, val_loader, device, strides_per_point, num_obj_classes, mt_loss)
        dt = time.time() - t0
        print(f"  train: {train_metrics}")
        print(f"  val  : {val_metrics}")
        print(f"  epoch time: {dt / 60:.1f} min")

        is_new_best, should_stop = early_stopper.step(val_metrics["total"])

        if is_new_best:
            torch.save({
                "full_state_dict": model.state_dict(),
                "backbone_state_dict": model.backbone.state_dict(),
                "neck_state_dict": model.neck.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "num_seg_classes": NUM_SEG_CLASSES,
                "num_obj_classes": num_obj_classes,
            }, args.out)
            print(f"  saved new best checkpoint -> {args.out}")
        else:
            print(f"  no improvement ({early_stopper.num_bad_epochs}/{early_stopper.patience} "
                  f"bad epochs, best val total {early_stopper.best:.4f})")

        if should_stop:
            print(f"\nEarly stopping: no val-loss improvement for {early_stopper.patience} "
                  f"consecutive epochs (best {early_stopper.best:.4f}).")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()