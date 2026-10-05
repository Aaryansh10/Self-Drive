"""
Pretrain HydraNet's backbone + neck + segmentation/detection heads on
BDD100K. The sign-authenticity head is not trained here.

Single GPU / CPU (unchanged):
    python pretrain_bdd.py --bdd_root /data/bdd100k --model_variant base

Multi-GPU (DistributedDataParallel), e.g. 3 GPUs on one machine:
    torchrun --standalone --nproc_per_node=3 pretrain_bdd.py \
        --bdd_root /data/bdd100k --model_variant base --batch_size 8

--batch_size is the PER-GPU batch size, so the command above trains at an
effective batch size of 24. Run it twice with --model_variant base and
--model_variant deep (see hydra_net.model.MODEL_VARIANTS) to get two
checkpoints you can hand to compare_checkpoints.py.
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hydra_net.model import MODEL_VARIANTS, build_hydranet
from training_scripts.bdd_dataset_loader import BDDDataset, bdd_collate_fn, BDD_DET_CLASSES
from training_scripts.target_assigner import build_strides_per_point, assign_targets_batch
from training_scripts.pretrain_losses import sigmoid_focal_loss, giou_loss, MultiTaskLoss

NUM_SEG_CLASSES = 3  # proxy classes: background, drivable area, lane marking


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_distributed():
    """True when launched via torchrun (it always sets these env vars)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_metrics(metrics, world_size, device):
    """Average a dict of python scalars across all processes so every rank
    logs/decides on the same (global) numbers, not just its local shard."""
    if world_size == 1:
        return metrics
    keys = sorted(metrics.keys())
    vals = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= world_size
    return dict(zip(keys, vals.tolist()))


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
                     num_obj_classes, scaler, base_lr, step, total_steps, warmup_steps,
                     rank=0, log_every=50):
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

        if is_main_process(rank) and n_batches % log_every == 0:
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
    parser.add_argument("--batch_size", type=int, default=8,
                         help="PER-GPU batch size. Effective batch size = batch_size * world_size "
                              "when run under torchrun.")
    parser.add_argument("--lr", type=float, default=3e-4,
                         help="Base learning rate (for the --batch_size given above).")
    parser.add_argument("--scale_lr_with_world_size", action="store_true",
                         help="Linearly scale --lr by the number of GPUs, a common rule of thumb "
                              "since effective batch size grows with world size.")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_variant", type=str, default="base", choices=list(MODEL_VARIANTS),
                         help="Named architecture from hydra_net.model.MODEL_VARIANTS to train.")
    parser.add_argument("--out", type=str, default=None,
                         help="Checkpoint path. Defaults to bdd_pretrained_<model_variant>.pt")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop training after this many consecutive epochs with no "
                              "validation-loss improvement (> --min_delta). Set to 0 to disable.")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                         help="Minimum decrease in val total loss to count as an improvement.")
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable train-time data augmentation (random crop/flip/color jitter).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"bdd_pretrained_{args.model_variant}.pt"

    distributed = is_distributed()
    if distributed:
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Different seed per rank so each process's dataloader workers/augmentation
    # don't all draw identical random streams.
    torch.manual_seed(args.seed + rank)

    if is_main_process(rank):
        print(f"Distributed: {distributed} | world_size {world_size} | device {device}")
        print(f"Model variant: '{args.model_variant}' -> {MODEL_VARIANTS[args.model_variant]}")

    num_obj_classes = len(BDD_DET_CLASSES)

    train_set = BDDDataset(args.bdd_root, split="train", img_size=(args.img_h, args.img_w),
                            augment=not args.no_augment)
    val_set = BDDDataset(args.bdd_root, split="val", img_size=(args.img_h, args.img_w),
                          augment=False)
    if is_main_process(rank):
        print(f"train images: {len(train_set)}, val images: {len(val_set)} "
              f"(augmentation {'off' if args.no_augment else 'on'})")

    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                            shuffle=True, seed=args.seed, drop_last=True)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank,
                                          shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                               shuffle=(train_sampler is None), sampler=train_sampler,
                               num_workers=args.num_workers, collate_fn=bdd_collate_fn,
                               drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             sampler=val_sampler, num_workers=args.num_workers,
                             collate_fn=bdd_collate_fn, pin_memory=True)

    model = build_hydranet(
        args.model_variant,
        input_size=(args.img_h, args.img_w),
        num_seg_classes=NUM_SEG_CLASSES,
        num_obj_classes=num_obj_classes,
    ).to(device)

    # Grab strides off the plain module before it's (possibly) wrapped in DDP.
    strides_per_point = build_strides_per_point(args.img_h, args.img_w, model.det_head.strides).to(device)

    if is_main_process(rank):
        print(f"Parameter counts: {model.count_parameters()}")

    if distributed:
        # SyncBatchNorm so BN statistics are computed across all GPUs rather
        # than each GPU's (smaller) local shard -- matters more the smaller
        # --batch_size is per GPU.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank) if torch.cuda.is_available() else {}
        model = DDP(model, **ddp_kwargs)
    model_without_ddp = model.module if distributed else model

    mt_loss = MultiTaskLoss(task_names=("seg", "cls", "reg", "ctr")).to(device)

    lr = args.lr * world_size if args.scale_lr_with_world_size else args.lr
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mt_loss.parameters()), lr=lr, weight_decay=0.05
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    total_steps = args.epochs * len(train_loader)
    step = 0

    patience = args.patience if args.patience > 0 else None
    early_stopper = EarlyStopper(patience=patience, min_delta=args.min_delta)

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)  # reshuffles differently each epoch across ranks

        t0 = time.time()
        if is_main_process(rank):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics, step = train_one_epoch(
            model, train_loader, optimizer, mt_loss, device, strides_per_point,
            num_obj_classes, scaler, lr, step, total_steps, args.warmup_steps, rank=rank,
        )
        val_metrics = validate(model, val_loader, device, strides_per_point, num_obj_classes, mt_loss)

        # Every rank only saw its own shard of data -- average across all
        # ranks so logging/early-stopping/checkpointing act on global numbers.
        train_metrics = reduce_metrics(train_metrics, world_size, device)
        val_metrics = reduce_metrics(val_metrics, world_size, device)

        dt = time.time() - t0
        if is_main_process(rank):
            print(f"  train: {train_metrics}")
            print(f"  val  : {val_metrics}")
            print(f"  epoch time: {dt / 60:.1f} min")

        is_new_best, should_stop = early_stopper.step(val_metrics["total"])

        if is_new_best:
            if is_main_process(rank):
                torch.save({
                    "full_state_dict": model_without_ddp.state_dict(),
                    "backbone_state_dict": model_without_ddp.backbone.state_dict(),
                    "neck_state_dict": model_without_ddp.neck.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "num_seg_classes": NUM_SEG_CLASSES,
                    "num_obj_classes": num_obj_classes,
                    "model_variant": args.model_variant,
                    "model_kwargs": MODEL_VARIANTS[args.model_variant],
                }, args.out)
                print(f"  saved new best checkpoint -> {args.out}")
        elif is_main_process(rank):
            print(f"  no improvement ({early_stopper.num_bad_epochs}/{early_stopper.patience} "
                  f"bad epochs, best val total {early_stopper.best:.4f})")

        if should_stop:
            if is_main_process(rank):
                print(f"\nEarly stopping: no val-loss improvement for {early_stopper.patience} "
                      f"consecutive epochs (best {early_stopper.best:.4f}).")
            break

    if is_main_process(rank):
        print("\nDone.")

    cleanup_distributed()


if __name__ == "__main__":
    main()