"""
Fine-tune the BDD-pretrained HydraNet on the sim dataset.

  seg classes (4): background, white_lane, yellow_lane, stop_line
  objects     (5): barrel, tire, pothole, sign (valid STOP sign), pedestrian

Stage A (first --freeze_epochs epochs): backbone + neck frozen (BatchNorm in eval mode too),
        only the heads are trained.
Stage B (rest): everything trains; backbone/neck LR = --backbone_lr_mult x head LR.

The best checkpoint is chosen by  score = 0.5 * mean(IoU of the 3 line classes) + 0.5 * mAP@0.5
on the val split (every class counts equally), NOT by the loss.

Run from the hydra_net folder (same way as BDD training):

  PYTHONPATH=.:.. python3 training_scripts/finetune_sim.py \
      --root ~/Manas/Self-Drive/src/Data/Sim_Data \
      --pretrained bdd_pretrained_base_v2.pt --out sim_finetuned.pt

Ctrl+C saves <out>_last.pt; continue later with  --resume <out>_last.pt  (same other arguments).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
import cv2
cv2.setNumThreads(0)

# Path setup: ensure proper directory hierarchy is in sys.path
script_path = Path(__file__).resolve()
training_scripts_dir = script_path.parents[0]  # points to .../training_scripts
hydra_net_dir = script_path.parents[1]         # points to .../hydra_net
architectures_dir = script_path.parents[2]     # points to .../architectures
vision_py_dir = script_path.parents[3]         # points to .../vision_py

for p in [training_scripts_dir, hydra_net_dir, architectures_dir, vision_py_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from architectures.model import MODEL_VARIANTS, build_hydranet
from sim_dataset_loader import (
    SimDataset, sim_collate_fn, SIM_SEG_CLASSES, SIM_OBJ_CLASSES,
)
from target_assigner import build_strides_per_point, assign_targets_batch
from pretrain_losses import sigmoid_focal_loss, giou_loss, MultiTaskLoss
from eval_utils import evaluate_model

"""
import argparse
import math
import os
import time
import sys
from pathlib import Path

# Add the parent architecture directory to sys.path so 'hydra_net' is recognized as a package
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import MODEL_VARIANTS, build_hydranet
from training_scripts.sim_dataset_loader import (
    SimDataset, sim_collate_fn, SIM_SEG_CLASSES, SIM_OBJ_CLASSES,
)
from training_scripts.target_assigner import build_strides_per_point, assign_targets_batch
from training_scripts.pretrain_losses import sigmoid_focal_loss, giou_loss, MultiTaskLoss
from training_scripts.eval_utils import evaluate_model
"""

NUM_SEG = len(SIM_SEG_CLASSES)   # 4
NUM_OBJ = len(SIM_OBJ_CLASSES)   # 5
# sim class -> index in BDD_DET_CLASSES whose classifier row is a sensible starting point
BDD_CLS_TRANSFER = {"pedestrian": 0, "sign": 9}   # 0 = pedestrian, 9 = traffic sign


# ----------------------------------------------------------------------------- setup helpers
def load_pretrained(model, path, transfer_cls=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["full_state_dict"]
    own = model.state_dict()
    # classifiers belong to the BDD label spaces -> never copied as-is
    skip_prefix = ("seg_head.classifier.", "det_head.cls_pred.", "sign_head.")
    keep = {k: v for k, v in sd.items()
            if k in own and own[k].shape == v.shape and not k.startswith(skip_prefix)}
    res = model.load_state_dict(keep, strict=False)
    print(f"Loaded {len(keep)}/{len(own)} tensors from {path} "
          f"(BDD epoch {ck.get('epoch', '?') + 1 if isinstance(ck.get('epoch'), int) else '?'})")
    print(f"  fresh init: {len(res.missing_keys)} tensors (seg classifier, cls_pred, sign head, ...)")

    if transfer_cls and "det_head.cls_pred.weight" in sd:
        with torch.no_grad():
            for name, bdd_idx in BDD_CLS_TRANSFER.items():
                j = SIM_OBJ_CLASSES.index(name)
                model.det_head.cls_pred.weight[j] = sd["det_head.cls_pred.weight"][bdd_idx]
                model.det_head.cls_pred.bias[j] = sd["det_head.cls_pred.bias"][bdd_idx]
        print(f"  classifier rows initialised from BDD: {list(BDD_CLS_TRANSFER)}")


def set_backbone_frozen(model, frozen):
    for p in list(model.backbone.parameters()) + list(model.neck.parameters()):
        p.requires_grad_(not frozen)


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, progress)))


def seg_class_weights(train_set, device):
    counts = train_set.seg_pixel_counts().astype("float64")
    freq = counts / counts.sum()
    w = (freq[0] / (freq + 1e-12)) ** 0.5          # background = 1, rarer classes more
    w = w.clip(1.0, 10.0)
    print("seg pixel share :", {n: f"{100 * f:.3f}%" for n, f in zip(SIM_SEG_CLASSES, freq)})
    print("seg class weight:", {n: round(float(x), 2) for n, x in zip(SIM_SEG_CLASSES, w)})
    return torch.tensor(w, dtype=torch.float32, device=device)


def oversampling_weights(train_set):
    """image weight = sqrt(N_max / N_c) of the rarest class it contains (images without objects: 1)"""
    n_inst = [0] * NUM_OBJ
    for s in train_set.stems:
        for l in train_set.cache[s]["labels"]:
            n_inst[l] += 1
    n_max = max(n_inst)
    cw = [math.sqrt(n_max / max(1, n)) for n in n_inst]
    w = []
    for present in train_set.object_classes_per_image():
        w.append(max([cw[c] for c in present]) if present else 1.0)
    print("instances per class:", dict(zip(SIM_OBJ_CLASSES, n_inst)))
    print("class sampling boost:", {n: round(c, 2) for n, c in zip(SIM_OBJ_CLASSES, cw)})
    return w


# ----------------------------------------------------------------------------- losses (fp32)
def seg_loss_fn(logits, target, class_w):
    logits = logits.float()
    ce = F.cross_entropy(logits, target, weight=class_w)
    prob = logits.softmax(dim=1)
    onehot = F.one_hot(target, NUM_SEG).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + 1.0) / (denom + 1.0)
    dice_loss = 1.0 - dice[1:].mean()               # the 3 line classes, background excluded
    return ce + dice_loss


def compute_losses(out, seg_masks, gt_boxes, gt_labels, strides_per_point, class_w):
    det = out["detection"]
    points = det["points"]
    with torch.amp.autocast("cuda", enabled=False):
        cls_t, reg_t, ctr_t, pos_mask = assign_targets_batch(
            points.float(), strides_per_point, gt_boxes, gt_labels, NUM_OBJ)
        seg = seg_loss_fn(out["lane_logits"], seg_masks, class_w)

        num_pos = pos_mask.sum().clamp(min=1).float()
        cls = sigmoid_focal_loss(det["cls_logits"].float(), cls_t, reduction="sum") / num_pos

        boxes = det["boxes"].float()
        ctr_logit = det["centerness"].float().squeeze(-1)
        if pos_mask.any():
            p = points.float().unsqueeze(0).expand(pos_mask.shape[0], -1, -1)[pos_mask]
            l, t, r, b = reg_t[pos_mask].unbind(-1)
            target_xyxy = torch.stack([p[:, 0] - l, p[:, 1] - t, p[:, 0] + r, p[:, 1] + b], dim=-1)
            reg = giou_loss(boxes[pos_mask], target_xyxy, reduction="mean")
            ctr = F.binary_cross_entropy_with_logits(ctr_logit[pos_mask], ctr_t[pos_mask])
        else:
            reg = boxes.sum() * 0.0
            ctr = ctr_logit.sum() * 0.0
    return {"seg": seg, "cls": cls, "reg": reg, "ctr": ctr}


# ----------------------------------------------------------------------------- checkpoints
def save_best(path, model, epoch, metrics, args):
    torch.save({
        "full_state_dict": model.state_dict(),
        "backbone_state_dict": model.backbone.state_dict(),
        "neck_state_dict": model.neck.state_dict(),
        "epoch": epoch, "val_metrics": metrics,
        "num_seg_classes": NUM_SEG, "num_obj_classes": NUM_OBJ,
        "seg_classes": SIM_SEG_CLASSES, "obj_classes": SIM_OBJ_CLASSES,
        "model_variant": args.model_variant,
        "model_kwargs": MODEL_VARIANTS[args.model_variant],
    }, path)


def save_last(path, model, optimizer, mt_loss, scaler, next_epoch, step, best_score, args):
    torch.save({
        "full_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(), "mt_loss": mt_loss.state_dict(),
        "scaler": scaler.state_dict(), "next_epoch": next_epoch, "step": step,
        "best_score": best_score,
        "num_seg_classes": NUM_SEG, "num_obj_classes": NUM_OBJ,
        "model_variant": args.model_variant,
    }, path)


def fmt_val(m):
    iou = m["seg_iou"]
    s = " ".join(f"{n[:6]} {iou[i]:.3f}" for i, n in enumerate(SIM_SEG_CLASSES) if i > 0)
    a = " ".join(f"{n[:6]} {m['ap50'][i]:.3f}" for i, n in enumerate(SIM_OBJ_CLASSES))
    return (f"  val IoU: {s} | mIoU {m['fg_miou']:.3f}\n"
            f"  val AP50: {a} | mAP50 {m['map50']:.3f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="sim folder containing images/ and labels/")
    ap.add_argument("--pretrained", default=None, help="BDD checkpoint to start from")
    ap.add_argument("--no_cls_transfer", action="store_true")
    ap.add_argument("--out", default="sim_finetuned.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--model_variant", default="base", choices=list(MODEL_VARIANTS))
    ap.add_argument("--img_h", type=int, default=544)
    ap.add_argument("--img_w", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4, help="head LR")
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_oversample", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    last_path = args.out.replace(".pt", "_last.pt")
    img_size = (args.img_h, args.img_w)
    print(f"device {device} | out {args.out} | last {last_path}")

    # ---- data
    split_kw = dict(val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed)
    train_set = SimDataset(args.root, "train", img_size=img_size, **split_kw)
    val_set = SimDataset(args.root, "val", img_size=img_size, augment=False, **split_kw)
    if args.no_oversample:
        sampler = None
    else:
        sampler = WeightedRandomSampler(oversampling_weights(train_set),
                                        num_samples=len(train_set), replacement=True)
    persistent = args.num_workers > 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=args.num_workers,
                              collate_fn=sim_collate_fn, drop_last=True, pin_memory=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=sim_collate_fn,
                            pin_memory=True, persistent_workers=persistent)
    class_w = seg_class_weights(train_set, device)

    # ---- model
    model = build_hydranet(args.model_variant, input_size=img_size,
                           num_seg_classes=NUM_SEG, num_obj_classes=NUM_OBJ).to(device)
    if args.pretrained and not args.resume:
        load_pretrained(model, args.pretrained, transfer_cls=not args.no_cls_transfer)
    strides_per_point = build_strides_per_point(
        args.img_h, args.img_w, model.det_head.strides).to(device)

    mt_loss = MultiTaskLoss(task_names=("seg", "cls", "reg", "ctr")).to(device)
    backbone_params = list(model.backbone.parameters()) + list(model.neck.parameters())
    head_params = (list(model.seg_head.parameters()) + list(model.det_head.parameters())
                   + list(mt_loss.parameters()))
    optimizer = torch.optim.AdamW(
        [{"params": head_params}, {"params": backbone_params}],
        lr=args.lr, weight_decay=args.weight_decay)
    lr_mults = [1.0, args.backbone_lr_mult]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    start_epoch, step, best_score, bad_epochs = 0, 0, -1.0, 0
    
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["full_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        mt_loss.load_state_dict(ck["mt_loss"])
        if ck["scaler"] and len(ck["scaler"]) > 0:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best_score = ck["next_epoch"], ck["step"], ck["best_score"]
        print(f"Resumed from {args.resume}: epoch {start_epoch + 1}, step {step}, best score {best_score:.4f}")

    print(f"train {len(train_set)} imgs ({steps_per_epoch} steps/epoch) | val {len(val_set)} imgs | "
          f"epochs {args.epochs} (heads-only for first {args.freeze_epochs})")

    state = {"step": step, "epoch": start_epoch}
    try:
        for epoch in range(start_epoch, args.epochs):
            state["epoch"] = epoch
            frozen = epoch < args.freeze_epochs
            set_backbone_frozen(model, frozen)
            if epoch == args.freeze_epochs:
                print(f"\n>>> unfreezing backbone + neck (LR x{args.backbone_lr_mult})")

            model.train()
            if frozen:                      # keep BN running stats of the pretrained backbone untouched
                model.backbone.eval()
                model.neck.eval()

            print(f"\nEpoch {epoch + 1}/{args.epochs} [{'heads only' if frozen else 'full'}]")
            t0 = time.time()
            run = {k: 0.0 for k in ("seg", "cls", "reg", "ctr", "total")}
            n = 0
            for batch in train_loader:
                state["step"] = step
                images = batch["images"].to(device, non_blocking=True)
                seg_masks = batch["seg_masks"].to(device, non_blocking=True)
                gt_boxes = [b.to(device, non_blocking=True) for b in batch["boxes"]]
                gt_labels = [l.to(device, non_blocking=True) for l in batch["labels"]]

                lr = cosine_warmup_lr(step, total_steps, args.warmup_steps, args.lr)
                for g, m in zip(optimizer.param_groups, lr_mults):
                    g["lr"] = lr * m

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(images)
                losses = compute_losses(out, seg_masks, gt_boxes, gt_labels,
                                        strides_per_point, class_w)
                total, _ = mt_loss(losses)

                if not torch.isfinite(total):
                    print(f"  [warn] non-finite loss, skipping batch {batch['names']}: "
                          f"{ {k: v.item() for k, v in losses.items()} }")
                    step += 1
                    continue

                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()

                for k in ("seg", "cls", "reg", "ctr"):
                    run[k] += losses[k].item()
                run["total"] += total.item()
                n += 1
                step += 1
                if n % 25 == 0:
                    print(f"  step {n}/{steps_per_epoch} | lr {lr:.2e} | "
                          + " ".join(f"{k} {v / n:.4f}" for k, v in run.items()))

            state["step"] = step
            m = evaluate_model(model, val_loader, device, NUM_SEG, NUM_OBJ, args.img_h, args.img_w)
            score = 0.5 * m["fg_miou"] + 0.5 * m["map50"]
            print(f"  train: " + " ".join(f"{k} {v / max(1, n):.4f}" for k, v in run.items()))
            print(fmt_val(m))
            print(f"  score {score:.4f} | epoch time {(time.time() - t0) / 60:.1f} min")

            if score > best_score + args.min_delta:
                best_score, bad_epochs = score, 0
                save_best(args.out, model, epoch, {k: m[k] for k in
                          ("seg_iou", "fg_miou", "ap50", "ap75", "map50", "map75")}, args)
                print(f"  saved new best -> {args.out}")
            elif epoch >= args.freeze_epochs:   # patience only counts once everything is training
                bad_epochs += 1
                print(f"  no improvement ({bad_epochs}/{args.patience}, best {best_score:.4f})")

            save_last(last_path, model, optimizer, mt_loss, scaler, epoch + 1, step, best_score, args)
            if bad_epochs >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs "
                      f"(best score {best_score:.4f}).")
                break

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C) -> saving latest checkpoint...")
        save_last(last_path, model, optimizer, mt_loss, scaler,
                  state["epoch"], state["step"], best_score, args)
        print(f"  saved -> {last_path}  (resume with --resume {last_path})")
        return

    print(f"\nDone. Best val score {best_score:.4f} -> {args.out}")


if __name__ == "__main__":
    main()