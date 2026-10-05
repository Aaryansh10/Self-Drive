"""
BDD100K dataset loader for HydraNet pretraining.

Expects the standard unzipped BDD100K layout:
    <root>/images/100k/{train,val}/*.jpg
    <root>/labels/det_20/det_{train,val}.json
    <root>/labels/drivable/masks/{train,val}/*.png
    <root>/labels/lane/masks/{train,val}/*.png

Produces, per sample:
  image     : (3, H, W) float tensor, ImageNet-normalized
  seg_mask  : (H, W) long tensor, proxy classes {0: background, 1: drivable
              area, 2: lane marking} -- NOT your final class set, just a
              pretraining proxy (see note below)
  boxes     : (K, 4) float tensor, xyxy pixel coords in the resized image
  labels    : (K,) long tensor, BDD detection category ids (0-9)

IMPORTANT label simplifications (read before using beyond pretraining):
  - Drivable-area masks encode {0: direct drivable, 1: alternative drivable,
    2: background} per BDD's format; both drivable classes are collapsed
    into a single "drivable area" class here.
  - Lane masks pack category/direction/style into one byte, with 255 =
    background (see BDD100K docs, "Lane Marking Format"). This loader
    treats any non-255 pixel as "lane marking" without decoding
    category/direction/style -- enough signal to pretrain generic
    lane-texture features, but not a substitute for decoding the full
    encoding if you later want per-category lane semantics.
  - Where drivable-area and lane pixels overlap after resizing, lane wins
    (thin lane lines are easy to lose to a coarser drivable blob otherwise).

TRAIN-TIME AUGMENTATION (only applied when augment=True, i.e. split="train"):
  - Random resized crop: crops a random region of the *original* image
    (and its masks, in original resolution) before the final resize to
    img_size, then re-derives box/mask coordinates against the crop. Boxes
    that fall (almost) entirely outside the crop are dropped; boxes that
    are partially outside are clipped to the crop boundary.
  - Random horizontal flip (p=0.5): mirrors image, masks, and boxes
    together. This does not attempt to relabel any left/right-specific
    semantics -- there are none in this proxy label set (seg classes are
    background/drivable/lane, and none of the BDD_DET_CLASSES are
    lateral-direction-specific), so a plain mirror is safe here.
  - Color jitter (brightness/contrast/saturation): image only, does not
    touch geometry, so boxes/masks are untouched.
  Validation/test splits are never augmented -- augment is forced off
  unless explicitly requested, and the training script only turns it on
  for the train split.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

# BDD100K "Detection 2020" categories, in a fixed, stable order.
BDD_DET_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]
BDD_DET_CLASS_TO_ID = {name: i for i, name in enumerate(BDD_DET_CLASSES)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _random_crop_box(orig_w, orig_h, scale=(0.75, 1.0), ratio=(0.9, 1.1111)):
    """Pick a random crop rectangle (x0, y0, cw, ch) from the original
    image. Kept fairly conservative (scale down to 0.75x area) since BDD
    scenes already have small/far objects that a more aggressive crop
    (e.g. classic RandomResizedCrop's 0.08-1.0 range) would frequently
    remove entirely."""
    area = orig_w * orig_h
    for _ in range(10):
        target_area = area * random.uniform(*scale)
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect = np.exp(random.uniform(*log_ratio))

        cw = int(round(np.sqrt(target_area * aspect)))
        ch = int(round(np.sqrt(target_area / aspect)))

        if 0 < cw <= orig_w and 0 < ch <= orig_h:
            x0 = random.randint(0, orig_w - cw)
            y0 = random.randint(0, orig_h - ch)
            return x0, y0, cw, ch

    # Fallback: no crop (use full image) if we couldn't sample a valid box.
    return 0, 0, orig_w, orig_h


def _crop_and_clip_boxes(boxes, labels, x0, y0, cw, ch, min_size=2.0):
    """Translate boxes into crop-local coordinates, clip to the crop, and
    drop any box that's degenerate after clipping (i.e. was almost
    entirely outside the crop)."""
    if len(boxes) == 0:
        return boxes, labels

    boxes = boxes.copy()
    boxes[:, [0, 2]] -= x0
    boxes[:, [1, 3]] -= y0
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, cw)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ch)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)

    return boxes[keep], labels[keep]


def _color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2):
    """Lightweight color jitter using PIL's ImageEnhance, applied in a
    random order with random factors in [1-x, 1+x]. Image-only; does not
    touch geometry, so no need to touch boxes/masks."""
    ops = []
    if brightness > 0:
        ops.append(("brightness", random.uniform(1 - brightness, 1 + brightness)))
    if contrast > 0:
        ops.append(("contrast", random.uniform(1 - contrast, 1 + contrast)))
    if saturation > 0:
        ops.append(("saturation", random.uniform(1 - saturation, 1 + saturation)))
    random.shuffle(ops)

    for name, factor in ops:
        if name == "brightness":
            img = ImageEnhance.Brightness(img).enhance(factor)
        elif name == "contrast":
            img = ImageEnhance.Contrast(img).enhance(factor)
        elif name == "saturation":
            img = ImageEnhance.Color(img).enhance(factor)
    return img


class BDDDataset(Dataset):
    def __init__(self, root, split="train", img_size=(544, 960), augment=None):
        """
        root: path to the unzipped bdd100k root
        split: "train" or "val"
        img_size: (H, W), should match HydraNet's input_size
        augment: whether to apply train-time augmentation (random crop,
            horizontal flip, color jitter). Defaults to True for
            split == "train" and False otherwise; pass explicitly to
            override.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.out_h, self.out_w = img_size
        self.augment = (split == "train") if augment is None else augment

        self.img_dir = self.root / "images" / "100k" / split
        self.drivable_dir = self.root / "labels" / "drivable" / "masks" / split
        self.lane_dir = self.root / "labels" / "lane" / "masks" / split
        det_json_path = self.root / "labels" / "det_20" / f"det_{split}.json"

        self.image_names = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg"))
        )

        self.det_by_name = {}
        if det_json_path.exists():
            with open(det_json_path, "r") as f:
                det_entries = json.load(f)
            for entry in det_entries:
                boxes, labels = [], []
                for lab in entry.get("labels", []):
                    cat = lab.get("category")
                    box2d = lab.get("box2d")
                    if cat in BDD_DET_CLASS_TO_ID and box2d is not None:
                        boxes.append([box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]])
                        labels.append(BDD_DET_CLASS_TO_ID[cat])
                self.det_by_name[entry["name"]] = (boxes, labels)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        stem = os.path.splitext(name)[0]

        # ---- load image at original resolution ----
        img = Image.open(self.img_dir / name).convert("RGB")
        orig_w, orig_h = img.size

        # ---- load raw boxes/labels (original-image pixel coords) ----
        raw_boxes, raw_labels = self.det_by_name.get(name, ([], []))
        boxes_np = np.array(raw_boxes, dtype=np.float32).reshape(-1, 4)
        labels_np = np.array(raw_labels, dtype=np.int64)

        # ---- load drivable/lane masks at original resolution (if present) ----
        drivable_path = self.drivable_dir / f"{stem}.png"
        drivable_img = Image.open(drivable_path) if drivable_path.exists() else None

        lane_path = self.lane_dir / f"{stem}.png"
        lane_img = Image.open(lane_path) if lane_path.exists() else None

        # ---- train-time augmentation, applied pre-resize on original-res data ----
        if self.augment:
            x0, y0, cw, ch = _random_crop_box(orig_w, orig_h)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            if drivable_img is not None:
                drivable_img = drivable_img.crop((x0, y0, x0 + cw, y0 + ch))
            if lane_img is not None:
                lane_img = lane_img.crop((x0, y0, x0 + cw, y0 + ch))
            boxes_np, labels_np = _crop_and_clip_boxes(boxes_np, labels_np, x0, y0, cw, ch)

            # dims to scale against are now the crop dims, not the original image
            src_w, src_h = cw, ch

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if drivable_img is not None:
                    drivable_img = drivable_img.transpose(Image.FLIP_LEFT_RIGHT)
                if lane_img is not None:
                    lane_img = lane_img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes_np) > 0:
                    x1 = boxes_np[:, 0].copy()
                    x2 = boxes_np[:, 2].copy()
                    boxes_np[:, 0] = src_w - x2
                    boxes_np[:, 2] = src_w - x1

            img = _color_jitter(img)
        else:
            src_w, src_h = orig_w, orig_h

        # ---- resize image to model input size ----
        img = img.resize((self.out_w, self.out_h), Image.BILINEAR)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        sx = self.out_w / src_w
        sy = self.out_h / src_h

        # ---- finalize boxes/labels in resized-image coords ----
        if len(boxes_np) > 0:
            boxes_np[:, [0, 2]] *= sx
            boxes_np[:, [1, 3]] *= sy
            boxes = torch.from_numpy(boxes_np).float()
            labels = torch.from_numpy(labels_np).long()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        # ---- segmentation proxy mask ----
        seg_mask = np.zeros((self.out_h, self.out_w), dtype=np.int64)

        if drivable_img is not None:
            drv = drivable_img.resize((self.out_w, self.out_h), Image.NEAREST)
            drv_np = np.asarray(drv)
            seg_mask[drv_np < 2] = 1  # 0,1 = drivable variants; 2 = background

        if lane_img is not None:
            lane = lane_img.resize((self.out_w, self.out_h), Image.NEAREST)
            lane_np = np.asarray(lane)
            seg_mask[lane_np != 255] = 2  # lane overrides drivable where present

        seg_mask = torch.from_numpy(seg_mask)

        return {"image": image, "seg_mask": seg_mask, "boxes": boxes, "labels": labels, "name": name}


def bdd_collate_fn(batch):
    """Images/masks stack normally; boxes/labels stay as a per-image list
    since each image has a different number of objects."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    seg_masks = torch.stack([b["seg_mask"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    names = [b["name"] for b in batch]
    return {"images": images, "seg_masks": seg_masks, "boxes": boxes, "labels": labels, "names": names}