"""
BDD100K dataset loader for HydraNet pretraining.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from PIL import Image, ImageEnhance, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# BDD100K "Detection 2020" categories
BDD_DET_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]
BDD_DET_CLASS_TO_ID = {name: i for i, name in enumerate(BDD_DET_CLASSES)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _random_crop_box(orig_w, orig_h, scale=(0.75, 1.0), ratio=(0.9, 1.1111)):
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

    return 0, 0, orig_w, orig_h


def _crop_and_clip_boxes(boxes, labels, x0, y0, cw, ch, min_size=2.0):
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
    def __init__(self, root, split="train", img_size=(544, 960), augment=None, img_dir=None,
                 label_size=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.out_h, self.out_w = img_size
        self.augment = (split == "train") if augment is None else augment

        # Handle path layouts for images and labels
        self.img_dir = self.root / "images" / "100k" / split
        if not self.img_dir.exists():
            self.img_dir = self.root / "images" / split
        if img_dir is not None:  # explicit folder that directly contains the images
            self.img_dir = Path(img_dir).expanduser()

        self.drivable_dir = self.root / "drivable_maps" / split
        self.seg_dir = self.root / "seg_maps" / split
        self.labels_dir = self.root / "labels" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found at: {self.img_dir}")

        self.image_names = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        print(f"[{split}] Pre-caching JSON labels for {len(self.image_names)} images...")
        self.label_cache = {}
        for name in self.image_names:
            stem = os.path.splitext(name)[0]
            json_path = self.labels_dir / f"{stem}.json"
            self.label_cache[stem] = self._load_json_labels(json_path)

        self.label_w, self.label_h = self._resolve_label_frame(label_size)
        print(f"[{split}] Successfully cached labels. Loaded dataset with {len(self.image_names)} images from {self.img_dir}")

    def _resolve_label_frame(self, label_size):
        """Returns (label_w, label_h): the pixel frame the JSON boxes live in.
        Boxes are later scaled by (image_w / label_w, image_h / label_h)."""
        img0 = Image.open(self.img_dir / self.image_names[0])
        W, H = img0.size
        if isinstance(label_size, (tuple, list)):
            return int(label_size[0]), int(label_size[1])
        if label_size == "raw":
            return 1280, 720
        if label_size == "image":
            return W, H

        xs = ys = 0.0
        n = 0
        for boxes, _ in self.label_cache.values():
            for b in boxes:
                xs, ys, n = max(xs, b[2]), max(ys, b[3]), n + 1
        if n < 100:
            print(f"[{self.split}] WARNING: only {n} boxes, cannot detect label frame; assuming raw 1280x720")
            return 1280, 720
        for fw, fh in ((1280, 720), (960, 544), (W, H)):
            if 0.97 * fw <= xs <= 1.005 * fw and ys <= 1.005 * fh:
                print(f"[{self.split}] label frame detected: {fw}x{fh} (max box x2={xs:.0f}, y2={ys:.0f}); "
                      f"image {W}x{H} -> box scale {W / fw:.4f}, {H / fh:.4f}")
                return fw, fh
        print(f"[{self.split}] WARNING: label frame unclear (max x2={xs:.0f}, y2={ys:.0f}, image {W}x{H}); "
              f"assuming labels are already in image frame. Pass label_size=... to override.")
        return W, H

    def _load_json_labels(self, json_path):
        boxes, labels = [], []
        if not json_path.exists():
            return boxes, labels

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract objects array from data["frames"][0]["objects"]
        objects_list = []
        if isinstance(data, dict):
            frames = data.get("frames", [])
            if frames and isinstance(frames[0], dict):
                objects_list = frames[0].get("objects", [])

        for obj in objects_list:
            if not isinstance(obj, dict):
                continue

            cat = obj.get("category")
            box2d = obj.get("box2d")

            if cat in BDD_DET_CLASS_TO_ID and isinstance(box2d, dict):
                x1 = box2d.get("x1")
                y1 = box2d.get("y1")
                x2 = box2d.get("x2")
                y2 = box2d.get("y2")

                if all(v is not None for v in (x1, y1, x2, y2)):
                    # kept in the label file's own coordinate frame; rescaled to the
                    # real image size in __getitem__ (see _resolve_label_frame)
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    labels.append(BDD_DET_CLASS_TO_ID[cat])

        return boxes, labels

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        stem = os.path.splitext(name)[0]

        img = Image.open(self.img_dir / name).convert("RGB")
        orig_w, orig_h = img.size

        # Retrieve cached labels directly from memory
        raw_boxes, raw_labels = self.label_cache.get(stem, ([], []))

        boxes_np = np.array(raw_boxes, dtype=np.float32).reshape(-1, 4)
        labels_np = np.array(raw_labels, dtype=np.int64)
        if len(boxes_np) > 0:  # label frame -> actual image pixels
            boxes_np[:, [0, 2]] *= orig_w / self.label_w
            boxes_np[:, [1, 3]] *= orig_h / self.label_h

        # Fast direct loading
        drivable_path = self.drivable_dir / f"{stem}_drivable_color.png"
        try:
            drivable_img = Image.open(drivable_path)
        except (FileNotFoundError, OSError):
            drivable_img = None

        # NOTE: seg_maps/*_train_color.png are Cityscapes-style SEMANTIC colour maps
        # (road, sidewalk, car, ...) and contain NO lane-marking class, so they cannot
        # fill class 2. Left disabled; class 2 stays unused until a real lane map exists.
        lane_img = None

        # make the maps pixel-aligned with the image before cropping/flipping
        if drivable_img is not None and drivable_img.size != img.size:
            drivable_img = drivable_img.resize(img.size, Image.NEAREST)
        if lane_img is not None and lane_img.size != img.size:
            lane_img = lane_img.resize(img.size, Image.NEAREST)

        if self.augment:
            x0, y0, cw, ch = _random_crop_box(orig_w, orig_h)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            if drivable_img is not None:
                drivable_img = drivable_img.crop((x0, y0, x0 + cw, y0 + ch))
            if lane_img is not None:
                lane_img = lane_img.crop((x0, y0, x0 + cw, y0 + ch))
            boxes_np, labels_np = _crop_and_clip_boxes(boxes_np, labels_np, x0, y0, cw, ch)

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
            boxes_np, labels_np = _crop_and_clip_boxes(boxes_np, labels_np, 0, 0, orig_w, orig_h)

        img = img.resize((self.out_w, self.out_h), Image.BILINEAR)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        sx = self.out_w / src_w
        sy = self.out_h / src_h

        if len(boxes_np) > 0:
            boxes_np[:, [0, 2]] *= sx
            boxes_np[:, [1, 3]] *= sy
            boxes = torch.from_numpy(boxes_np).float()
            labels = torch.from_numpy(labels_np).long()
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        seg_mask = np.zeros((self.out_h, self.out_w), dtype=np.int64)

        if drivable_img is not None:
            # colour map: black = background, red = direct drivable, blue = alternative
            drv = drivable_img.convert("RGB").resize((self.out_w, self.out_h), Image.NEAREST)
            drv_np = np.asarray(drv).max(axis=2)
            seg_mask[drv_np > 0] = 1

        if lane_img is not None:
            lane = lane_img.resize((self.out_w, self.out_h), Image.NEAREST)
            lane_np = np.asarray(lane)
            seg_mask[lane_np != 255] = 2

        seg_mask = torch.from_numpy(seg_mask)

        return {"image": image, "seg_mask": seg_mask, "boxes": boxes, "labels": labels, "name": name}

def bdd_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    seg_masks = torch.stack([b["seg_mask"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    names = [b["name"] for b in batch]
    return {"images": images, "seg_masks": seg_masks, "boxes": boxes, "labels": labels, "names": names}