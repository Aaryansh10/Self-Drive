"""
Sim dataset loader (labelme-style JSON, one per image).

Layout:
    <root>/images/<stem>.png|jpg
    <root>/labels/<stem>.json     polygons -> lane / stop-line masks, rectangles -> boxes

Returns the same dict as BDDDataset (image, seg_mask, boxes, labels, name), so the
existing collate function, target assigner and losses work unchanged.

Seg classes (NUM_SEG = 4):  0 background, 1 white_lane, 2 yellow_lane, 3 stop_line
Object classes (NUM_OBJ = 5): barrel, tire, pothole, sign, pedestrian
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile, ImageEnhance
from torch.utils.data import Dataset

from training_scripts.bdd_dataset_loader import (
    _random_crop_box, _crop_and_clip_boxes, _color_jitter,
    IMAGENET_MEAN, IMAGENET_STD, bdd_collate_fn,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

SIM_SEG_CLASSES = ["background", "white_lane", "yellow_lane", "stop_line"]
SIM_OBJ_CLASSES = ["barrel", "tire", "pothole", "sign", "pedestrian"]
_SEG_ID = {n: i for i, n in enumerate(SIM_SEG_CLASSES) if i > 0}
_OBJ_ID = {n: i for i, n in enumerate(SIM_OBJ_CLASSES)}

_ALIASES = {
    "tyre": "tire",
    "tyres": "tire",
    "tires": "tire",
    "barrels": "barrel",
    "pedestrians": "pedestrian",
    "signs": "sign",
    "potholes": "pothole",
    "yellow_line": "yellow_lane",
    "yellow-line": "yellow_lane",
    "white_line": "white_lane",
    "white-line": "white_lane",
    "stopline": "stop_line",
    "stop-line": "stop_line",
}

sim_collate_fn = bdd_collate_fn


def _add_random_shadow(img, p=0.5):
    """
    Applies a random polygonal shadow overlay to a PIL image.
    
    Args:
        img: PIL Image in RGB mode.
        p: Probability of applying the shadow.
    """
    if random.random() > p:
        return img

    w, h = img.size
    # Generate random polygon coordinates covering a slice of the image
    num_vertices = random.choice([3, 4, 5])
    points = []
    for _ in range(num_vertices):
        x = random.randint(0, w)
        y = random.randint(0, h)
        points.append((x, y))

    # Create a grayscale mask for the shadow
    shadow_mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(shadow_mask)
    draw.polygon(points, fill=0)

    # Darkness factor: 0.3 to 0.7 (lower is darker)
    darkness = random.uniform(0.3, 0.7)
    darkened_img = ImageEnhance.Brightness(img).enhance(darkness)

    # Composite original image and darkened image using shadow_mask
    shadowed_img = Image.composite(img, darkened_img, shadow_mask)
    return shadowed_img


class SimDataset(Dataset):
    def __init__(self, root, split="train", img_size=(544, 960), augment=None,
                 val_frac=0.10, test_frac=0.10, seed=0, flip=False, shadow_p=0.5):
        super().__init__()
        self.root = Path(root).expanduser()
        self.out_h, self.out_w = img_size
        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.flip = flip
        self.shadow_p = shadow_p
        
        img_dir, label_dir = self.root / "images", self.root / "labels"
        if not img_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"expected {img_dir} and {label_dir}")

        img_by_stem = {p.stem: p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        stems = sorted(p.stem for p in label_dir.glob("*.json") if p.stem in img_by_stem)
        if not stems:
            raise FileNotFoundError("no <stem>.json that has a matching image <stem>.png/.jpg")

        order = stems[:]
        random.Random(seed).shuffle(order)
        n_test = int(len(order) * test_frac)
        n_val = int(len(order) * val_frac)
        test = set(order[:n_test])
        val = set(order[n_test:n_test + n_val])
        if split == "test":
            stems = [s for s in stems if s in test]
        elif split == "val":
            stems = [s for s in stems if s in val]
        elif split == "train":
            stems = [s for s in stems if s not in test and s not in val]
        if not stems:
            raise RuntimeError(f"split '{split}' is empty; check val_frac / test_frac")

        self.stems = stems
        self.img_paths = {s: img_by_stem[s] for s in stems}
        self.cache = {}
        census, unknown = Counter(), Counter()
        for s in stems:
            rec = self._parse(label_dir / f"{s}.json", census, unknown)
            self.cache[s] = rec

        print(f"[sim/{split}] {len(stems)} images | label census: {dict(census)}")
        if unknown:
            print(f"[sim/{split}] WARNING skipped unknown labels: {dict(unknown)} "
                  f"(known seg={list(_SEG_ID)}, objects={SIM_OBJ_CLASSES})")

    @staticmethod
    def _parse(path, census, unknown):
        d = json.load(open(path))
        rec = {"w": float(d["imageWidth"]), "h": float(d["imageHeight"]),
               "polys": [], "boxes": [], "labels": []}
        for sh in d.get("shapes", []):
            name = _ALIASES.get(sh["label"], sh["label"])
            pts = sh["points"]
            if name in _SEG_ID and sh["shape_type"] in ("polygon", "linestrip", "line"):
                rec["polys"].append((_SEG_ID[name], pts))
                census[name] += 1
            elif name in _OBJ_ID and sh["shape_type"] in ("rectangle", "polygon"):
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                rec["boxes"].append([min(xs), min(ys), max(xs), max(ys)])
                rec["labels"].append(_OBJ_ID[name])
                census[name] += 1
            else:
                unknown[sh["label"]] += 1
        rec["polys"].sort(key=lambda t: t[0])  # stop_line painted last (on top)
        return rec

    def object_classes_per_image(self):
        """Returns a list of sets of object class indices present in each cached image."""
        return [set(self.cache[s]["labels"]) for s in self.stems]

    def seg_pixel_counts(self):
        """Approximates segmentation pixel frequency across dataset samples for loss weighting."""
        counts = np.zeros(len(SIM_SEG_CLASSES), dtype=np.int64)
        for idx in range(len(self)):
            item = self[idx]
            mask = item["seg_mask"].numpy()
            unq, cnts = np.unique(mask, return_counts=True)
            for u, c in zip(unq, cnts):
                if u < len(SIM_SEG_CLASSES):
                    counts[u] += c
        return counts

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = Image.open(self.img_paths[stem]).convert("RGB")
        ow, oh = img.size
        rec = self.cache[stem]
        sx, sy = ow / rec["w"], oh / rec["h"]

        mask = Image.new("L", (ow, oh), 0)
        draw = ImageDraw.Draw(mask)
        for cid, pts in rec["polys"]:
            xy = [(p[0] * sx, p[1] * sy) for p in pts]
            if len(xy) >= 3:
                draw.polygon(xy, fill=cid)
            elif len(xy) == 2:
                draw.line(xy, fill=cid, width=2)

        boxes = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.array(rec["labels"], dtype=np.int64)
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        if self.augment:
            x0, y0, cw, ch = _random_crop_box(ow, oh)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            mask = mask.crop((x0, y0, x0 + cw, y0 + ch))
            boxes, labels = _crop_and_clip_boxes(boxes, labels, x0, y0, cw, ch)
            src_w, src_h = cw, ch
            if self.flip and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes):
                    x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
                    boxes[:, 0], boxes[:, 2] = src_w - x2, src_w - x1
            
            # Apply color jitter and random shadow augmentation
            img = _color_jitter(img)
            img = _add_random_shadow(img, p=self.shadow_p)
        else:
            src_w, src_h = ow, oh
            boxes, labels = _crop_and_clip_boxes(boxes, labels, 0, 0, ow, oh)

        img = img.resize((self.out_w, self.out_h), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(arr.transpose(2, 0, 1)).float()

        if len(boxes):
            boxes[:, [0, 2]] *= self.out_w / src_w
            boxes[:, [1, 3]] *= self.out_h / src_h
            boxes_t = torch.from_numpy(boxes).float()
            labels_t = torch.from_numpy(labels).long()
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        mask = mask.resize((self.out_w, self.out_h), Image.NEAREST)
        seg_mask = torch.from_numpy(np.asarray(mask).astype(np.int64))
        return {"image": image, "seg_mask": seg_mask, "boxes": boxes_t,
                "labels": labels_t, "name": self.img_paths[stem].name}