"""
Draw the ground-truth boxes the DATALOADER hands to the model, on top of the image the
model actually sees. If the green boxes are not sitting on the objects, the labels are
wrong and no model can learn correct boxes from them.

Run from the hydra_net folder (same as training):
  PYTHONPATH=.:.. python3 training_scripts/check_labels.py \
      --bdd_root ~/Manas/Self-Drive/src/Data/bdd100k_960x544 --split train --n 8
Writes label_check/<split>_<k>_{plain,aug}.jpg
"""
import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from training_scripts.bdd_dataset_loader import (
    BDDDataset, BDD_DET_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
)


def to_pil(sample):
    img = sample["image"].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    img = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
    seg = sample["seg_mask"].numpy()
    over = np.asarray(img).copy()
    over[seg == 1] = (0.6 * over[seg == 1] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    img = Image.fromarray(over)
    d = ImageDraw.Draw(img)
    for b, l in zip(sample["boxes"].numpy(), sample["labels"].numpy()):
        d.rectangle([float(v) for v in b], outline=(255, 0, 0), width=2)
        d.text((b[0] + 2, max(0, b[1] - 11)), BDD_DET_CLASSES[int(l)], fill=(255, 255, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out_dir", default="label_check")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plain = BDDDataset(args.bdd_root, args.split, augment=False)
    aug = BDDDataset(args.bdd_root, args.split, augment=True)
    first = Image.open(plain.img_dir / plain.image_names[0])
    print(f"first image size: {first.size} | label frame: {plain.label_w}x{plain.label_h}")

    random.seed(0)
    for k, i in enumerate(random.sample(range(len(plain)), args.n)):
        to_pil(plain[i]).save(os.path.join(args.out_dir, f"{args.split}_{k}_plain.jpg"))
        to_pil(aug[i]).save(os.path.join(args.out_dir, f"{args.split}_{k}_aug.jpg"))
    print(f"saved {2 * args.n} images to {args.out_dir}/  -> boxes (red) must sit ON the objects")


if __name__ == "__main__":
    main()