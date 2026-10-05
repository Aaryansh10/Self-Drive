#!/usr/bin/env python3
"""
Resize BDD100k dataset (images + labels) to a target resolution for
multi-task training (detection + drivable-area / lane segmentation).

Handles three asset types differently, because naive resizing corrupts
two of them:

  1. RGB images            -> cv2.resize with INTER_AREA (good for downscaling)
  2. Segmentation masks     -> cv2.resize with INTER_NEAREST
                               (masks are label IDs / colors, not pixel
                               intensities -- linear/area interpolation would
                               invent new "blended" class IDs at edges)
  3. Detection labels (json)-> coordinates scaled by (new/old) ratio,
                               pixel values are NOT touched

BDD100k native resolution is 1280x720. This script defaults to a direct
(non-letterboxed) resize to 960x544, matching what most BDD100k multitask
baselines (e.g. YOLOP, HybridNets) do. If you need aspect-ratio-preserving
letterbox resizing instead, use --letterbox.

Usage examples
--------------
# Images only (e.g. 100k/train, 100k/val)
python resize_bdd100k.py images \
    --src /data/bdd100k/images/100k/train \
    --dst /data/bdd100k_960x544/images/100k/train

# Segmentation masks (drivable area or lane, single-channel label PNGs)
python resize_bdd100k.py masks \
    --src /data/bdd100k/labels/drivable/masks/train \
    --dst /data/bdd100k_960x544/labels/drivable/masks/train

# Detection labels (bdd100k_labels_images_det_coco / det_20 style json)
python resize_bdd100k.py det-json \
    --src /data/bdd100k/labels/det_20/det_train.json \
    --dst /data/bdd100k_960x544/labels/det_20/det_train.json \
    --orig-w 1280 --orig-h 720

Run with --letterbox to pad instead of stretch (keeps aspect ratio, adds
gray bars). If you use --letterbox for images, use it for masks and
det-json too so everything stays aligned.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png")


def compute_letterbox_params(orig_w, orig_h, new_w, new_h):
    """Scale to fit inside (new_w, new_h) preserving aspect ratio, then pad."""
    scale = min(new_w / orig_w, new_h / orig_h)
    resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return scale, resized_w, resized_h, top, bottom, left, right


def resize_one_file(rel_path, src_root, dst_root, new_w, new_h, interp, letterbox, pad_value):
    src_path = os.path.join(src_root, rel_path)
    dst_path = os.path.join(dst_root, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"FAILED to read: {src_path}"

    h, w = img.shape[:2]

    if letterbox:
        scale, rw, rh, top, bottom, left, right = compute_letterbox_params(w, h, new_w, new_h)
        resized = cv2.resize(img, (rw, rh), interpolation=interp)
        border_value = pad_value if img.ndim == 2 else (pad_value,) * img.shape[2]
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value
        )
    else:
        out = cv2.resize(img, (new_w, new_h), interpolation=interp)

    cv2.imwrite(dst_path, out)
    return None


def collect_files(src_root):
    files = []
    for dirpath, _, filenames in os.walk(src_root):
        for fname in filenames:
            if fname.lower().endswith(IMG_EXTS):
                rel = os.path.relpath(os.path.join(dirpath, fname), src_root)
                files.append(rel)
    return files


def run_resize(src, dst, new_w, new_h, interp, letterbox, pad_value, workers):
    files = collect_files(src)
    if not files:
        print(f"No image files found under {src}")
        return

    fn = partial(
        resize_one_file,
        src_root=src,
        dst_root=dst,
        new_w=new_w,
        new_h=new_h,
        interp=interp,
        letterbox=letterbox,
        pad_value=pad_value,
    )

    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for result in tqdm(ex.map(fn, files), total=len(files), desc=f"Resizing -> {dst}"):
            if result:
                errors.append(result)

    print(f"Done: {len(files) - len(errors)}/{len(files)} succeeded.")
    if errors:
        print(f"{len(errors)} failures, e.g.:")
        for e in errors[:10]:
            print(" ", e)


def resize_det_json(src, dst, orig_w, orig_h, new_w, new_h, letterbox):
    """Rescale BDD100k detection-style JSON label boxes/polygons in place.

    Works for the standard BDD100k det_20 format:
    [{ "name": "...", "labels": [{"box2d": {"x1","y1","x2","y2"}, ...}, ...] }, ...]
    Also rescales "poly2d" vertices (lane/drivable polygon annotations) if present.
    """
    with open(src, "r") as f:
        data = json.load(f)

    if letterbox:
        scale, rw, rh, top, bottom, left, right = compute_letterbox_params(
            orig_w, orig_h, new_w, new_h
        )

        def tx(x):
            return x * scale + left

        def ty(y):
            return y * scale + top
    else:
        sx = new_w / orig_w
        sy = new_h / orig_h

        def tx(x):
            return x * sx

        def ty(y):
            return y * sy

    n_boxes = 0
    n_polys = 0
    for item in data:
        for lbl in item.get("labels", []):
            box = lbl.get("box2d")
            if box:
                box["x1"], box["y1"] = tx(box["x1"]), ty(box["y1"])
                box["x2"], box["y2"] = tx(box["x2"]), ty(box["y2"])
                n_boxes += 1
            poly = lbl.get("poly2d")
            if poly:
                for seg in poly:
                    seg["vertices"] = [[tx(px), ty(py)] for px, py in seg["vertices"]]
                    n_polys += 1

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f)

    print(f"Rescaled {n_boxes} boxes and {n_polys} polygon segments.")
    print(f"Wrote {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = dict(
        new_w=960,
        new_h=544,
    )

    p_img = sub.add_parser("images", help="Resize RGB images (INTER_AREA)")
    p_img.add_argument("--src", required=True)
    p_img.add_argument("--dst", required=True)
    p_img.add_argument("--new-w", type=int, default=common["new_w"])
    p_img.add_argument("--new-h", type=int, default=common["new_h"])
    p_img.add_argument("--letterbox", action="store_true", help="Pad to preserve aspect ratio instead of stretching")
    p_img.add_argument("--pad-value", type=int, default=114, help="Gray padding value used with --letterbox")
    p_img.add_argument("--workers", type=int, default=os.cpu_count())

    p_mask = sub.add_parser("masks", help="Resize label/segmentation masks (INTER_NEAREST)")
    p_mask.add_argument("--src", required=True)
    p_mask.add_argument("--dst", required=True)
    p_mask.add_argument("--new-w", type=int, default=common["new_w"])
    p_mask.add_argument("--new-h", type=int, default=common["new_h"])
    p_mask.add_argument("--letterbox", action="store_true")
    p_mask.add_argument("--pad-value", type=int, default=0, help="Fill value for padded/ignore regions (usually 0 or 255)")
    p_mask.add_argument("--workers", type=int, default=os.cpu_count())

    p_det = sub.add_parser("det-json", help="Rescale detection/polygon JSON label coordinates")
    p_det.add_argument("--src", required=True, help="Path to source det_*.json")
    p_det.add_argument("--dst", required=True, help="Path to write rescaled json")
    p_det.add_argument("--orig-w", type=int, default=1280)
    p_det.add_argument("--orig-h", type=int, default=720)
    p_det.add_argument("--new-w", type=int, default=common["new_w"])
    p_det.add_argument("--new-h", type=int, default=common["new_h"])
    p_det.add_argument("--letterbox", action="store_true")

    args = p.parse_args()

    if args.mode == "images":
        run_resize(
            args.src, args.dst, args.new_w, args.new_h,
            interp=cv2.INTER_AREA, letterbox=args.letterbox,
            pad_value=args.pad_value, workers=args.workers,
        )
    elif args.mode == "masks":
        run_resize(
            args.src, args.dst, args.new_w, args.new_h,
            interp=cv2.INTER_NEAREST, letterbox=args.letterbox,
            pad_value=args.pad_value, workers=args.workers,
        )
    elif args.mode == "det-json":
        resize_det_json(
            args.src, args.dst, args.orig_w, args.orig_h,
            args.new_w, args.new_h, args.letterbox,
        )


if __name__ == "__main__":
    main()