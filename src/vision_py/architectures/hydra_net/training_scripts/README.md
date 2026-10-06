# convert_resolution.py:

## 1. Images (train + val)
python resize_bdd100k.py images --src /data/bdd100k/images/100k/train --dst /data/bdd100k_960x544/images/100k/train
python resize_bdd100k.py images --src /data/bdd100k/images/100k/val   --dst /data/bdd100k_960x544/images/100k/val

## 2. Drivable-area / lane segmentation masks (nearest-neighbor, preserves label IDs)
python resize_bdd100k.py masks --src /data/bdd100k/labels/drivable/masks/train --dst /data/bdd100k_960x544/labels/drivable/masks/train
python resize_bdd100k.py masks --src /data/bdd100k/labels/lane/masks/train    --dst /data/bdd100k_960x544/labels/lane/masks/train

## 3. Detection boxes (BDD100k det_20 json — coordinates rescaled, no image touched)
python resize_bdd100k.py det-json --src /data/bdd100k/labels/det_20/det_train.json --dst /data/bdd100k_960x544/labels/det_20/det_train.json

# pretrain_bdd.py

torchrun --standalone --nproc_per_node=3 pretrain_bdd.py --bdd_root /data/bdd100k --model_variant base --batch_size 8
torchrun --standalone --nproc_per_node=3 pretrain_bdd.py --bdd_root /data/bdd100k --model_variant deep --batch_size 8