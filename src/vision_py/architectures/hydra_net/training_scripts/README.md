# convert_resolution.py:

## 1. Images (train + val)
python convert_resolution.py images   --src ~/Manas/Self-Drive/src/Data/bdd100k_images_100k/100k/train   --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/images/100k/train
python convert_resolution.py images   --src ~/Manas/Self-Drive/src/Data/bdd100k_images_100k/100k/val   --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/images/100k/val

## 2. Drivable-area / lane segmentation masks (nearest-neighbor, preserves label IDs)
python convert_resolution.py masks --src ~/Manas/Self-Drive/src/Data/bdd100k_seg_maps/color_labels/train --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/seg_maps/train
python convert_resolution.py masks --src ~/Manas/Self-Drive/src/Data/bdd100k_seg_maps/color_labels/val --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/seg_maps/val

python convert_resolution.py masks --src ~/Manas/Self-Drive/src/Data/bdd100k_drivable_maps/color_labels/train --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/drivable_maps/train
python convert_resolution.py masks --src ~/Manas/Self-Drive/src/Data/bdd100k_drivable_maps/color_labels/val --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/drivable_maps/val

## 3. Detection boxes (BDD100k det_20 json — coordinates rescaled, no image touched)
python convert_resolution.py det-json --src ~/Manas/Self-Drive/src/Data/bdd100k_labels/100k/train --dst ~/Manas/Self-Drive/src/Data/bdd100k_960x544/labels/train

# pretrain_bdd.py

torchrun --standalone --nproc_per_node=1 pretrain_bdd.py --bdd_root /data/bdd100k --model_variant base --batch_size 8
torchrun --standalone --nproc_per_node=1 pretrain_bdd.py --bdd_root /data/bdd100k --model_variant deep --batch_size 8
  
PYTHONPATH=.:.. python3 training_scripts/pretrain_bdd.py     --bdd_root ~/Manas/Self-Drive/src/Data/bdd100k_960x544     --model_variant base     --batch_size 8     --num_workers 8  <--init_from bdd_pretrained_base.pt --out bdd_pretrained_base_v2.pt
Distributed: False | world_size 1 | device cuda>
