import torch
from .model import HydraNet

torch.manual_seed(0)

H, W = 544, 960  # matches the 960x544 (WxH) input spec, divisible by 32
model = HydraNet(input_size=(H, W))
model.eval()

x = torch.randn(2, 3, H, W)

sign_boxes = [
    torch.tensor([[100.0, 80.0, 160.0, 140.0], [300.0, 200.0, 340.0, 240.0]]),
    torch.tensor([[50.0, 50.0, 90.0, 90.0]]),
]

with torch.no_grad():
    out = model(x, sign_boxes=sign_boxes)

print("lane_logits :", out["lane_logits"].shape)          # (B, num_lane_classes, H, W)
det = out["detection"]
print("cls_logits  :", det["cls_logits"].shape)             # (B, N, num_obj_classes)
print("reg_dist    :", det["reg_dist"].shape)                # (B, N, 4)
print("centerness  :", det["centerness"].shape)               # (B, N, 1)
print("boxes       :", det["boxes"].shape)                     # (B, N, 4)
print("sign_logits :", out["sign_logits"].shape)                 # (sum K_i, 2)

params = model.count_parameters()
print("\nParameter counts:")
for k, v in params.items():
    print(f"  {k:10s}: {v/1e6:.3f} M")

# sanity checks
assert out["lane_logits"].shape == (2, 4, H, W)
assert det["cls_logits"].shape[0] == 2
assert det["boxes"].shape[-1] == 4
assert out["sign_logits"].shape == (3, 2)  # 2 + 1 = 3 total proposed boxes
print("\nAll shape checks passed.")

# rough per-frame latency proxy: single image, CPU timing (just a sanity smoke test,
import time
model_single = model
xs = torch.randn(1, 3, H, W)
with torch.no_grad():
    for _ in range(2):
        model_single(xs)
    t0 = time.time()
    for _ in range(5):
        model_single(xs)
    t1 = time.time()
print(f"\nCPU eager-mode forward (no ROI boxes): {(t1 - t0) / 5 * 1000:.1f} ms/frame ")