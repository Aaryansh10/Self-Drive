import os
import glob
import time
import cv2
import torch
import torch.amp
import torchvision.transforms as T
from PIL import Image

# 1. Update this import path to match your HydraNet class definition
from hydra_net.model import HydraNet 


@torch.no_grad()
def benchmark_test_images(
    image_dir="/home/sunanda/Manas/Self-Drive/src/Data/bdd100k_images_100k/100k/images/test",
    checkpoint_path=None,
    model_variant="base",
    img_size=(544, 960),  # (H, W) expected by model
    num_samples=100,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # Load Model
    model = HydraNet().to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    model.eval()

    # Get list of image paths from your test directory
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:num_samples]
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in path: {image_dir}")

    # Image Preprocessing Transform
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Warmup CUDA kernels
    dummy = torch.randn(1, 3, *img_size, device=device)
    with torch.amp.autocast("cuda"):
        for _ in range(30):
            _ = model(dummy)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    gpu_latencies = []
    e2e_latencies = []

    print(f"\nBenchmarking {len(image_paths)} images from test set...")
    for path in image_paths:
        t_e2e_start = time.perf_counter()

        # Step A: Load & Preprocess (CPU)
        img = Image.open(path).convert("RGB")
        tensor_img = transform(img).unsqueeze(0).to(device, non_blocking=True)

        # Step B: Model Forward Pass (GPU timing)
        start_evt.record()
        with torch.amp.autocast("cuda"):
            _ = model(tensor_img)
        end_evt.record()

        torch.cuda.synchronize()
        t_e2e_end = time.perf_counter()

        gpu_latencies.append(start_evt.elapsed_time(end_evt))  # ms
        e2e_latencies.append((t_e2e_end - t_e2e_start) * 1000.0)  # ms

    avg_gpu_ms = sum(gpu_latencies) / len(gpu_latencies)
    avg_e2e_ms = sum(e2e_latencies) / len(e2e_latencies)

    print("\n" + "=" * 50)
    print(" INFERENCE BENCHMARK RESULTS (TEST SET)")
    print("=" * 50)
    print(f" Test Path        : {image_dir}")
    print(f" Images Processed : {len(image_paths)}")
    print(f" GPU Model Speed  : {avg_gpu_ms:.2f} ms/frame ({1000.0/avg_gpu_ms:.1f} FPS)")
    print(f" Full Pipeline    : {avg_e2e_ms:.2f} ms/frame ({1000.0/avg_e2e_ms:.1f} FPS)")
    print("=" * 50)


if __name__ == "__main__":
    benchmark_test_images()