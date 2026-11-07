#!/usr/bin/env python3
"""Export HydraNet to ONNX with the tensor names the C++ node expects.

  python3 export_onnx.py --weights sim_finetuned.pt --out hydranet.onnx [--variant base] [--h 544 --w 960]

Normalisation is NOT baked in: the C++ preprocess kernel applies (x/255 - mean)/std.
Default is an fp16 export (needs a GPU) -> fp16 engine. Use --fp32 for an fp32 engine.
"""
import argparse

import torch

from vision_py.architectures.hydra_net.model import build_hydranet
from vision_py.model_helpers import load_weights


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        d = out["detection"]
        # boxes stay fp32 (fp16 only resolves 0.5 px at x~512-1024); the C++ side reads each dtype
        return out["lane_logits"], d["cls_logits"], d["centerness"], d["boxes"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="hydranet.onnx")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--h", type=int, default=544)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--fp32", action="store_true", help="export an fp32 model (default: fp16 for TensorRT)")
    # The dynamo exporter (default in recent PyTorch) emits >= opset 18; asking for 17 triggers a failed down-conversion.
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--verify", action="store_true", help="compare against onnxruntime")
    a = ap.parse_args()

    model = build_hydranet(a.variant)
    load_weights(model, a.weights)
    # Newer TensorRT builds strongly-typed networks: the ONNX dtypes ARE the engine precision.
    dev = "cuda" if (torch.cuda.is_available() and not a.fp32) else "cpu"  # fp16 ops need the GPU
    model = model.float() if a.fp32 else model.half()
    model = Wrap(model.to(dev)).eval()
    x = torch.zeros(1, 3, a.h, a.w, dtype=torch.float32 if a.fp32 else torch.float16, device=dev)
    names = ["lane_logits", "cls_logits", "centerness", "boxes"]
    with torch.no_grad():
        torch.onnx.export(model, x, a.out, input_names=["image"], output_names=names,
                          opset_version=a.opset, do_constant_folding=True)
    print("wrote", a.out)

    if a.verify:
        import numpy as np
        import onnxruntime as ort

        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        if not a.fp32 and "CUDAExecutionProvider" not in providers:
            print("verify skipped: fp16 model needs onnxruntime-gpu (CUDAExecutionProvider). "
                  "Install it, or re-run with --fp32 --verify.")
            return

        xr = torch.rand(1, 3, a.h, a.w).to(dev, x.dtype)
        with torch.no_grad():
            ref = [t.detach().cpu().float().numpy() for t in model(xr)]
        sess = ort.InferenceSession(a.out, providers=providers)
        got = sess.run(None, {"image": xr.cpu().numpy()})
        for n, r, g in zip(names, ref, got):
            g = g.astype(np.float32)
            print(f"{n:12s} max|diff| = {np.abs(r - g).max():.3e}  shape {g.shape}")


if __name__ == "__main__":
    main()