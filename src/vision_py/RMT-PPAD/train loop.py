"""
Phase 3 — Gradient Routing Training Loop

Key point: you don't need manual gradient surgery here. Because
det_gca/det_head and seg_gca/seg_head are DISJOINT parameter sets that
only appear in their own forward path, standard autograd already routes
gradients correctly when you backprop a single combined loss:

    L = L_det + L_seg
    L.backward()

  - dL_det/d(seg params) = 0  (seg params never touched L_det's graph)
  - dL_seg/d(det params) = 0  (det params never touched L_seg's graph)
  - Backbone/AIFI/CCFM params receive the SUM of both gradients, since
    both L_det and L_seg flow back through them.

This is both correct and the most efficient option (single backward
pass, shared activations computed once). Below is that version, plus a
commented-out alternative using two backward() calls with separate
optimizers, for cases where you want per-branch LR/schedule control.
"""

import torch
from torch.utils.data import DataLoader

from RMT_ppad_phase_1 import RMTPPAD


def det_loss_fn(det_out, det_targets):
    # placeholder — plug in your Hungarian-matched cls + L1/GIoU loss
    cls_l = torch.nn.functional.cross_entropy(
        det_out["pred_logits"].flatten(0, 1), det_targets["labels"].flatten(0, 1)
    )
    box_l = torch.nn.functional.l1_loss(det_out["pred_boxes"], det_targets["boxes"])
    return cls_l + box_l


def seg_loss_fn(seg_out, seg_targets):
    # 4-channel binary masks -> BCE (seg_out already sigmoided)
    return torch.nn.functional.binary_cross_entropy(seg_out, seg_targets)


def train_one_epoch(model: RMTPPAD, loader: DataLoader, opt, device):
    model.train()
    for im, det_t, seg_t in loader:
        im, det_t, seg_t = im.to(device), det_t, seg_t.to(device)

        out = model(im)  # single forward, shared trunk computed once

        l_det = det_loss_fn(out["detection"], det_t)
        l_seg = seg_loss_fn(out["segmentation"], seg_t)
        L = l_det + l_seg

        opt.zero_grad()
        L.backward()   # autograd routes grads correctly per param set above
        opt.step()

        print(f"L_det={l_det.item():.4f}  L_seg={l_seg.item():.4f}  L={L.item():.4f}")


# ---------------------------------------------------------------------------
# Setup: one optimizer covering all params is simplest and correct, since
# gradient routing is already handled by the graph. Param groups let you
# give the two branches different LRs if desired.
# ---------------------------------------------------------------------------

def build_optimizer(model: RMTPPAD, lr_shared=1e-4, lr_det=1e-4, lr_seg=1e-4):
    return torch.optim.AdamW([
        {"params": model.shared_parameters(), "lr": lr_shared},
        {"params": model.detection_parameters(), "lr": lr_det},
        {"params": model.segmentation_parameters(), "lr": lr_seg},
    ])


# ---------------------------------------------------------------------------
# Alternative: two separate backward() calls / two optimizers. Only needed
# if you want to, e.g., step the shared trunk on a different schedule than
# the branches, or inspect each branch's gradient on the trunk separately.
# Requires retain_graph=True on the first backward since both calls share
# the backbone/AIFI/CCFM subgraph.
# ---------------------------------------------------------------------------
"""
def train_one_epoch_split(model, loader, opt_shared, opt_det, opt_seg, device):
    model.train()
    for im, det_t, seg_t in loader:
        im, seg_t = im.to(device), seg_t.to(device)
        out = model(im)

        l_det = det_loss_fn(out["detection"], det_t)
        l_seg = seg_loss_fn(out["segmentation"], seg_t)

        opt_shared.zero_grad(); opt_det.zero_grad(); opt_seg.zero_grad()

        l_det.backward(retain_graph=True)  # populates grads on shared + det params
        l_seg.backward()                   # accumulates onto shared, populates seg params

        opt_shared.step(); opt_det.step(); opt_seg.step()
"""


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RMTPPAD().to(device)
    opt = build_optimizer(model)
    # loader = DataLoader(YourDataset(), batch_size=4, shuffle=True)
    # train_one_epoch(model, loader, opt, device)