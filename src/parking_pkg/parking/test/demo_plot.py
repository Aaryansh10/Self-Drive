import sys, os
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parking.slot_core import Params, VoxelMap, SlotTracker, extract_segments, find_slots, split_at_junctions
from synth import world, seg_points, visible

rng = np.random.default_rng(3)
kind = "comb"
segs_w, gt = world(kind, rng)
P = np.vstack([seg_points(a, b, rng=rng) for a, b in segs_w])
p, vm, tr = Params(), VoxelMap(0.05), SlotTracker()
hist = []
x = 0.0
while x <= 19.0:
    vm.add(P[visible(P, (x, 0, 0), rmax=10.0)])
    if abs(x / 1.5 - round(x / 1.5)) < 1e-6 or x > 18.9:
        pts = vm.points(p.min_voxel_hits)
        segs = extract_segments(pts, p)
        dets = find_slots(segs, p, ref_xy=(x, 0))
        tr.update(dets)
        hist.append((x, [(t.id, round(t.confidence, 2), t.slot.back_seen) for t in tr.tracks]))
    x += 0.3
for h in hist[-4:]:
    print("robot x=%.1f tracks(id,conf,back)=%s" % h)
ids = sorted({t.id for t in tr.tracks})
print("final track ids:", ids, "(expected 4 stable tracks)")

pts = vm.points(p.min_voxel_hits)
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(pts[:, 0], pts[:, 1], s=2, c="0.55", label="accumulated white-line map")
for s in split_at_junctions(segs, p):
    ax.plot([s.p0[0], s.p1[0]], [s.p0[1], s.p1[1]], c="k", lw=1)
for t in tr.tracks:
    poly = np.vstack([t.slot.polygon(), t.slot.polygon()[:1]])
    ax.plot(poly[:, 0], poly[:, 1], c="tab:green" if t.slot.back_seen else "tab:orange", lw=2)
    ax.arrow(*t.slot.center, *(1.0 * np.array([np.cos(t.slot.yaw), np.sin(t.slot.yaw)])), head_width=0.2, color="tab:red")
    ax.text(t.slot.center[0], t.slot.center[1] - 0.3, f"#{t.id} c={t.confidence:.2f}", ha="center", fontsize=8)
ax.plot(19.0, 0, "bs"); ax.text(19.0, 0.3, "robot", color="b")
ax.set_aspect("equal"); ax.set_title("Slot detector on synthetic comb lot (green = U with back line, arrow = into slot)")
ax.legend(loc="lower left"); plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/slot_detector_demo.png", dpi=110)
