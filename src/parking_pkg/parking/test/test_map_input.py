import sys, os
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
import time
import numpy as np
from parking.slot_core import Params, grid_to_points, extract_segments, find_slots, thin_binary
from synth import world
from test_slot_core import match


def make_grid(segs, res, ox, oy, oyaw, W, H, band_m, halo_m):
    """Rasterise world segments as thick bands (like an inflated costmap): 100 core, 50 halo."""
    arr = np.zeros((H, W), np.int8)
    c, s = np.cos(oyaw), np.sin(oyaw)
    for a, b in segs:
        a, b = np.array(a, float), np.array(b, float)
        n = int(np.linalg.norm(b - a) / (res * 0.5)) + 2
        P = a + np.linspace(0, 1, n)[:, None] * (b - a)
        for w_m, val in ((halo_m, 50), (band_m, 100)):
            k = int(w_m / res / 2)
            for p in P:
                d = p - [ox, oy]
                gx, gy = c * d[0] + s * d[1], -s * d[0] + c * d[1]
                j, i = int(gx / res), int(gy / res)
                i0, i1, j0, j1 = max(i - k, 0), min(i + k + 1, H), max(j - k, 0), min(j + k + 1, W)
                if i0 < i1 and j0 < j1:
                    blk = arr[i0:i1, j0:j1]
                    blk[blk < val] = val
    return arr


rng = np.random.default_rng(0)
for kind in ("isolated", "comb"):
    segs, gt = world(kind, rng)
    for oyaw, band, res in ((0.0, 0.30, 0.05), (0.25, 0.30, 0.05), (0.0, 0.50, 0.10)):
        W = int(60 / res); H = int(50 / res)
        grid = make_grid(segs, res, -8.0, -26.0, oyaw, W, H, band, band + 0.3)
        t0 = time.time()
        pts = grid_to_points(grid.ravel(), W, H, res, -8.0, -26.0, oyaw, (16.0, 0.0), 14.0)
        t1 = time.time()
        p = Params()
        slots = find_slots(extract_segments(pts, p), p, ref_xy=(16.0, 0.0))
        ms = match(slots, gt)
        ok = sum(1 for r in ms if r and r[0] < 0.3 and r[1] < 0.1)
        print(f"[{kind}] band={band:.2f}m res={res:.2f} grid_yaw={oyaw:.2f}: thinned pts={len(pts)} "
              f"matched {ok}/{len(gt)} extra={len(slots)-ok} convert={1000*(t1-t0):.0f}ms fit={1000*(time.time()-t1):.0f}ms")
        assert ok == len(gt), "slot missed"
print("OK")
