import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from parking.slot_core import Params, extract_segments, find_slots, SlotTracker
from synth import drive_and_map


def match(det, gt):
    """Return list of (pos_err, yaw_err, w_err, d_err) for each GT matched by a detection."""
    res = []
    for g in gt:
        gc = np.array(g["center"]); best = None
        for d in det:
            e = np.linalg.norm(d.center - gc)
            if best is None or e < best[0]:
                best = (e, abs(((d.yaw - g["yaw"] + np.pi) % (2*np.pi)) - np.pi), abs(d.width - g["width"]), d.depth - g["depth"])
        res.append(best)
    return res


def run(kind, x_end, rmax=7.5, lateral=0.0):
    p = Params()
    vm, gt, pose, _ = drive_and_map(kind, x_end=x_end, rmax=rmax, lateral=lateral)
    pts = vm.points(p.min_voxel_hits)
    t0 = time.time()
    segs = extract_segments(pts, p)
    slots = find_slots(segs, p, ref_xy=(pose[0], pose[1]))
    dt = time.time() - t0
    return p, pts, segs, slots, gt, dt


if __name__ == "__main__":
    for kind, rmax in (("isolated", 12.0), ("comb", 12.0), ("isolated", 7.5), ("comb", 7.5)):
        p, pts, segs, slots, gt, dt = run(kind, 17.0 if kind == "isolated" else 20.0, rmax=rmax)
        print(f"[{kind}, view {rmax:.0f} m] pts={len(pts)} segs={len(segs)} slots_found={len(slots)} (gt={len(gt)}) t={dt*1000:.0f}ms")
        for g, r in zip(gt, match(slots, gt)):
            if r is None:
                print("   GT@%s: NOT FOUND" % (g['center'],)); continue
            print(f"   GT@{g['center']}: pos_err={r[0]:.3f}m yaw_err={np.degrees(r[1]):.1f}deg width_err={r[2]:.3f} depth_err={r[3]:+.2f}")
        print('   found:', [(tuple(np.round(d.center,2)), round(d.width,2), round(d.depth,2), d.back_seen, round(d.score,2)) for d in slots])
