"""Synthetic parking-lot generator + camera-like visibility model (no ROS)."""
import numpy as np


def seg_points(p0, p1, step=0.007, noise=0.008, rng=None):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    t = np.arange(0, L, step)[:, None] / max(L, 1e-9)
    P = p0 + t * (p1 - p0)
    nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(L, 1e-9)
    thick = (rng.uniform(-0.04, 0.04, len(P)))[:, None] * nrm      # paint width
    return P + thick + rng.normal(0, noise, P.shape)


def world(kind="isolated", rng=None):
    """Return (list of segments, ground-truth slots) . Road along +x, slots on the right (y<0).
    GT slot: entrance centre, yaw INTO slot (-y => -pi/2), width, depth."""
    segs, gt = [], []
    segs.append(((0, 1.5), (30, 1.5)))                      # road edge (left, solid)
    segs.append(((0, -1.8), (9.5, -1.8)))                   # right edge before the lot
    if kind == "isolated":
        xs = [10.0, 14.5, 19.0]
        for x in xs:
            w, d = 3.4, 5.0
            segs += [((x, -1.8), (x, -1.8 - d)), ((x + w, -1.8), (x + w, -1.8 - d)),
                     ((x, -1.8 - d), (x + w, -1.8 - d))]
            gt.append(dict(center=(x + w / 2, -1.8), yaw=-np.pi / 2, width=w, depth=d))
    elif kind == "comb":                                    # shared walls, per-slot back segment
        xs = [10.0, 13.4, 16.8, 20.2]
        for x in xs:
            segs.append(((x, -1.8), (x, -6.8)))
        segs.append(((23.6, -1.8), (23.6, -6.8)))
        for x in xs:
            segs.append(((x, -6.8), (x + 3.4, -6.8)))
            gt.append(dict(center=(x + 1.7, -1.8), yaw=-np.pi / 2, width=3.4, depth=5.0))
    return segs, gt


def visible(P, pose, rmin=1.2, rmax=7.5, fov_deg=55):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    d = P - np.array([x, y])
    fx, fy = d @ [c, s], d @ [-s, c]
    ang = np.degrees(np.abs(np.arctan2(fy, fx)))
    r = np.hypot(fx, fy)
    return (fx > rmin) & (r < rmax) & (ang < fov_deg)


def drive_and_map(kind="isolated", x_end=14.0, step=0.3, seed=1, voxel_map=None, rmax=7.5, lateral=0.0):
    from parking.slot_core import VoxelMap
    rng = np.random.default_rng(seed)
    segs, gt = world(kind, rng)
    P = np.vstack([seg_points(a, b, rng=rng) for a, b in segs])
    vm = voxel_map or VoxelMap(0.05)
    x = 0.0
    while x <= x_end:
        pose = (x, lateral, 0.0)
        vm.add(P[visible(P, pose, rmax=rmax)])
        x += step
    return vm, gt, (x_end, lateral, 0.0), segs
