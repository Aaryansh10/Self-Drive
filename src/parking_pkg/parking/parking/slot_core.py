"""Parking-slot detection core (numpy only, no ROS imports).

Pipeline:  accumulated white-line points (odom frame)
        -> sequential RANSAC line segments
        -> U-shape (back line + two side lines) slot hypotheses
        -> SlotTracker (association + smoothing + confidence).

Slot frame convention (used everywhere):
  origin  = centre of the OPEN side (entrance)
  +x      = unit vector pointing INTO the slot (towards the back line)
  +y      = x rotated +90 deg (left when looking into the slot)
"""
from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------- params
@dataclass
class Params:
    voxel: float = 0.05            # map / downsample resolution [m]
    min_voxel_hits: int = 2        # a voxel must be seen this often to be used
    ransac_iters: int = 300
    ransac_tol: float = 0.05       # inlier distance to a line [m]
    min_seg_len: float = 0.8       # shortest line segment kept [m]
    gap_max: float = 0.45          # split a line at gaps larger than this [m]
    max_lines: int = 40
    corner_tol: float = 0.40       # back-end <-> side-end distance [m]
    parallel_tol_deg: float = 10.0
    perp_tol_deg: float = 12.0
    width_range: tuple = (1.8, 6.0)    # slot width  (back-line length) [m]
    depth_min: float = 1.5             # min visible side length [m]
    width_mismatch: float = 0.5        # |back length - side separation| [m]
    side_len_max: float = 9.0          # longer lines are road/lane edges, not slot sides
    partial_reject_ref_inside: bool = True   # robot between the two sides => it is a lane (False for pull-out)
    seed: int = 0


# ---------------------------------------------------------------- helpers
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def line_angle_diff(a, b):
    """Smallest angle between two undirected directions, in [0, pi/2]."""
    d = abs(wrap(a - b)) % np.pi
    return min(d, np.pi - d)


@dataclass
class Segment:
    p0: np.ndarray       # endpoint (x, y)
    p1: np.ndarray
    n_pts: int

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def angle(self):     # undirected direction angle
        d = self.p1 - self.p0
        return float(np.arctan2(d[1], d[0]))

    @property
    def unit(self):
        return (self.p1 - self.p0) / max(self.length, 1e-9)


@dataclass
class Slot:
    center: np.ndarray        # entrance centre (x, y)
    yaw: float                # direction INTO the slot
    width: float
    depth: float              # visible depth (lower bound)
    score: float              # 0..1 geometric quality of this single detection
    back_seen: bool = True    # False => depth is only a lower bound (back line not visible)
    corners: np.ndarray = field(default=None)   # 4x2: ent_L, ent_R, back_R, back_L (y-left order)

    def polygon(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        hw = self.width / 2
        local = np.array([[0, hw], [0, -hw], [self.depth, -hw], [self.depth, hw]])
        return local @ R.T + self.center


# ------------------------------------------------------ voxel point map
class VoxelMap:
    """Static line map in the odom frame; accumulates detections over time."""

    def __init__(self, voxel=0.05):
        self.v = voxel
        self.hits = {}   # (ix, iy) -> count

    def add(self, xy):
        if len(xy) == 0:
            return
        idx = np.floor(xy / self.v).astype(np.int64)
        keys, cnt = np.unique(idx, axis=0, return_counts=True)
        h = self.hits
        for k, c in zip(map(tuple, keys), cnt):
            h[k] = min(h.get(k, 0) + int(min(c, 3)), 50)

    def prune(self, centre, radius):
        c = np.floor(np.asarray(centre) / self.v).astype(np.int64)
        r = int(radius / self.v)
        self.hits = {k: n for k, n in self.hits.items()
                     if abs(k[0] - c[0]) <= r and abs(k[1] - c[1]) <= r}

    def points(self, min_hits=2):
        if not self.hits:
            return np.zeros((0, 2))
        k = np.array([k for k, n in self.hits.items() if n >= min_hits], dtype=np.float64)
        return (k + 0.5) * self.v if len(k) else np.zeros((0, 2))


# ------------------------------------------------------ line extraction
def extract_segments(pts, p: Params):
    rng = np.random.default_rng(p.seed)
    segs = []
    remaining = pts.copy()
    min_inl = max(int(p.min_seg_len / p.voxel * 0.6), 8)
    for _ in range(p.max_lines):
        n = len(remaining)
        if n < min_inl:
            break
        best_cnt, best = 0, None
        for _ in range(p.ransac_iters):
            i = rng.integers(n)
            near = np.where(np.linalg.norm(remaining - remaining[i], axis=1) < 2.0)[0]
            if len(near) < 2:
                continue
            j = near[rng.integers(len(near))]
            d = remaining[j] - remaining[i]
            L = np.linalg.norm(d)
            if L < 0.3:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            dist = np.abs((remaining - remaining[i]) @ nrm)
            cnt = int((dist < p.ransac_tol).sum())
            if cnt > best_cnt:
                best_cnt, best = cnt, (remaining[i], d / L)
        if best is None or best_cnt < min_inl:
            break
        o, u = best
        nrm = np.array([-u[1], u[0]])
        inl = np.abs((remaining - o) @ nrm) < p.ransac_tol * 1.5
        P = remaining[inl]
        c = P.mean(0)                                  # refit by PCA
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        u = vt[0]
        nrm = np.array([-u[1], u[0]])
        inl = np.abs((remaining - c) @ nrm) < p.ransac_tol
        P = remaining[inl]
        t = (P - c) @ u
        order = np.argsort(t)
        t = t[order]
        cuts = np.where(np.diff(t) > p.gap_max)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(t) - 1]
        kept = False
        for s, e in zip(starts, ends):
            if t[e] - t[s] >= p.min_seg_len and (e - s + 1) >= min_inl:
                segs.append(Segment(c + u * t[s], c + u * t[e], e - s + 1))
                kept = True
        remaining = remaining[~inl]
        if not kept and best_cnt < min_inl * 2:
            continue
    return segs


# ----------------------------------------------------------- U detection
def _end_near(seg, pt, tol):
    """Return (end_point, far_point) of seg whose end is within tol of pt, else None."""
    d0 = np.linalg.norm(seg.p0 - pt)
    d1 = np.linalg.norm(seg.p1 - pt)
    if min(d0, d1) > tol:
        return None
    return (seg.p0, seg.p1) if d0 <= d1 else (seg.p1, seg.p0)


def split_at_junctions(segs, p: Params):
    """Split a segment where a perpendicular segment touches its interior (shared back lines)."""
    out = []
    for bi, b in enumerate(segs):
        u = b.unit
        nrm = np.array([-u[1], u[0]])
        L = b.length
        cuts = []
        for si, s in enumerate(segs):
            if si == bi or line_angle_diff(s.angle, b.angle) < np.deg2rad(60):
                continue
            for q in (s.p0, s.p1):
                if abs((q - b.p0) @ nrm) < p.corner_tol * 0.6:
                    t = (q - b.p0) @ u
                    if p.corner_tol < t < L - p.corner_tol:
                        cuts.append(t)
        if not cuts:
            out.append(b)
            continue
        cuts = sorted(cuts)
        merged = [cuts[0]]
        for t in cuts[1:]:
            if t - merged[-1] > 0.3:
                merged.append(t)
        edges = [0.0] + merged + [L]
        for a, c in zip(edges[:-1], edges[1:]):
            if c - a >= p.min_seg_len:
                out.append(Segment(b.p0 + u * a, b.p0 + u * c, max(int(b.n_pts * (c - a) / L), 1)))
    return out


def find_slots(segs, p: Params, ref_xy=None):
    """U-shaped slots (back + 2 sides). If ref_xy (e.g. robot position) is given, also returns
    partial slots made of two parallel sides when the back line is not (yet) visible."""
    segs = split_at_junctions(segs, p)
    out, used = [], set()
    ptol = np.deg2rad(p.parallel_tol_deg)
    qtol = np.deg2rad(p.perp_tol_deg)
    for bi, b in enumerate(segs):
        W = b.length
        if not (p.width_range[0] <= W <= p.width_range[1]):
            continue
        ends = (b.p0, b.p1)
        perp = np.array([-b.unit[1], b.unit[0]])
        for sgn in (+1, -1):               # +perp or -perp = direction to OPEN side
            open_dir = sgn * perp
            sides = []
            for e in ends:
                best = None
                for si, s in enumerate(segs):
                    if si == bi or line_angle_diff(s.angle, b.angle) < np.pi / 2 - qtol:
                        continue
                    r = _end_near(s, e, p.corner_tol)
                    if r is None:
                        continue
                    near, far = r
                    if (far - near) @ open_dir <= 0.5 * s.length:   # must extend to open side
                        continue
                    if best is None or s.length > best[1].length:
                        best = (near, s, far, si)
                if best is None:
                    break
                sides.append(best)
            if len(sides) != 2:
                continue
            (n0, s0, f0, i0), (n1, s1, f1, i1) = sides
            if line_angle_diff(s0.angle, s1.angle) > ptol:
                continue
            depth = min(s0.length, s1.length)
            if depth < p.depth_min:
                continue
            u_in = -open_dir                       # unit vector INTO the slot
            sep = abs((n1 - n0) @ b.unit)
            if abs(sep - W) > p.width_mismatch:
                continue
            ax = s0.unit * np.sign(s0.unit @ u_in) + s1.unit * np.sign(s1.unit @ u_in)
            ax /= np.linalg.norm(ax)
            yaw = float(np.arctan2(ax[1], ax[0]))
            back_mid = 0.5 * (n0 + n1)
            center = back_mid - ax * depth
            score = max(0.0, 1 - abs(sep - W) / p.width_mismatch)
            score *= max(0.3, 1 - line_angle_diff(s0.angle, s1.angle) / ptol * 0.5)
            score *= min(1.0, 0.5 + depth / 4.0)
            slot = Slot(center=center, yaw=yaw, width=0.5 * (sep + W), depth=depth,
                        score=score, back_seen=True)
            slot.corners = slot.polygon()
            out.append(slot)
            used.update((i0, i1))
    if ref_xy is not None:
        out += _partial_slots(segs, used, p, np.asarray(ref_xy, float))
    return _dedupe(out)


def _partial_slots(segs, used, p, ref):
    """Two parallel sides with a plausible gap, no other side in between, back unseen."""
    res = []
    ptol = np.deg2rad(p.parallel_tol_deg)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if i in used or j in used:
                continue
            a, b = segs[i], segs[j]
            if line_angle_diff(a.angle, b.angle) > ptol:
                continue
            if a.length > p.side_len_max or b.length > p.side_len_max:
                continue
            u = a.unit
            nrm = np.array([-u[1], u[0]])
            sep = abs((b.p0 + b.p1) / 2 @ nrm - (a.p0 + a.p1) / 2 @ nrm)
            if not (p.width_range[0] <= sep <= p.width_range[1]):
                continue
            ta = sorted(((a.p0 - a.p0) @ u, (a.p1 - a.p0) @ u))
            tb = sorted(((b.p0 - a.p0) @ u, (b.p1 - a.p0) @ u))
            lo, hi = max(ta[0], tb[0]), min(ta[1], tb[1])
            if hi - lo < p.depth_min:
                continue
            if p.partial_reject_ref_inside:       # robot driving between the two lines => a lane
                ra = (ref - a.p0) @ nrm
                rb = (b.p0 - a.p0) @ nrm
                rt = (ref - a.p0) @ u
                if min(0, rb) < ra < max(0, rb) and lo - 0.5 < rt < hi + 0.5:
                    continue
            blocked = False                       # another side between them => not one slot
            for k, c in enumerate(segs):
                if k in (i, j) or line_angle_diff(c.angle, a.angle) > ptol:
                    continue
                off = ((c.p0 + c.p1) / 2 - (a.p0 + a.p1) / 2) @ nrm
                sa = (b.p0 + b.p1) / 2 @ nrm - (a.p0 + a.p1) / 2 @ nrm
                if min(0, sa) + 0.3 < off < max(0, sa) - 0.3:
                    blocked = True
                    break
            if blocked:
                continue
            # open side = axis end closest to the reference point
            tref = (ref - a.p0) @ u
            open_is_lo = abs(tref - lo) < abs(tref - hi)
            u_in = u if open_is_lo else -u
            t_open = lo if open_is_lo else hi
            ent = a.p0 + t_open * u + nrm * (((b.p0 - a.p0) @ nrm) / 2)
            yaw = float(np.arctan2(u_in[1], u_in[0]))
            slot = Slot(center=ent, yaw=yaw, width=sep, depth=hi - lo,
                        score=0.45 * min(1.0, 0.5 + (hi - lo) / 4.0), back_seen=False)
            slot.corners = slot.polygon()
            res.append(slot)
    return res


def _dedupe(slots):
    slots = sorted(slots, key=lambda s: -s.score)
    kept = []
    for s in slots:
        mid = s.center + np.array([np.cos(s.yaw), np.sin(s.yaw)]) * s.depth / 2
        dup = False
        for k in kept:
            mk = k.center + np.array([np.cos(k.yaw), np.sin(k.yaw)]) * k.depth / 2
            if np.linalg.norm(mid - mk) < 0.5 and line_angle_diff(s.yaw, k.yaw) < 0.3:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


# --------------------------------------------------------------- tracking
@dataclass
class Track:
    id: int
    slot: Slot
    hits: int = 1
    misses: int = 0
    locked: bool = False

    @property
    def confidence(self):
        return float(min(1.0, self.hits / 8.0) * self.slot.score * (0.5 ** (self.misses / 6.0)))


class SlotTracker:
    def __init__(self, alpha=0.3, gate=0.8, max_misses=40):
        self.tracks, self.next_id = [], 1
        self.alpha, self.gate, self.max_misses = alpha, gate, max_misses

    def update(self, dets):
        used = set()
        for t in self.tracks:
            best, bd = None, self.gate
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = np.linalg.norm(d.center - t.slot.center)
                if dist < bd and line_angle_diff(d.yaw, t.slot.yaw) < 0.3:
                    best, bd = i, dist
            if best is None:
                t.misses += 1
                continue
            used.add(best)
            t.hits += 1
            t.misses = 0
            if not t.locked:
                a, d, s = self.alpha, dets[best], t.slot
                s.center = (1 - a) * s.center + a * d.center
                s.yaw = float(s.yaw + a * wrap(d.yaw - s.yaw))
                s.width = (1 - a) * s.width + a * d.width
                if d.back_seen and not s.back_seen:
                    s.center, s.depth, s.back_seen = d.center.copy(), d.depth, True
                elif d.back_seen == s.back_seen:
                    s.depth = (1 - a) * s.depth + a * d.depth
                else:
                    s.depth = max(s.depth, d.depth)
                s.score = (1 - a) * s.score + a * d.score
                s.corners = s.polygon()
        for i, d in enumerate(dets):
            if i not in used:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1
        self.tracks = [t for t in self.tracks
                       if t.misses <= self.max_misses and not (t.misses > 8 and t.confidence < 0.05 and not t.locked)]
        return self.tracks

    def lock(self, track_id):
        for t in self.tracks:
            t.locked = (t.id == track_id)


# ------------------------------------------------------- ROS-side helpers
def parse_xy(data: bytes, point_step: int, n_points: int, offsets=(0, 4)):
    """x, y (float32) from a PointCloud2 buffer without sensor_msgs_py."""
    raw = np.frombuffer(data, dtype=np.uint8, count=n_points * point_step).reshape(n_points, point_step)
    x = raw[:, offsets[0]:offsets[0] + 4].copy().view(np.float32).ravel()
    y = raw[:, offsets[1]:offsets[1] + 4].copy().view(np.float32).ravel()
    xy = np.stack([x, y], 1).astype(np.float64)
    return xy[np.isfinite(xy).all(1)]


def to_odom(xy, tx, ty, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return xy @ np.array([[c, s], [-s, c]]) + np.array([tx, ty])
