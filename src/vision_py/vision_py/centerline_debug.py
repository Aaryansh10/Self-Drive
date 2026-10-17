"""
centerline_debug.py

DebugCenterlineGenerator
    A drop-in subclass of CenterlineGenerator that behaves identically but records every
    intermediate result of a frame into `self.dbg`.

    The recorded methods are line-for-line copies of the originals with recording added
    (generate_centerlines, label_lines, line_centerline, get_yellow_candidates, yellow_gate,
    fuse_side, bezier_lane_change). Everything else (memory update, resampling, Savitzky-Golay,
    masked_smooth, lock_near_field, ...) is inherited unchanged. If you change the logic of one
    of those copied methods in centerline.py, mirror the change here.

build_markers()
    Turns `dbg` into one MarkerArray per pipeline stage (see STAGE_TOPICS).
"""
import numpy as np
import torch

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from vision_py.centerline import CenterlineGenerator


def _np(t):
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu().numpy().copy()
    return np.array(t)


# =====================================================================================
# Instrumented generator
# =====================================================================================

class DebugCenterlineGenerator(CenterlineGenerator):

    def __init__(self, *args, **kwargs):
        self.dbg = {}
        self._side = None   # which side bezier_lane_change is currently running for
        self._src = ''      # name of the line currently being processed (W0, W1, Y0, ...)
        super().__init__(*args, **kwargs)
        self.dbg = self._new_dbg()

    # --- debug bookkeeping ---

    @staticmethod
    def _new_dbg():
        return dict(
            white_xy=None, yellow_pts=None, yellow_keep=None, yellow_gate=None,
            yellow_used=False, yellow_note='',
            mem_used=None, mem_after=None, votes=None, unknown_frac=0.0,
            lines=[], smoothing=[], candidates=[], rejects=[],
            counts={}, fuse={}, track={}, pre_bezier={}, bezier={}, final={},
            failed=False, bad_streak=0, memory_reset=False, timing={}, n_pts=(0, 0, 0),
        )

    def _mem_snapshot(self):
        return dict(
            left=_np(self.prev_left_y), right=_np(self.prev_right_y),
            width=_np(self.road_width), stale=_np(self.width_stale),
        )

    def refresh_derived(self):
        """Recompute attributes derived from constructor args after live-editing a parameter."""
        self.width_stale_decay_frames = max(1, int(self.width_stale_decay_frames))
        self.min_offset = self.nominal_road_width * self.min_offset_fraction
        self.sg_polyorder = max(1, int(self.sg_polyorder))
        self.sg_min_window = self.sg_polyorder + 1
        if self.sg_min_window % 2 == 0:
            self.sg_min_window += 1
        self.sg_coeffs = {}

    # --- entry point (copy of the original + recording) ---

    @torch.no_grad()
    def generate_centerlines(self, pc_whitelane, pc_yellow=None):
        d = self.dbg = self._new_dbg()
        near = {'left': [], 'right': []}
        far = {'left': [], 'right': []}
        yellow = {'left': [], 'right': []}

        if pc_whitelane is not None and pc_whitelane.shape[0] > 0:
            pc_whitelane = torch.as_tensor(pc_whitelane).to(self.device, self.dtype)
            x, y = pc_whitelane[:, 0], pc_whitelane[:, 1]
            d['white_xy'] = (_np(x), _np(y))
            line_id, pts_per_line = self.split_lines(x, y)

            pt_dist_bin_width = self.get_dist_bin_width(x)
            d['mem_used'] = self._mem_snapshot()          # memory exactly as label_lines sees it
            line_is_left = self.label_lines(y, line_id, pt_dist_bin_width, pts_per_line)
            pt_is_left = line_is_left[line_id]
            self.update_memory(y, pt_dist_bin_width, pt_is_left)
            d['mem_after'] = self._mem_snapshot()

            start = 0
            for li, (n, is_left) in enumerate(zip(pts_per_line.tolist(), line_is_left.tolist())):
                self._src = f'W{li}'
                seg_x, seg_y = x[start:start + n], y[start:start + n]
                d['lines'].append(dict(
                    id=li, x=_np(seg_x), y=_np(seg_y), is_left=bool(is_left), n=n,
                    vote=float(d['votes'][li]) if d['votes'] is not None else float('nan'),
                ))
                candidates = self.line_centerline(
                    seg_x, seg_y, self.WHITE_LEFT if is_left else self.WHITE_RIGHT
                )
                start += n
                for lane, kind, cx, cy, rd, frac in (candidates or []):
                    (near if kind == 'near' else far)[lane].append((cx, cy, rd, frac))
        else:
            d['rejects'].append(('white', 'no white-lane points this frame'))

        if pc_yellow is not None and pc_yellow.shape[0] > 0:
            d['yellow_pts'] = (_np(pc_yellow[:, 0]), _np(pc_yellow[:, 1]))

        if (not near['left'] or not near['right']) and pc_yellow is not None and pc_yellow.shape[0] > 0:
            d['yellow_used'] = True
            pc_yellow = torch.as_tensor(pc_yellow).to(self.device, self.dtype)
            self.get_yellow_candidates(pc_yellow, yellow)
        else:
            d['yellow_note'] = ('skipped: both near candidates exist'
                                if (near['left'] and near['right']) else 'skipped: no yellow points')

        sel_left = self.select(near['left'], far['left'], yellow['left'])
        sel_right = self.select(near['right'], far['right'], yellow['right'])
        used = {id(c[0]) for c in sel_left} | {id(c[0]) for c in sel_right}
        for c in d['candidates']:
            c['used'] = c['uid'] in used
        d['counts'] = dict(
            near=(len(near['left']), len(near['right'])),
            far=(len(far['left']), len(far['right'])),
            yellow=(len(yellow['left']), len(yellow['right'])),
        )

        left_xy = self.fuse_side('left', sel_left)
        right_xy = self.fuse_side('right', sel_right)
        d['track'] = {s: (_np(self.track[s][0]), _np(self.track[s][1])) for s in ('left', 'right')}

        if left_xy is None and right_xy is None:
            d['failed'] = True
            res = self.failed_frame()
            d['bad_streak'] = self.bad_frame_streak
            d['memory_reset'] = self.bad_frame_streak == 0
            return res

        self.bad_frame_streak = 0
        d['pre_bezier'] = {'left': _np(left_xy), 'right': _np(right_xy)}
        if self.lane_change_bezier:
            self._side = 'left'
            left_xy = self.bezier_lane_change(left_xy)
            self._side = 'right'
            right_xy = self.bezier_lane_change(right_xy)
            self._side = None
        return left_xy, right_xy

    # --- lane labelling (copy + vote recording) ---

    def label_lines(self, y, line_id, pt_dist_bin_width, pts_per_line):
        prev_left_y = self.prev_left_y[pt_dist_bin_width]
        prev_right_y = self.prev_right_y[pt_dist_bin_width]
        road_width = self.road_width[pt_dist_bin_width]

        unknown_left = torch.isnan(prev_left_y)
        unknown_right = torch.isnan(prev_right_y)

        prev_left_y = torch.where(
            unknown_left & ~unknown_right, prev_right_y + road_width, prev_left_y
        )
        prev_right_y = torch.where(
            unknown_right & ~unknown_left, prev_left_y - road_width, prev_right_y
        )

        unknown_both = unknown_left & unknown_right
        closer_to_left = (y - prev_left_y).abs() <= (y - prev_right_y).abs()
        pts_vote_left = torch.where(unknown_both, y > 0.0, closer_to_left)

        left_votes_per_line = torch.bincount(
            line_id, weights=pts_vote_left.to(self.dtype), minlength=pts_per_line.numel()
        )
        line_is_left = (left_votes_per_line * 2) >= pts_per_line

        self.dbg['votes'] = _np(left_votes_per_line / pts_per_line.to(self.dtype).clamp(min=1))
        self.dbg['unknown_frac'] = float(unknown_both.to(self.dtype).mean())
        return line_is_left

    # --- per-line candidates (copy + recording) ---

    def line_centerline(self, x, y, offsets):
        d = self.dbg
        src = self._src

        if x.numel() < 3:
            d['rejects'].append((src, f'only {x.numel()} raw points (<3)'))
            return None

        if torch.hypot(x[0], y[0]) > torch.hypot(x[-1], y[-1]):
            x = x.flip(0)
            y = y.flip(0)

        resampled = self.resample_line(x.contiguous(), y.contiguous())
        if resampled is None:
            d['rejects'].append((src, f'line shorter than {3 * self.resample_step:.2f} m'))
            return None
        x, y = resampled
        n = x.numel()

        wp = self.odd_window(self.position_window_m, n)
        wt = self.odd_window(self.tangent_window_m, n)

        if wt >= self.sg_min_window:
            mode = 'SG'
            wp = max(wp, self.sg_min_window)
            x_smoothed = self.sg_apply(x, wp, 0)
            y_smoothed = self.sg_apply(y, wp, 0)
            tangent_x = self.sg_apply(x, wt, 1)
            tangent_y = self.sg_apply(y, wt, 1)
        else:
            mode = 'box'
            w = min(7, n)
            if w % 2 == 0:
                w -= 1
            x_smoothed = self.box_smooth(x, w)
            y_smoothed = self.box_smooth(y, w)
            tangent_x = torch.gradient(x_smoothed)[0]
            tangent_y = torch.gradient(y_smoothed)[0]

        norm = torch.hypot(tangent_x, tangent_y) + 1e-5
        tangent_x = tangent_x / norm
        tangent_y = tangent_y / norm

        d['smoothing'].append(dict(
            src=src, mode=mode, n=n, wp=wp, wt=wt,
            rs_x=_np(x), rs_y=_np(y), xs=_np(x_smoothed), ys=_np(y_smoothed),
            tx=_np(tangent_x), ty=_np(tangent_y),
        ))

        bins = self.get_dist_bin_width(x_smoothed)
        road_w = self.road_width[bins]

        out = []
        for lane, fraction, toward_right, kind in offsets:
            normal_x, normal_y = self.get_inward_normal(tangent_x, tangent_y, toward_right)
            offset = torch.maximum(
                road_w * fraction,
                torch.full_like(road_w, self.min_offset * fraction)
            )
            centerline_x = x_smoothed + normal_x * offset
            centerline_y = y_smoothed + normal_y * offset
            pre_x, pre_y = _np(centerline_x), _np(centerline_y)

            radial_dist = torch.hypot(centerline_x, centerline_y)
            order = torch.argsort(radial_dist)
            n_reordered = int((order != torch.arange(order.numel(), device=order.device)).sum())
            centerline_x, centerline_y = centerline_x[order], centerline_y[order]
            radial_dist = radial_dist[order]
            radial_dist = radial_dist + torch.arange(
                radial_dist.numel(), device=self.device, dtype=self.dtype
            ) * 1e-4
            out.append((lane, kind, centerline_x, centerline_y, radial_dist, fraction))

            d['candidates'].append(dict(
                uid=id(centerline_x), src=src, lane=lane, kind=kind, frac=fraction, used=True,
                x=_np(centerline_x), y=_np(centerline_y), pre_x=pre_x, pre_y=pre_y,
                base_x=_np(x_smoothed), base_y=_np(y_smoothed), n_reordered=n_reordered,
                r_lo=float(radial_dist[0]), r_hi=float(radial_dist[-1]),
            ))

        return out

    # --- yellow (copy + recording) ---

    def get_yellow_candidates(self, pc_yellow, yellow):
        x, y = pc_yellow[:, 0], pc_yellow[:, 1]
        keep = self.yellow_gate(x, y)
        self.dbg['yellow_keep'] = _np(keep)
        x, y = x[keep], y[keep]

        if x.numel() < 3:
            self.dbg['rejects'].append(('yellow', f'only {x.numel()} points pass the gate (<3)'))
            return

        line_id, pts_per_line = self.split_lines(x, y)

        start = 0
        for li, n in enumerate(pts_per_line.tolist()):
            self._src = f'Y{li}'
            candidates = self.line_centerline(
                x[start:start + n], y[start:start + n],
                self.YELLOW
            )
            start += n
            for lane, kind, centerline_x, centerline_y, radial_dist, frac in (candidates or []):
                yellow[lane].append((centerline_x, centerline_y, radial_dist, frac))

    def yellow_gate(self, x, y):
        bins = self.get_dist_bin_width(x)
        left = self.prev_left_y[bins]
        right = self.prev_right_y[bins]
        width = self.road_width[bins]

        mid = torch.where(
            ~torch.isnan(left) & ~torch.isnan(right),
            0.5 * (left + right),
            torch.where(
                ~torch.isnan(left),
                left - 0.5 * width,
                torch.where(
                    ~torch.isnan(right),
                    right + 0.5 * width,
                    torch.full_like(width, float('nan'))
                )
            )
        )

        known = ~torch.isnan(mid)
        self.dbg['yellow_gate'] = dict(
            x=_np(x), mid=_np(mid), width=_np(width), frac=self.yellow_gate_fraction
        )
        return ~known | ((y - mid).abs() <= self.yellow_gate_fraction * width)

    # --- fusion (copy + recording) ---

    def fuse_side(self, side, candidates):
        rec = dict(side=side, ok=False, reason='', n_cand=len(candidates), contribs=[], n_valid=0)
        self.dbg['fuse'][side] = rec

        if not candidates:
            rec['reason'] = 'no candidates'
            return None

        sum_x = torch.zeros_like(self.d_grid)
        sum_y = torch.zeros_like(self.d_grid)
        sum_w = torch.zeros_like(self.d_grid)

        for candidate_x, candidate_y, radial_dist, fraction in candidates:
            lo, hi = radial_dist[0], radial_dist[-1]
            valid = (self.d_grid >= lo) & (self.d_grid <= hi)
            if not bool(valid.any()):
                rec['contribs'].append(dict(frac=fraction, lo=float(lo), hi=float(hi), used=False))
                continue
            dist_grid = self.d_grid[valid]

            w = (self.ref_fraction / max(fraction, 1e-3)) ** 2

            edge = torch.minimum(dist_grid - lo, hi - dist_grid)
            ramp = (edge / self.taper_len).clamp(0.0, 1.0)
            weight = w * (0.02 + 0.98 * ramp)

            sum_x[valid] += weight * self.interpolate(dist_grid, radial_dist, candidate_x)
            sum_y[valid] += weight * self.interpolate(dist_grid, radial_dist, candidate_y)
            sum_w[valid] += weight
            rec['contribs'].append(dict(frac=fraction, lo=float(lo), hi=float(hi), used=True, w=w))

        valid_grid = sum_w > 0
        rec['n_valid'] = int(valid_grid.sum())
        if rec['n_valid'] < 3:
            rec['reason'] = f"only {rec['n_valid']} grid points covered (<3)"
            return None

        x_mean = sum_x / sum_w.clamp(min=1e-9)
        y_mean = sum_y / sum_w.clamp(min=1e-9)
        rec.update(
            grid=_np(self.d_grid[valid_grid]), sum_w=_np(sum_w[valid_grid]),
            raw_x=_np(x_mean[valid_grid]), raw_y=_np(y_mean[valid_grid]),
        )

        x_mean = self.masked_smooth(x_mean, valid_grid, self.final_smooth_window)[valid_grid]
        y_mean = self.masked_smooth(y_mean, valid_grid, self.final_smooth_window)[valid_grid]
        rec.update(sm_x=_np(x_mean), sm_y=_np(y_mean))

        has_close = any(c[3] <= self.ref_fraction + 1e-6 for c in candidates)
        start_limit = self.near_field_range if has_close else self.far_near_field_range
        rec.update(start_x=float(x_mean.min()), start_limit=start_limit, has_close=has_close)
        if x_mean.numel() < self.min_near_pts:
            rec['reason'] = f'{x_mean.numel()} pts < min_near_pts={self.min_near_pts}'
            return None
        if float(x_mean.min()) > start_limit:
            which = 'near_field_range' if has_close else 'far_near_field_range'
            rec['reason'] = f"path starts at x={float(x_mean.min()):.2f} m > {which}={start_limit:.1f} m"
            return None

        track = self.track[side]
        prev_x = track[0][valid_grid]
        prev_y = track[1][valid_grid]
        has_prev = ~torch.isnan(prev_x)
        rec.update(prev_x=_np(prev_x), prev_y=_np(prev_y))

        alpha = self.centerline_ema_alpha
        smoothed_x = torch.where(has_prev, alpha * x_mean + (1.0 - alpha) * prev_x, x_mean)
        smoothed_y = torch.where(has_prev, alpha * y_mean + (1.0 - alpha) * prev_y, y_mean)
        rec.update(ema_x=_np(smoothed_x), ema_y=_np(smoothed_y))

        smoothed_x, smoothed_y = self.lock_near_field(smoothed_x, smoothed_y)
        rec.update(lock_x=_np(smoothed_x), lock_y=_np(smoothed_y))
        track[0][valid_grid] = smoothed_x
        track[1][valid_grid] = smoothed_y
        rec['ok'] = True
        return torch.stack([smoothed_x, smoothed_y], dim=1)

    # --- bezier (copy + recording) ---

    def bezier_lane_change(self, path_xy):
        if path_xy is None or path_xy.shape[0] < 3:
            return path_xy

        n = path_xy.shape[0]
        x, y = path_xy[:, 0], path_xy[:, 1]
        r = torch.hypot(x, y)

        lateral = float(y[:min(n, 5)].abs().mean())
        chord = max(self.lane_change_min_len, self.lane_change_factor * lateral)
        chord = min(chord, 0.8 * float(r.max()))

        ahead = torch.nonzero(r >= chord)
        j = int(ahead[0]) if ahead.numel() > 0 else n - 1

        k = 5
        lo, hi = max(j - k, 0), min(j + k, n - 1)
        tangent = path_xy[hi] - path_xy[lo]
        tangent = tangent / tangent.norm().clamp(min=1e-6)

        p0 = torch.zeros(2, device=path_xy.device, dtype=path_xy.dtype)
        p3 = path_xy[j]
        handle = self.bezier_handle * float(p3.norm())
        heading = torch.tensor([1.0, 0.0], device=path_xy.device, dtype=path_xy.dtype)
        p1 = p0 + handle * heading
        p2 = p3 - handle * tangent

        u = torch.linspace(0.0, 1.0, 200, device=path_xy.device, dtype=path_xy.dtype).unsqueeze(1)
        bez = ((1 - u) ** 3) * p0 + 3 * ((1 - u) ** 2) * u * p1 + 3 * (1 - u) * (u ** 2) * p2 + (u ** 3) * p3

        seg = torch.hypot(torch.diff(bez[:, 0]), torch.diff(bez[:, 1]))
        arclen = torch.cat((seg.new_zeros(1), torch.cumsum(seg, 0)))
        arclen = arclen + torch.arange(arclen.numel(), device=arclen.device, dtype=arclen.dtype) * 1e-7
        m = max(int(float(arclen[-1]) / self.path_spacing) + 1, 2)
        query = torch.linspace(0.0, float(arclen[-1]), m, device=arclen.device, dtype=arclen.dtype)
        bx = self.interpolate(query, arclen, bez[:, 0])
        by = self.interpolate(query, arclen, bez[:, 1])

        result = torch.cat((torch.stack((bx, by), dim=1)[:-1], path_xy[j:]), dim=0)

        if self._side is not None:
            self.dbg['bezier'][self._side] = dict(
                post=_np(result), ctrl=_np(torch.stack([p0, p1, p2, p3])),
                j=j, chord=chord, lateral=lateral, n=n,
            )
        return result


# =====================================================================================
# RViz marker building
# =====================================================================================

# One MarkerArray topic per stage: <prefix>/<suffix>. Order = order of the pipeline.
STAGE_TOPICS = (
    's0_status',      # text: per-frame summary, per-side failure reason, timing
    's1_lines',       # split + left/right labelled white lines (with vote fraction)
    's2_memory',      # tracked left/right y per bin + road width used for labelling
    's3_yellow',      # yellow points kept/rejected by the gate + gate band
    's4_smoothing',   # resampled line, SG-smoothed line, tangents, offset normals
    's5_candidates',  # every near/far/yellow candidate (and which ones were used)
    's6_fusion',      # fuse_side stages: raw weighted mean -> smooth -> EMA -> lock + track memory
    's7_bezier',      # lane-change bezier: pre/post path and control points
)

LEFT = (0.1, 0.85, 1.0)
RIGHT = (1.0, 0.3, 0.9)
GREEN = (0.2, 1.0, 0.2)
ORANGE = (1.0, 0.55, 0.0)
YELLOW = (1.0, 0.95, 0.0)
RED = (1.0, 0.15, 0.15)
WHITE = (1.0, 1.0, 1.0)
GRAY = (0.6, 0.6, 0.6)
BLUE = (0.3, 0.5, 1.0)
PURPLE = (0.75, 0.4, 1.0)
KIND_COLOR = {'near': GREEN, 'far': ORANGE, 'yellow': YELLOW}


def _rgba(c, a=1.0):
    m = ColorRGBA()
    m.r, m.g, m.b, m.a = float(c[0]), float(c[1]), float(c[2]), float(a)
    return m


def _pt(x, y, z):
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


class _Builder:
    def __init__(self, header):
        self.header = header
        self.arr = MarkerArray()
        self._id = 0
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        self.arr.markers.append(clear)

    def _base(self, ns, mtype):
        m = Marker()
        m.header = self.header
        m.ns = ns
        m.id = self._id
        self._id += 1
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def strip(self, ns, xs, ys, color, width=0.05, z=0.0, alpha=1.0):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if len(xs) < 2:
            return
        m = self._base(ns, Marker.LINE_STRIP)
        m.scale.x = float(width)
        m.color = _rgba(color, alpha)
        m.points = [_pt(a, b, z) for a, b in zip(xs, ys)]
        self.arr.markers.append(m)

    def points(self, ns, xs, ys, color, size=0.08, z=0.0, alpha=1.0, colors=None):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if not ok.any():
            return
        m = self._base(ns, Marker.POINTS)
        m.scale.x = m.scale.y = float(size)
        m.color = _rgba(color, alpha)
        m.points = [_pt(a, b, z) for a, b in zip(xs[ok], ys[ok])]
        if colors is not None:
            m.colors = [_rgba(c, alpha) for c, k in zip(colors, ok) if k]
        self.arr.markers.append(m)

    def spheres(self, ns, xs, ys, color, size=0.2, z=0.0, alpha=1.0):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if not ok.any():
            return
        m = self._base(ns, Marker.SPHERE_LIST)
        m.scale.x = m.scale.y = m.scale.z = float(size)
        m.color = _rgba(color, alpha)
        m.points = [_pt(a, b, z) for a, b in zip(xs[ok], ys[ok])]
        self.arr.markers.append(m)

    def segments(self, ns, x0, y0, x1, y1, color, width=0.02, z=0.0, alpha=1.0):
        x0, y0, x1, y1 = (np.asarray(v, dtype=float) for v in (x0, y0, x1, y1))
        ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
        if not ok.any():
            return
        m = self._base(ns, Marker.LINE_LIST)
        m.scale.x = float(width)
        m.color = _rgba(color, alpha)
        for a, b, c, e in zip(x0[ok], y0[ok], x1[ok], y1[ok]):
            m.points.append(_pt(a, b, z))
            m.points.append(_pt(c, e, z))
        self.arr.markers.append(m)

    def text(self, ns, x, y, s, color=WHITE, size=0.25, z=0.3):
        m = self._base(ns, Marker.TEXT_VIEW_FACING)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        m.scale.z = float(size)
        m.color = _rgba(color)
        m.text = s
        self.arr.markers.append(m)


def _mid(a):
    return len(a) // 2


# --- stage builders ---

def _stage_status(b, d, gen):
    f = d['fuse']
    t = d['timing']
    head = 'FAILED FRAME' if d['failed'] else 'ok'
    lines = [
        f"frame: {head}   bad_streak={d['bad_streak']}" + ('  MEMORY RESET' if d['memory_reset'] else ''),
        f"pts  white={d['n_pts'][0]}  yellow={d['n_pts'][1]}  stop={d['n_pts'][2]}   lines={len(d['lines'])}"
        f" (L={sum(l['is_left'] for l in d['lines'])} R={sum(not l['is_left'] for l in d['lines'])})",
        f"label memory unknown (both sides): {100 * d['unknown_frac']:.0f}% of pts",
        f"cand near L/R={d['counts'].get('near', '-')}  far={d['counts'].get('far', '-')}"
        f"  yellow={d['counts'].get('yellow', '-')}",
        f"yellow: {'evaluated' if d['yellow_used'] else d['yellow_note']}",
    ]
    if t:
        lines.append('time ms  ' + '  '.join(f'{k}={v:.1f}' for k, v in t.items()))
    for src, why in d['rejects'][:6]:
        lines.append(f'  reject {src}: {why}')
    b.text('status', 0.0, 0.0, '\n'.join(lines), RED if d['failed'] else WHITE, 0.22, 2.2)

    for side, ycol, yy in (('left', LEFT, 3.6), ('right', RIGHT, -3.6)):
        rec = f.get(side)
        if rec is None:
            continue
        if rec['ok']:
            s = f"{side.upper()}: ok  ({rec['n_valid']} grid pts, {rec['n_cand']} cand)"
            b.text('side', 1.0, yy, s, GREEN, 0.25, 0.5)
        else:
            s = f"{side.upper()}: DROPPED - {rec['reason']}"
            b.text('side', 1.0, yy, s, RED, 0.25, 0.5)


def _stage_lines(b, d, gen):
    for ln in d['lines']:
        side = 'left' if ln['is_left'] else 'right'
        col = LEFT if ln['is_left'] else RIGHT
        b.strip(f'{side}_lines', ln['x'], ln['y'], col, 0.04, 0.0)
        b.points(f'{side}_lines', ln['x'], ln['y'], col, 0.07, 0.0)
        if ln['n'] > 0:
            k = _mid(ln['x'])
            vote = ln['vote']
            unsure = abs(vote - 0.5) < 0.2
            s = f"W{ln['id']} {side[0].upper()} n={ln['n']} votes_left={vote:.2f}" + ('  (UNSURE)' if unsure else '')
            b.text('line_labels', ln['x'][k], ln['y'][k], s, RED if unsure else col, 0.2, 0.2)


def _stage_memory(b, d, gen):
    mu, ma = d['mem_used'], d['mem_after']
    if mu is None:
        return
    nb = len(mu['left'])
    bx = (np.arange(nb) + 0.5) * gen.bin_width
    # What label_lines used (big, translucent) vs what the memory looks like after this frame (small, solid)
    b.spheres('used_left', bx, mu['left'], LEFT, 0.28, 0.0, 0.35)
    b.spheres('used_right', bx, mu['right'], RIGHT, 0.28, 0.0, 0.35)
    b.spheres('after_left', bx, ma['left'], LEFT, 0.1, 0.02)
    b.spheres('after_right', bx, ma['right'], RIGHT, 0.1, 0.02)

    # Inferred missing side (how label_lines fills in an unknown side from road_width)
    only_l = np.isfinite(mu['left']) & ~np.isfinite(mu['right'])
    only_r = np.isfinite(mu['right']) & ~np.isfinite(mu['left'])
    b.spheres('inferred', bx[only_l], (mu['left'] - mu['width'])[only_l], ORANGE, 0.14, 0.01)
    b.spheres('inferred', bx[only_r], (mu['right'] + mu['width'])[only_r], ORANGE, 0.14, 0.01)

    both = np.isfinite(ma['left']) & np.isfinite(ma['right'])
    b.segments('width', bx[both], ma['left'][both], bx[both], ma['right'][both], GREEN, 0.02, 0.0, 0.6)
    for i in range(nb):
        stale = mu['stale'][i] > 0
        meas = ma['left'][i] - ma['right'][i] if both[i] else float('nan')
        txt = f"{mu['width'][i]:.1f}" + (f"\n({meas:.1f})" if np.isfinite(meas) else '')
        b.text('width_txt', bx[i], 0.0, txt, RED if stale else WHITE, 0.14, 0.05)


def _stage_yellow(b, d, gen):
    pts, keep = d['yellow_pts'], d['yellow_keep']
    if pts is None:
        return
    x, y = pts
    if keep is None:
        b.points('yellow_pts', x, y, GRAY, 0.08, 0.0)
        b.text('yellow_txt', 1.0, 0.0, f"yellow {d['yellow_note']}", GRAY, 0.2, 0.3)
        return
    b.points('yellow_kept', x[keep], y[keep], YELLOW, 0.09, 0.0)
    b.points('yellow_rejected', x[~keep], y[~keep], RED, 0.09, 0.0)
    g = d['yellow_gate']
    if g is not None:
        order = np.argsort(g['x'])
        gx, mid, half = g['x'][order], g['mid'][order], (g['frac'] * g['width'])[order]
        b.strip('gate_mid', gx, mid, WHITE, 0.02, 0.0, 0.6)
        b.strip('gate_band', gx, mid + half, ORANGE, 0.02, 0.0, 0.8)
        b.strip('gate_band', gx, mid - half, ORANGE, 0.02, 0.0, 0.8)
    b.text('yellow_txt', 1.0, 0.0, f'yellow kept={int(keep.sum())} rejected={int((~keep).sum())}', YELLOW, 0.2, 0.3)


def _stage_smoothing(b, d, gen):
    cands_by_src = {}
    for c in d['candidates']:
        cands_by_src.setdefault(c['src'], []).append(c)

    for sm in d['smoothing']:
        src = sm['src']
        b.strip(f'{src}_resampled', sm['rs_x'], sm['rs_y'], GRAY, 0.02, 0.0, 0.7)
        b.strip(f'{src}_smoothed', sm['xs'], sm['ys'], WHITE, 0.05, 0.02)
        idx = np.arange(0, sm['n'], 10)
        tl = 0.4
        b.segments(f'{src}_tangent', sm['xs'][idx], sm['ys'][idx],
                   sm['xs'][idx] + tl * sm['tx'][idx], sm['ys'][idx] + tl * sm['ty'][idx], BLUE, 0.03, 0.03)
        for c in cands_by_src.get(src, []):
            idx5 = np.arange(0, sm['n'], 5)
            b.segments(f'{src}_normal_{c["lane"]}_{c["kind"]}', c['base_x'][idx5], c['base_y'][idx5],
                       c['pre_x'][idx5], c['pre_y'][idx5], KIND_COLOR[c['kind']], 0.01, 0.01, 0.6)
        b.text('smooth_txt', sm['xs'][0], sm['ys'][0],
               f"{src} {sm['mode']} n={sm['n']} wp={sm['wp']} wt={sm['wt']}", WHITE, 0.16, 0.1)


def _stage_candidates(b, d, gen):
    for c in d['candidates']:
        col = KIND_COLOR[c['kind']]
        alpha = 1.0 if c['used'] else 0.25
        width = 0.03 + 0.08 * (0.25 / max(c['frac'], 1e-3)) ** 2   # thicker = higher fusion weight
        b.strip(f"{c['lane']}_{c['kind']}", c['x'], c['y'], col, width, 0.0, alpha)
        if c['n_reordered'] > 0:
            b.strip('REORDERED_pre_sort', c['pre_x'], c['pre_y'], RED, 0.02, 0.05)
        s = f"{c['lane']} {c['kind']} f={c['frac']:.2f} {c['src']}"
        s += '' if c['used'] else ' (UNUSED)'
        s += f" reorder={c['n_reordered']}" if c['n_reordered'] else ''
        b.text('cand_txt', c['x'][0], c['y'][0], s, col if c['used'] else GRAY, 0.15, 0.2)


def _stage_fusion(b, d, gen):
    for side, scol in (('left', LEFT), ('right', RIGHT)):
        tx, ty = d['track'].get(side, (None, None))
        if tx is not None:
            b.points(f'{side}_track_memory', tx, ty, GRAY, 0.05, -0.02, 0.5)
        rec = d['fuse'].get(side)
        if rec is None or 'raw_x' not in rec:
            continue
        w = np.clip(rec['sum_w'], 0.0, 1.0)
        cols = [(1.0 - t, t, 0.2) for t in w]       # red = barely supported, green = full weight
        b.points(f'{side}_1_raw_mean(color=weight)', rec['raw_x'], rec['raw_y'], BLUE, 0.1, 0.0, colors=cols)
        b.strip(f'{side}_2_masked_smooth', rec['sm_x'], rec['sm_y'], PURPLE, 0.04, 0.01)
        if 'ema_x' in rec:
            b.strip(f'{side}_3_ema', rec['ema_x'], rec['ema_y'], WHITE, 0.04, 0.02)
            b.strip(f'{side}_4_locked_final', rec['lock_x'], rec['lock_y'], scol, 0.07, 0.03)
            b.spheres(f'{side}_start', rec['lock_x'][:1], rec['lock_y'][:1], scol, 0.25, 0.03)


def _stage_bezier(b, d, gen):
    for side, scol in (('left', LEFT), ('right', RIGHT)):
        pre = d['pre_bezier'].get(side)
        if pre is not None:
            b.strip(f'{side}_pre_bezier', pre[:, 0], pre[:, 1], GRAY, 0.04, 0.0, 0.7)
        bz = d['bezier'].get(side)
        if bz is None:
            continue
        post = bz['post']
        b.strip(f'{side}_post_bezier', post[:, 0], post[:, 1], scol, 0.08, 0.02)
        c = bz['ctrl']
        b.strip(f'{side}_ctrl_polygon', c[:, 0], c[:, 1], ORANGE, 0.015, 0.03, 0.8)
        b.spheres(f'{side}_ctrl', c[:, 0], c[:, 1], ORANGE, 0.15, 0.03)
        b.spheres(f'{side}_join', pre[bz['j']:bz['j'] + 1, 0], pre[bz['j']:bz['j'] + 1, 1], RED, 0.22, 0.03)
        b.text('bezier_txt', c[3, 0], c[3, 1],
               f"{side} j={bz['j']}/{bz['n']} chord={bz['chord']:.1f} lat={bz['lateral']:.2f}", scol, 0.18, 0.3)


_STAGE_FUNCS = {
    's0_status': _stage_status, 's1_lines': _stage_lines, 's2_memory': _stage_memory,
    's3_yellow': _stage_yellow, 's4_smoothing': _stage_smoothing, 's5_candidates': _stage_candidates,
    's6_fusion': _stage_fusion, 's7_bezier': _stage_bezier,
}


def build_markers(dbg, gen, header):
    """Returns {stage_suffix: MarkerArray}."""
    out = {}
    for name in STAGE_TOPICS:
        b = _Builder(header)
        _STAGE_FUNCS[name](b, dbg, gen)
        out[name] = b.arr
    return out