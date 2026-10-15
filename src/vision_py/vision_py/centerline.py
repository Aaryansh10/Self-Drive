import numpy as np
import torch
import math
import torch.nn.functional as F

class CenterlineGenerator():

    WHITE_LEFT = (('left', 0.25, True, 'near'), ('right', 0.75, True, 'far'))
    WHITE_RIGHT = (('right', 0.25, False, 'near'), ('left', 0.75, False, 'far'))
    YELLOW = (('left', 0.25, False, 'yellow'), ('right', 0.25, True, 'yellow'))

    def __init__(
        self, max_gap=1.0, device='cuda', bin_width=0.5, max_bad_frame_streak=20, 
        max_range=15.0, nominal_road_width=7.4, min_road_width=6.6, 
        max_road_width=8.2, min_bin_points=1, max_width_update=0.35, 
        max_sg_window_size=11,  # unused: windows are now metric (position_window_m / tangent_window_m)
        normal_blend_tx_min=None, normal_blend_tx_max=None, 
        min_offset_fraction=0.5, yellow_gate_fraction=0.25, path_spacing=0.1, path_start=0.2, 
        final_smooth_window=25, near_field_range=2.0, min_near_pts=2, 
        width_stale_decay_frames=30, width_stale_decay_alpha=0.05, sg_polyorder=2,
        track_ema_alpha=0.35, width_ema_alpha=0.15, centerline_ema_alpha=0.4,
        width_update_max_x = 2.5, resample_step=0.1, position_window_m=1.0, tangent_window_m=2.5,
        far_near_field_range=4.0, lane_change_bezier=True, lane_change_factor=2.5,
        lane_change_min_len=3.0, bezier_handle=0.4
    ):
        self.max_gap = max_gap
        self.device = device
        self.bin_width = bin_width
        self.max_bad_frame_streak = max_bad_frame_streak
        self.max_range = max_range
        self.nominal_road_width = nominal_road_width
        self.min_road_width = min_road_width
        self.max_road_width = max_road_width
        self.min_bin_points = min_bin_points
        self.max_width_update = max_width_update
        self.width_stale_decay_frames = max(1, width_stale_decay_frames)
        self.width_stale_decay_alpha = width_stale_decay_alpha
        # Optional lateral-normal fallback. Disabled by default: vision.py declares
        # these parameters but never applies them, and forcing a lateral offset when
        # tangent_x is small points the offset the wrong way at bends.
        self.normal_blend_tx_min = normal_blend_tx_min
        self.normal_blend_tx_max = normal_blend_tx_max
        self.use_normal_blend = normal_blend_tx_min is not None and normal_blend_tx_max is not None
        self.min_offset_fraction = min_offset_fraction
        self.yellow_gate_fraction = yellow_gate_fraction
        self.path_spacing = path_spacing
        self.path_start = path_start
        self.final_smooth_window = final_smooth_window
        self.near_field_range = near_field_range
        self.min_near_pts = min_near_pts
        self.track_ema_alpha = track_ema_alpha
        self.width_ema_alpha = width_ema_alpha
        self.centerline_ema_alpha = centerline_ema_alpha
        self.width_update_max_x = width_update_max_x
        self.resample_step = resample_step
        self.position_window_m = position_window_m
        self.tangent_window_m = tangent_window_m
        # Far-lane paths are offset ~0.75 road widths sideways, so they never get close to the robot
        # radially; they are allowed to start further out than same-lane paths.
        self.far_near_field_range = far_near_field_range
        # Bezier lane-change blending: the path leaves the robot along its heading (+x) and joins the
        # lane centerline tangentially. Chord length = max(min_len, factor * lateral offset).
        self.lane_change_bezier = lane_change_bezier
        self.lane_change_factor = lane_change_factor
        self.lane_change_min_len = lane_change_min_len
        self.bezier_handle = bezier_handle

        self.dtype = torch.float32
        self.max_bins = int(math.ceil(max_range / bin_width))
        self.min_offset = nominal_road_width * self.min_offset_fraction
        self.num_out = int(math.ceil((max_range - path_start) / path_spacing)) + 1
        self.d_grid = torch.linspace(
            self.path_start, self.max_range, self.num_out, device=self.device, dtype=self.dtype
        )
        self.sg_polyorder = max(1, sg_polyorder)
        self.sg_min_window = self.sg_polyorder + 1
        if self.sg_min_window % 2 == 0:
            self.sg_min_window += 1
        self.sg_coeffs = {}
        self.ref_fraction = 0.25      # smallest offset fraction in use; gets weight 1.0
        self.taper_len = 1.0          # metres over which a candidate's weight ramps in/out

        self.reset_memory()

    # --- Entry point ---
    
    @torch.no_grad()
    def generate_centerlines(self, pc_whitelane, pc_yellow=None):
        near = {'left': [], 'right': []}
        far = {'left': [], 'right': []}
        yellow = {'left': [], 'right': []}

        if pc_whitelane is not None and pc_whitelane.shape[0] > 0:
            pc_whitelane = torch.as_tensor(pc_whitelane).to(self.device, self.dtype)
            x, y = pc_whitelane[:, 0], pc_whitelane[:, 1]
            line_id, pts_per_line = self.split_lines(x, y)

            pt_dist_bin_width = self.get_dist_bin_width(x)
            line_is_left = self.label_lines(y, line_id, pt_dist_bin_width, pts_per_line)
            pt_is_left = line_is_left[line_id]
            self.update_memory(y, pt_dist_bin_width, pt_is_left)

            start = 0
            for n, is_left in zip(pts_per_line.tolist(), line_is_left.tolist()):
                candidates = self.line_centerline(
                    x[start:start + n], y[start:start + n],
                    self.WHITE_LEFT if is_left else self.WHITE_RIGHT
                )
                start += n
                for lane, kind, cx, cy, rd, frac in (candidates or []):
                    (near if kind == 'near' else far)[lane].append((cx, cy, rd, frac))

        if (not near['left'] or not near['right']) and pc_yellow is not None and pc_yellow.shape[0] > 0:
            pc_yellow = torch.as_tensor(pc_yellow).to(self.device, self.dtype)
            self.get_yellow_candidates(pc_yellow, yellow)

        left_xy = self.fuse_side('left', self.select(near['left'], far['left'], yellow['left']))
        right_xy = self.fuse_side('right', self.select(near['right'], far['right'], yellow['right']))

        if left_xy is None and right_xy is None:
            return self.failed_frame()

        self.bad_frame_streak = 0
        if self.lane_change_bezier:
            # Applied after fuse_side, so the tracked (EMA) memory stays raw
            left_xy = self.bezier_lane_change(left_xy)
            right_xy = self.bezier_lane_change(right_xy)
        return left_xy, right_xy
    
    # --- Memory update ---

    def failed_frame(self):
        # Increments bad frame streak if there is no point cloud and resets the memory if it goes beyond the maximum

        self.bad_frame_streak += 1
        if self.bad_frame_streak >= self.max_bad_frame_streak:
            self.reset_memory()
        return None, None

    def reset_memory(self):
        # Resets all the memory of the tracked lanes

        nan = float('nan')
        num_bins, num_out = self.max_bins, self.num_out
        kw = dict(device=self.device, dtype=self.dtype)
        
        self.prev_left_y = torch.full((num_bins,), nan, **kw)
        self.prev_right_y = torch.full((num_bins,), nan, **kw)
        self.road_width = torch.full((num_bins,), self.nominal_road_width, **kw)
        self.width_stale = torch.zeros((num_bins,), **kw)
        self.track = {
            'left': torch.full((2, num_out), nan, **kw),
            'right': torch.full((2, num_out), nan, **kw)
        }
        self.bad_frame_streak = 0

    def update_memory(self, y, pt_dist_bin_width, pt_is_left):
        # Updates the previous left and right track along with the road width

        left_sum, left_count = self.sum_and_count_bins(
            y[pt_is_left], pt_dist_bin_width[pt_is_left] 
        )
        right_sum, right_count = self.sum_and_count_bins(
            y[~pt_is_left], pt_dist_bin_width[~pt_is_left]
        )

        left_has_enough = left_count >= self.min_bin_points
        right_has_enough = right_count >= self.min_bin_points
        left_mean = left_sum / left_count.clamp(min=1)
        right_mean = right_sum / right_count.clamp(min=1)

        a = self.track_ema_alpha

        def ema(prev, curr, has):
            blended = torch.where(torch.isnan(prev), curr, a * curr + (1.0 - a) * prev)
            return torch.where(has, blended, prev)

        self.prev_left_y = ema(self.prev_left_y, left_mean, left_has_enough)
        self.prev_right_y = ema(self.prev_right_y, right_mean, right_has_enough)

        raw_road_width = left_mean - right_mean
        plausible = (raw_road_width >= self.min_road_width) & (raw_road_width <= self.max_road_width)
        bin_x = (torch.arange(self.max_bins, device=self.device, dtype=self.dtype) + 0.5) * self.bin_width
        trusted = bin_x <= self.width_update_max_x
        confirmed = left_has_enough & right_has_enough & plausible & trusted

        aw = self.width_ema_alpha
        target = aw * raw_road_width.clamp(self.min_road_width, self.max_road_width) + (1 - aw) * self.road_width 
        step = (target - self.road_width).clamp(-self.max_width_update, self.max_width_update)
        self.road_width = torch.where(confirmed, self.road_width + step, self.road_width)

        self.width_stale = torch.where(
            confirmed, torch.zeros_like(self.width_stale), self.width_stale + 1.0
        )
        stale = self.width_stale > self.width_stale_decay_frames
        decayed = self.road_width + self.width_stale_decay_alpha * (
            self.nominal_road_width - self.road_width
        )
        self.road_width = torch.where(stale, decayed, self.road_width)

        fresh = trusted & (self.width_stale <= self.width_stale_decay_frames)
        if bool(fresh.any()):
            self.road_width = torch.where(~trusted, self.road_width[fresh].mean(), self.road_width)

    # --- Lane splitting and labelling ---

    def split_lines(self, x, y):
        # Calculates and returns the line id of each point and the number of points in each line

        dist_bw_points = torch.hypot(torch.diff(x), torch.diff(y))
        new_line_starts = torch.cat((
            torch.ones(1, dtype=torch.bool, device=self.device), 
            dist_bw_points > self.max_gap
        ))
        line_id = torch.cumsum(new_line_starts, dim=0) - 1
        pts_per_line = torch.bincount(line_id)
        return line_id, pts_per_line

    def get_dist_bin_width(self, x):
        # Calculates and returns the distance of each point in terms of bin width
        return torch.clamp((x / self.bin_width).to(torch.int64), 0, self.max_bins - 1)

    def label_lines(self, y, line_id, pt_dist_bin_width, pts_per_line):
        # Finds out and returns if a line is left by majority voting of the bins of each line
        
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
        return line_is_left

    # --- Centerline functions ---

    def odd_window(self, meters, n):
        # Converts a window length in metres to an odd sample count that fits in n samples
        w = int(round(meters / self.resample_step))
        w = min(w, n)
        return w if w % 2 == 1 else w - 1

    def resample_line(self, x, y):
        # Resamples a line to uniform arclength spacing. The raw leading-edge points are one per
        # image column: dense near the robot, sparse far away, and a staircase at range (one pixel
        # row spans a lot of ground, so x stays constant while y steps). A fixed sample-count
        # window then covers wildly different physical lengths; on a uniform grid it doesn't.

        seg = torch.hypot(torch.diff(x), torch.diff(y))
        arclen = torch.cat((torch.zeros(1, device=self.device, dtype=self.dtype), torch.cumsum(seg, 0)))
        total = float(arclen[-1])
        if total < 3 * self.resample_step:
            return None
        n = int(total / self.resample_step) + 1
        query = torch.linspace(0.0, total, n, device=self.device, dtype=self.dtype)
        return self.interpolate(query, arclen, x), self.interpolate(query, arclen, y)

    def line_centerline(self, x, y, offsets):
        # Generates and returns the centerline candidates by offseting using the normal vector

        if x.numel() < 3:
            return None

        if torch.hypot(x[0], y[0]) > torch.hypot(x[-1], y[-1]):
            x = x.flip(0)
            y = y.flip(0)

        resampled = self.resample_line(x.contiguous(), y.contiguous())
        if resampled is None:
            return None
        x, y = resampled
        n = x.numel()

        # Position uses a short window; the tangent uses a longer one, because any tangent error
        # is multiplied by the offset (up to ~0.75 * road width) when the path is pushed sideways.
        wp = self.odd_window(self.position_window_m, n)
        wt = self.odd_window(self.tangent_window_m, n)

        if wt >= self.sg_min_window:
            wp = max(wp, self.sg_min_window)
            x_smoothed = self.sg_apply(x, wp, 0)
            y_smoothed = self.sg_apply(y, wp, 0)
            tangent_x = self.sg_apply(x, wt, 1)
            tangent_y = self.sg_apply(y, wt, 1)
        else:
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
            radial_dist = torch.hypot(centerline_x, centerline_y)
            order = torch.argsort(radial_dist)
            centerline_x, centerline_y = centerline_x[order], centerline_y[order]
            radial_dist = radial_dist[order]
            # Safety to make sure interpolation math doesnt break
            radial_dist = radial_dist + torch.arange(
                radial_dist.numel(), device=self.device, dtype=self.dtype
            ) * 1e-4
            out.append((lane, kind, centerline_x, centerline_y, radial_dist, fraction))

        return out

    def get_yellow_candidates(self, pc_yellow, yellow):
        # Generates yellow centerline candidates

        x, y = pc_yellow[:, 0], pc_yellow[:, 1]
        keep = self.yellow_gate(x, y)
        x, y = x[keep], y[keep]

        if x.numel() < 3:
            return

        line_id, pts_per_line = self.split_lines(x, y)
        
        start = 0
        for n in pts_per_line.tolist():
            candidates = self.line_centerline(
                x[start:start + n], y[start:start + n],
                self.YELLOW 
            )
            start += n
            for lane, kind, centerline_x, centerline_y, radial_dist, frac in (candidates or []):
                yellow[lane].append((centerline_x, centerline_y, radial_dist, frac))

    def yellow_gate(self, x, y):
        # Acts as a gate and returns a mask for those yellow points which are close to the center of the lane

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
        return ~known | ((y - mid).abs() <= self.yellow_gate_fraction * width)

    def fuse_side(self, side, candidates):
        # Creates and returns the final path by blending all the candidate paths

        if not candidates:
            return None

        sum_x = torch.zeros_like(self.d_grid)
        sum_y = torch.zeros_like(self.d_grid)
        sum_w = torch.zeros_like(self.d_grid)

        for candidate_x, candidate_y, radial_dist, fraction in candidates:
            lo, hi = radial_dist[0], radial_dist[-1]
            valid = (self.d_grid >= lo) & (self.d_grid <= hi)
            if not bool(valid.any()):
                continue
            dist_grid = self.d_grid[valid]

            # Reliability weight. A candidate offset by `fraction` of the road width
            # carries `fraction * width_error` of lateral error, so a 0.75 offset is
            # 3x noisier than a 0.25 one. Weight by 1 / offset^2 (near/yellow = 1,
            # far = 1/9) instead of averaging them equally.
            w = (self.ref_fraction / max(fraction, 1e-3)) ** 2

            # Ramp the weight in/out over the candidate's radial span so a candidate
            # appearing or disappearing doesn't make the fused path jump. The floor
            # keeps a sole candidate at full effect (weights cancel when normalised).
            edge = torch.minimum(dist_grid - lo, hi - dist_grid)
            ramp = (edge / self.taper_len).clamp(0.0, 1.0)
            weight = w * (0.02 + 0.98 * ramp)

            sum_x[valid] += weight * self.interpolate(dist_grid, radial_dist, candidate_x)
            sum_y[valid] += weight * self.interpolate(dist_grid, radial_dist, candidate_y)
            sum_w[valid] += weight

        valid_grid = sum_w > 0
        if int(valid_grid.sum()) < 3:
            return None

        x_mean = sum_x / sum_w.clamp(min=1e-9)
        y_mean = sum_y / sum_w.clamp(min=1e-9)

        # Smooth on the FULL uniform grid with a validity mask, then compact. The old
        # code compacted first, so a gap in coverage got smoothed across as if the
        # points on either side were neighbours, and the replicate padding dragged both
        # ends of the path inward.
        x_mean = self.masked_smooth(x_mean, valid_grid, self.final_smooth_window)[valid_grid]
        y_mean = self.masked_smooth(y_mean, valid_grid, self.final_smooth_window)[valid_grid]

        # The path has to start close to the robot in forward distance. This used to count grid points
        # with x < near_field_range, but the grid is spaced in *radial* distance: a lane 3-6 m to the
        # side packs only 1-3 samples into the first 2 m of x, so far-lane paths almost always failed.
        has_close = any(c[3] <= self.ref_fraction + 1e-6 for c in candidates)
        start_limit = self.near_field_range if has_close else self.far_near_field_range
        if x_mean.numel() < self.min_near_pts or float(x_mean.min()) > start_limit:
            return None

        track = self.track[side]
        prev_x = track[0][valid_grid]
        prev_y = track[1][valid_grid]
        has_prev = ~torch.isnan(prev_x)

        alpha = self.centerline_ema_alpha
        smoothed_x = torch.where(has_prev, alpha * x_mean + (1.0 - alpha) * prev_x, x_mean)
        smoothed_y = torch.where(has_prev, alpha * y_mean + (1.0 - alpha) * prev_y, y_mean)

        smoothed_x, smoothed_y = self.lock_near_field(smoothed_x, smoothed_y)
        track[0][valid_grid] = smoothed_x
        track[1][valid_grid] = smoothed_y
        return torch.stack([smoothed_x, smoothed_y], dim=1)

    def anchor_to_origin(self, path_xy, blend_dist=None):
        # Kept for compatibility with vision.py. Now a Bezier lane-change blend (blend_dist unused);
        # a no-op if the path already starts at the origin.

        if path_xy is None or path_xy.shape[0] == 0:
            return path_xy
        if float(torch.hypot(path_xy[0, 0], path_xy[0, 1])) < 1e-3:
            return path_xy
        return self.bezier_lane_change(path_xy)

    def bezier_lane_change(self, path_xy):
        # Replaces the start of the path with a cubic Bezier that leaves the origin along the robot
        # heading (+x) and joins the lane centerline tangentially. The longer the lateral offset, the
        # longer the transition, so an adjacent-lane path is reached with a gentle S-curve instead of
        # a sharp jump. Returns a path starting at (0, 0).

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

        return torch.cat((torch.stack((bx, by), dim=1)[:-1], path_xy[j:]), dim=0)

    def lock_near_field(self, smoothed_x, smoothed_y):
        # Re-projects the first two points of the path to prevent hooks at turns, and returns the new path

        if smoothed_x.numel() < 5:
            return smoothed_x, smoothed_y

        smoothed_x = smoothed_x.clone()
        smoothed_y = smoothed_y.clone()

        heading_x = smoothed_x[4] - smoothed_x[2]
        heading_y = smoothed_y[4] - smoothed_y[2]

        heading_norm = torch.hypot(heading_x, heading_y) + 1e-5
        heading_x = heading_x / heading_norm
        heading_y = heading_y / heading_norm

        d0 = torch.hypot(smoothed_x[2] - smoothed_x[0], smoothed_y[2] - smoothed_y[0])
        d1 = torch.hypot(smoothed_x[2] - smoothed_x[1], smoothed_y[2] - smoothed_y[1])

        smoothed_x[0] = smoothed_x[2] - heading_x * d0
        smoothed_y[0] = smoothed_y[2] - heading_y * d0
        smoothed_x[1] = smoothed_x[2] - heading_x * d1
        smoothed_y[1] = smoothed_y[2] - heading_y * d1

        return smoothed_x, smoothed_y

    def select(self, near, far, yellow):
        # Selects which candidates to fuse for a lane. The far candidate is no longer dropped when
        # yellow exists; fuse_side already weights it 1/9 of a near/yellow candidate.

        if near:
            return near + far
        return yellow + far

    # --- Savitzky Golay ---

    def find_sg_coeffs(self, window, deriv):
        # Returns (interior kernel, left-edge matrix, right-edge matrix) for a window/derivative.
        # The edge matrices evaluate the polynomial fitted to the first/last `window` samples
        # at the samples that do not have a full window around them. This replaces replicate
        # padding, which flattens the slope at the ends of a line and rotates the tangent
        # (and therefore the offset normal) there.

        key = window, deriv
        cached = self.sg_coeffs.get(key)
        if cached is not None:
            return cached

        half = window // 2
        idx = np.arange(-half, half + 1, dtype=np.float64)
        order = self.sg_polyorder

        A = np.vander(idx, order + 1, increasing=True)
        pinv = np.linalg.pinv(A)                      # (order+1, window): samples -> poly coeffs
        coeffs = pinv[deriv] * math.factorial(deriv)

        def eval_rows(taus):
            rows = []
            for tau in taus:
                if deriv == 0:
                    rows.append([tau ** k for k in range(order + 1)])
                else:
                    rows.append([k * tau ** (k - 1) if k >= 1 else 0.0 for k in range(order + 1)])
            return np.array(rows, dtype=np.float64)

        # Slope at the ends comes from a straight-line fit (a quadratic extrapolated to the last
        # sample is far too noisy, and the tangent error is multiplied by the lane offset).
        edge_order = min(order, 1) if deriv == 1 else order
        edge_pinv = np.linalg.pinv(np.vander(idx, edge_order + 1, increasing=True))
        edge_pinv = np.pad(edge_pinv, ((0, order - edge_order), (0, 0)))  # zero higher coeffs

        left = eval_rows(np.arange(-half, 0)) @ edge_pinv      # window centre is sample `half`
        right = eval_rows(np.arange(1, half + 1)) @ edge_pinv  # window centre is sample L-1-half

        to_t = lambda a: torch.as_tensor(a.astype(np.float32), device=self.device)
        out = (to_t(coeffs).view(1, 1, -1), to_t(left), to_t(right))
        self.sg_coeffs[key] = out
        return out

    def sg_apply(self, arr, window, deriv):
        # Applies the Savitzky Golay filter (value or slope) and returns the result.
        # Requires arr.numel() >= window.

        kernel, left, right = self.find_sg_coeffs(window, deriv)
        a = arr.contiguous()
        interior = F.conv1d(a.view(1, 1, -1), kernel).view(-1)
        if window < 3:
            return interior
        head = left @ a[:window]
        tail = right @ a[-window:]
        return torch.cat((head, interior, tail))

    # --- Low level helper functions ---

    def sum_and_count_bins(self, y, pt_dist_bin_width, ):
        # Calculates and returns the sum of all the y-coordinates and the number of of points in each bin 
        
        sums = torch.zeros(self.max_bins, device=self.device, dtype=self.dtype)
        sums.scatter_add_(0, pt_dist_bin_width, y)        

        counts = torch.bincount(pt_dist_bin_width, minlength=self.max_bins)
        return sums, counts

    def box_smooth(self, arr, w):
        # Smooths the array by applying average pooling

        w = min(w, arr.numel()) 
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return arr
        pad = w // 2
        padded = F.pad(arr.view(1, 1, -1), (pad, pad), mode='replicate')
        return F.avg_pool1d(padded, kernel_size=w, stride=1).view(-1)

    def masked_smooth(self, values, valid, w):
        # Windowed local-linear smoother on a uniform grid that ignores invalid samples.
        # Interior samples get the plain box average (same as before); samples near a gap
        # or an end of the path fit a line through the valid neighbours instead of being
        # pulled toward the edge value.

        w = min(w, int(valid.sum()))
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return values

        h = w // 2
        v = valid.to(self.dtype)
        val = torch.where(valid, values, torch.zeros_like(values))
        t = torch.arange(-h, h + 1, device=self.device, dtype=self.dtype)

        def corr(sig, kernel):
            return F.conv1d(F.pad(sig.view(1, 1, -1), (h, h)), kernel.view(1, 1, -1)).view(-1)

        one = torch.ones_like(t)
        s0, s1, s2 = corr(v, one), corr(v, t), corr(v, t * t)
        sv, stv = corr(val, one), corr(val, t)

        det = s0 * s2 - s1 * s1
        linear = (s2 * sv - s1 * stv) / det.clamp(min=1e-6)
        box = sv / s0.clamp(min=1e-6)
        return torch.where(det > 0.5, linear, box)

    def get_inward_normal(self, tangent_x, tangent_y, toward_right):
        # Calculares and returns the inward normal vector \

        if toward_right:
            g_normal_x, g_normal_y, lat_y = tangent_y, -tangent_x, -1.0
        else:
            g_normal_x, g_normal_y, lat_y = -tangent_y, tangent_x, 1.0

        if not self.use_normal_blend:
            return g_normal_x, g_normal_y

        span = max(self.normal_blend_tx_max - self.normal_blend_tx_min, 1e-6)
        w_geo = ((tangent_x - self.normal_blend_tx_min) / span).clamp(0.0, 1.0)

        nx = w_geo * g_normal_x
        ny = w_geo * g_normal_y + (1.0 - w_geo) * lat_y
        nn = torch.hypot(nx, ny).clamp(min=1e-6)

        return nx / nn, ny / nn

    def interpolate(self, x_query, x_points, f_points):
        # Interpolates and returns the interpolated points
        
        idx = torch.searchsorted(x_points, x_query).clamp(1, x_points.numel() - 1)

        x_left, x_right = x_points[idx - 1], x_points[idx]
        f_left, f_right = f_points[idx - 1], f_points[idx]

        ratio = (x_query - x_left) / (x_right - x_left).clamp(min=1e-6)
        return f_left + ratio * (f_right - f_left)