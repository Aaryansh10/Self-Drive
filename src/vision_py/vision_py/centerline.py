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
        max_range=15.0, nominal_road_width=8.5, min_road_width=7.5, 
        max_road_width=9.5, min_bin_points=1, max_width_update=0.35, 
        max_sg_window_size=11,  normal_blend_tx_min=0.15, normal_blend_tx_max=0.6, 
        min_offset_fraction=0.5, yellow_gate_fraction=0.25, path_spacing=0.1, path_start=0.2, 
        final_smooth_window=25, near_field_range=2.0, min_near_pts=2, 
        width_stale_decay_frames=30, width_stale_decay_alpha=0.05, sg_polyorder=2,
        track_ema_alpha=0.35, width_ema_alpha=0.15, centerline_ema_alpha=0.4,
        width_update_max_x = 2.5
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
        self.max_sg_window_size = max_sg_window_size
        self.normal_blend_tx_min = normal_blend_tx_min
        self.normal_blend_tx_max = normal_blend_tx_max
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
        self.sg_window_max = max_sg_window_size + (1 - max_sg_window_size % 2)
        self.sg_window_max = max(self.sg_window_max, self.sg_min_window)
        self.sg_coeffs = {}

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
                for lane, kind, cx, cy, rd in (candidates or []):
                    (near if kind == 'near' else far)[lane].append((cx, cy, rd))

        if (not near['left'] or not near['right']) and pc_yellow is not None and pc_yellow.shape[0] > 0:
            pc_yellow = torch.as_tensor(pc_yellow).to(self.device, self.dtype)
            self.get_yellow_candidates(pc_yellow, yellow)

        left_xy = self.fuse_side('left', self.select(near['left'], far['left'], yellow['left']))
        right_xy = self.fuse_side('right', self.select(near['right'], far['right'], yellow['right']))

        if left_xy is None and right_xy is None:
            return self.failed_frame()

        self.bad_frame_streak = 0
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

    def line_centerline(self, x, y, offsets):
        # Generates and returns the centerline candidates by offseting using the normal vector

        n = x.numel()
        if n < 3:
            return None

        if torch.hypot(x[0], y[0]) > torch.hypot(x[-1], y[-1]):
            x = x.flip(0)
            y = y.flip(0)

        window = min(n, self.sg_window_max)
        if window % 2 == 0:
            window -= 1

        if window >= self.sg_min_window:
            x_smoothed = self.sg_apply(x, window, 0)
            y_smoothed = self.sg_apply(y, window, 0)
            tangent_x = self.sg_apply(x, window, 1)
            tangent_y = self.sg_apply(y, window, 1)
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
            out.append((lane, kind, centerline_x, centerline_y, radial_dist))

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
            for lane, kind, centerline_x, centerline_y, radial_dist in (candidates or []):
                yellow[lane].append((centerline_x, centerline_y, radial_dist))

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
        count = torch.zeros_like(self.d_grid)

        for candidate_x, candidate_y, radial_dist in candidates:
            valid = (self.d_grid >= radial_dist[0]) & (self.d_grid <= radial_dist[-1])
            if not bool(valid.any()):
                continue
            dist_grid = self.d_grid[valid]
            sum_x[valid] += self.interpolate(dist_grid, radial_dist, candidate_x)
            sum_y[valid] += self.interpolate(dist_grid, radial_dist, candidate_y)
            count[valid] += 1.0

        valid_grid = count > 0
        if int(valid_grid.sum()) < 3:
            return None

        x_mean = sum_x[valid_grid] / count[valid_grid]
        y_mean = sum_y[valid_grid] / count[valid_grid]

        x_mean = self.box_smooth(x_mean, self.final_smooth_window)
        y_mean = self.box_smooth(y_mean, self.final_smooth_window)

        if int((x_mean < self.near_field_range).sum()) < self.min_near_pts:
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
        # Selects and returns which lane to use

        if near :
            return near + far
        if yellow:
            return yellow
        return far

    # --- Savitzky Golay ---

    def find_sg_coeffs(self, window, deriv):
        # Finds the Savitzky Golay coefficients and returns the kernel

        key = window, deriv
        is_cached = self.sg_coeffs.get(key)
        if is_cached is not None:
            return is_cached

        half = window // 2
        idx = np.arange(-half, half + 1, dtype=np.float64)
        
        A = np.vander(idx, self.sg_polyorder + 1, increasing=True)
        coeffs = np.linalg.pinv(A)[deriv] * math.factorial(deriv)
        kernel = torch.as_tensor(
            coeffs.astype(np.float32), device=self.device
        ).view(1, 1, -1)

        self.sg_coeffs[key] = kernel
        return kernel

    def sg_apply(self, arr, window, deriv):
        # Applies the respective Savitzky Golay kernel and returns the result

        half = window // 2
        padded = F.pad(arr.view(1, 1, -1), (half, half), mode='replicate')
        return F.conv1d(padded, self.find_sg_coeffs(window, deriv)).view(-1)

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

    def get_inward_normal(self, tangent_x, tangent_y, toward_right):
        # Calculares and returns the inward normal vector \

        if toward_right:
            g_normal_x, g_normal_y, lat_y = tangent_y, -tangent_x, -1.0
        else:
            g_normal_x, g_normal_y, lat_y = -tangent_y, tangent_x, 1.0

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