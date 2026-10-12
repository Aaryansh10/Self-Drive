import numpy as np
import torch
import math
import torch.nn.functional as F

class CenterlineGenerator():
    def __init__(
        self, max_gap, device, bin_width, max_bins,
        max_bad_frame_streak, max_range, nominal_road_width, 
        min_road_width, max_road_width, min_bin_points, 
        max_width_update, max_sg_window_size, min_sg_window_size,
        normal_blend_tx_min, normal_blend_tx_max, min_offset_fraction,
        yellow_gate_fraction, path_spacing, path_start, final_smooth_window,
        
    ):
        self.max_gap = max_gap
        self.device = device
        self.bin_width = bin_width
        self.max_bins = max_bins
        self.max_bad_frame_streak = max_bad_frame_streak
        self.max_range = max_range
        self.nominal_road_width = nominal_road_width
        self.min_road_width = min_road_width
        self.max_road_width = max_road_width
        self.min_bin_points = min_bin_points
        self.max_width_update = max_width_update
        self.max_sg_window_size = max_sg_window_size
        self.min_sg_window_size = min_sg_window_size
        self.normal_blend_tx_min = normal_blend_tx_min
        self.normal_blend_tx_max = normal_blend_tx_max
        self.min_offset_fraction = min_offset_fraction
        self.yellow_gate_fraction = yellow_gate_fraction
        self.path_spacing = path_spacing
        self.path_start = path_start
        self.final_smooth_window = final_smooth_window

        self.dtype = torch.float32
        self.min_offset = nominal_road_width * self.min_offset_fraction
        self.num_out = int(math.ceil((max_range - path_start) / path_spacing)) + 1
        self.d_grid = torch.linspace(
            self.path_start, self.max_range, self.num_out, device=self.device, dtype=self.dtype
        )

        self.reset_memory()

    # --- Entry point ---
    
    @torch.no_grad()
    def generate_centerlines(self, pc_whitelane, pc_yellow = None):
        near = {'left': [], 'right': []}
        far = {'left': [], 'right': []}
        yellow = {'left': [], 'right': []}

        if pc_whitelane is None or pc_whitelane.shape[0] == 0:
            return self.failed_frame()

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
                self.WHITE_LEFT if line_is_left else self.WHITE_RIGHT
            )
            start += n
            for lane, kind, centerline_x, centerline_y, radial_dist in (candidates or []):
                (near if kind == 'near' else far)[lane].append(centerline_x, centerline_y, radial_dist)

        if (not near['left'] or not near['right']) and pc_yellow is not None and pc_yellow.shape[0] > 0:
            self.get_yellow_candidates(pc_yellow, yellow)

        left_xy = self.fuse_side()

    # --- Memory update ---

    def failed_frame(self):
        # Increments bad frame streak if there is no point cloud and resets the memory if it goes beyond the maximum

        self.bad_frame_streak += 1
        if self.bad_frame_streak >= self.max_bad_frame_streak:
            self.reset_memory()
        return None, None

    def reset_memory(self):
        nan = float('nan')
        num_bins, num_out = self.num_bins, self.num_out
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
        confirmed = left_has_enough & right_has_enough & plausible

        aw = self.width_ema_alpha
        target = aw * raw_road_width.clamp(self.min_road_width, self.max_road_width) + (1 - aw) * self.road_width 
        step = (target - self.road_width).clamp(-self.max_width_update, self.max_width_update)
        self.road_width = torch.where(confirmed, self.road_width + step, self.road_width)

    # --- Lane splitting and labelling ---

    def split_lines(self, x, y):
        # Calculates and returns the line id of each point and the number of points in each line

        dist_bw_points = torch.hypot(torch.diff(x), torch.diff(y))
        new_line_starts = torch.cat((
            torch.zeros(1, dtype=self.dtype, device=self.device), 
            dist_bw_points > self.max_gap
        ))
        line_id = torch.cumsum(new_line_starts, dim=0) - 1
        pts_per_line = torch.bincount(line_id)
        return line_id, pts_per_line

    def get_dist_bin_width(self, x):
        # Calculates and returns the distance of each point in terms of bin width
        return torch.clamp(x / self.bin_width, 0, self.max_bins - 1)

    def label_lines(self, y, line_id, pt_dist_bin_width, pts_per_line):
        # Finds out and returns if a line is left by majority voting of the bins of each line
        
        prev_left_y = self.prev_left_y[pt_dist_bin_width]
        prev_right_y = self.prev_right_y[pt_dist_bin_width]
        road_width = self.road_width

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
            line_id, weights=pts_vote_left, minlength=pts_per_line.numel()
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

        window = min(n, self.max_sg_window_size)
        if window % 2 == 0:
            window -= 1

        if window >= self.min_sg_window_size:
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
            normal_x, normal_y = self.get_inward_normal()
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
            for lane, kind, centerline_x, centerline_y, radial_dist in (candidates, []):
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

        x_mean = self._box_smooth(x_mean, self.final_smooth_window)
        y_mean = self._box_smooth(y_mean, self.final_smooth_window)

        if int()


    # --- Savitzky Golay ---

    def find_sg_coeffs(self, window, deriv):
        # Finds the Savitzky Golay coefficients and returns the kernel

        key = window, deriv
        is_cached = self.sg_coeffs.get(key)
        if is_cached is not None:
            return is_cached

        half = window // 2
        idx = np.arange(-half, half + 1, np.float64)
        
        A = np.vander(idx, self.sg_polyorder + 1, increasing=True)
        coeffs = np.linalg.pinv(A)[deriv] * math.factorial(deriv)
        kernel = torch.as_tensor(
            coeffs.astype(np.float32), device=self.device
        ).view

        self.find_sg_coeffs[key] = kernel
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
        
        idx = torch.searchsorted(x_points, x_query).clamp(1, x_points.numel())

        x_left, x_right = x_points[idx - 1], x_points[idx]
        f_left, f_right = f_points[idx - 1], f_points[idx]

        ratio = (x_query - x_left) / (x_right - x_left).clamp(min=1e-6)
        return f_left + ratio * (f_right - f_left)