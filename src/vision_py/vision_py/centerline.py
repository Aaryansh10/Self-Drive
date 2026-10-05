import numpy as np
import torch
import torch.nn.functional as F

class CenterlineGenerator():
    def __init__(
        self, line_gap, device, bin_width,
    ):
        self.cluster_gap = line_gap
        self.device = device
        self.bin_width = bin_width
        self.num_bins
        self.dtype = torch.float32
        
    def failed_frame(self):
        pass

    def split_lines(self, x, y):
        dist_bw_pts = torch.hypot(torch.diff(x), torch.diff(y))
        new_line_start = torch.cat((
            torch.ones(1, dtype=torch.bool, device=self.device),
            dist_bw_pts > self.line_gap
        ))
        pts_id = torch.cumsum(new_line_start, dim=0) - 1
        pts_per_line = torch.bincount(pts_id)
        return pts_id, pts_per_line

    def dist_bin_pts(self, x):
        return (x / self.bin_size).to(torch.int64).clamp(0, self.bin_size - 1)

    def label_lines(self, y, dist_bin_of_pts, pts_id, pts_per_line):
        prev_left_y = self.prev_left_y[dist_bin_of_pts]
        prev_rigth_y = self.prev_right_y[dist_bin_of_pts]
        lane_width = self.lane_width[dist_bin_of_pts]

        left_unknown = torch.isnan(prev_left_y)
        right_unknown = torch.isnan(prev_rigth_y)

        prev_left_y = torch.where(
            left_unknown & ~right_unknown, prev_rigth_y + lane_width, prev_left_y
        )
        prev_rigth_y = torch.where(
            right_unknown & ~left_unknown, prev_left_y - lane_width, prev_rigth_y
        )
        no_memory = left_unknown & right_unknown

        closer_to_left = (y - prev_left_y).abs() <= (y - prev_rigth_y).abs()
        pts_vote_left = torch.where(no_memory, y > 0.0, closer_to_left)

        left_votes_per_line = torch.bincount(
            pts_id, weights=pts_vote_left.to(self.dtype),
            minlength=pts_per_line.numel()
        )
        line_is_left = (left_votes_per_line * 2) >= pts_per_line
        return line_is_left

    def update_memory(self, y, dist_bin_of_pts, pt_is_left):



    # --- ENTRY POINT ---
    @torch.no_grad()
    def generate_centerlines_torch(self, pc_whitelane):
        if pc_whitelane is None or pc_whitelane.shape[0] == 0:
            return self.failed_frame()

        x = pc_whitelane[:, 0]
        y = pc_whitelane[:, 1]

        pts_id, pts_per_line = self.split_lines(x, y)
        dist_bin_of_pts = self.dist_bin_pts(x)
        line_is_left = self.label_lines(y, dist_bin_of_pts, pts_id, pts_per_line)
        pt_is_left = line_is_left[pts_id]

        self.update_memory(y, dist_bin_of_pts, pt_is_left)
        



    # --- LOW LEVEL HELPERS ---
    def sum_and_count_bins(self, bin_of_pt, values):
        sums = torch.zeros(self.num_bins)