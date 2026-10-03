import numpy as np
import torch

def project_pixels_np(
    v_sub, u_sub, subsample_step, bot_pitch,
    cx, cy, fx, fy, cam_pitch, cam_height, 
    min_angle=0.01, max_range=7.5
):
    N = v_sub.size
    if N == 0:
        return np.empty((0, 3))

    v_full = v_sub * subsample_step
    u_full = u_sub * subsample_step

    alpha = np.arctan((v_full - cy) / fy)
    angles = alpha + cam_pitch - bot_pitch

    ground_mask = angles >= min_angle
    valid_idx = np.nonzero(ground_mask)[0]
    valid_N = valid_idx.size
    if valid_N == 0:
        return np.empty((0, 3))

    valid_alpha = alpha[valid_idx]
    valid_angles = angles[valid_idx]
    valid_u = u_full[valid_idx]

    fwd_dist = cam_height / np.tan(valid_angles)
    range_mask = (fwd_dist > np.float32(0.0)) & (fwd_dist <= max_range)
    range_idx = np.nonzero(range_mask)[0]
    range_N = range_idx.size
    if range_N == 0:
        return np.empty((0, 3))

    range_u = valid_u[range_idx]
    range_alpha = valid_alpha[range_idx]
    range_angles = valid_angles[range_idx]
    range_fwd_dist = fwd_dist[range_idx]

    point_cloud = np.zeros((range_N, 3))
    point_cloud[:, 0] = range_fwd_dist
    point_cloud[:, 1] = cx - range_u
    point_cloud[:, 1] /= fx
    point_cloud[:, 1] *= range_fwd_dist
    point_cloud[:, 1] *= np.cos(range_alpha)
    point_cloud[:, 1] /= np.cos(range_angles)
    point_cloud[:, 2] = np.float32(0.0)

    return point_cloud

def project_pixels_torch(
    v_sub, u_sub, subsample_step, bot_pitch,
    cx, cy, fx, fy, cam_pitch, cam_height, 
    min_angle=0.01, max_range=7.5
):
    N = v_sub.numel()
    if N == 0:
        return torch.empty((0, 3), device=v_sub.device)

    v_full = v_sub * subsample_step
    u_full = u_sub * subsample_step

    alpha = torch.atan((v_full - cy) / fy)
    angles = alpha + cam_pitch - bot_pitch

    ground_mask = angles >= min_angle
    valid_idx = torch.where(ground_mask)[0]
    valid_N = valid_idx.numel()
    if valid_N == 0:
        return torch.empty((0, 3), device=v_sub.device)

    valid_alpha = alpha[valid_idx]
    valid_angles = angles[valid_idx]
    valid_u = u_full[valid_idx]

    fwd_dist = cam_height / torch.tan(valid_angles)
    range_mask = (fwd_dist > 0.0) & (fwd_dist <= max_range)
    range_idx = torch.where(range_mask)[0]
    range_N = range_idx.numel()
    if range_N == 0:
        return torch.empty((0, 3), device=v_sub.device)

    range_u = valid_u[range_idx]
    range_alpha = valid_alpha[range_idx]
    range_angles = valid_angles[range_idx]
    range_fwd_dist = fwd_dist[range_idx]

    point_cloud = torch.zeros((range_N, 3), device=v_sub.device)
    point_cloud[:, 0] = range_fwd_dist
    point_cloud[:, 1] = cx - range_u
    point_cloud[:, 1] /= fx
    point_cloud[:, 1] *= range_fwd_dist
    point_cloud[:, 1] *= torch.cos(range_alpha)
    point_cloud[:, 1] /= torch.cos(range_angles)
    point_cloud[:, 2] = 0.0

    return point_cloud