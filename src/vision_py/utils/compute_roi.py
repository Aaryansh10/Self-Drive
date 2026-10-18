"""Compute LaneGrid ROI params from the same model used in ipm.py.

Fill in the CONFIG block with your real values, run, and paste the output into lanegrid::Params.
"""
import numpy as np

# ------------------------------ CONFIG (edit these) ------------------------------
IMG_W = 960                 # sim camera width (masks are native image size, no resize)
V_TOP = 0
V_BOTTOM = 543              # height 544 - 1
CX, CY = 480.5, 272.5       # Gazebo default; confirm with logged CameraInfo
FX, FY = 343.3, 343.3      # from URDF hfov 1.9 rad; confirm with logged CameraInfo
CAM_PITCH = 0.4              # vision.py default
CAM_HEIGHT = 1.606           # vision.py default
CAM_X_OFFSET = 0.02         # MEASURE: camera ahead of link_base
MAX_RANGE = 6.0             # farthest range you choose to TRUST (<= projection max_range of 8)
MIN_ANGLE = 0.01
BOT_PITCH_TOL = 0.03        # rad, expected bot pitch variation (use +/- this)
MARGIN_NEAR = 0.05          # m, shrink near edge
MARGIN_FAR = 0.30           # m, shrink far edge (IPM is noisiest here)
MARGIN_SIDE = 0.05          # m, shrink lateral edges
# ---------------------------------------------------------------------------------


def row_extent(v, bot_pitch):
    """Forward distance (camera-relative) and left/right lateral extents for image row v."""
    alpha = np.arctan((v - CY) / FY)
    angle = alpha + CAM_PITCH + bot_pitch
    if angle >= MIN_ANGLE and CAM_HEIGHT / np.tan(angle) <= MAX_RANGE:
        x = CAM_HEIGHT / np.tan(angle)
    else:
        # Row sees ground past max_range (or the horizon): the footprint ends where fwd == MAX_RANGE,
        # so use the row/angle that actually lands there for the lateral extent.
        x = MAX_RANGE
        angle = np.arctan(CAM_HEIGHT / x)
        alpha = angle - CAM_PITCH - bot_pitch
    scale = x * np.cos(alpha) / np.cos(angle)
    y_left = (CX - 0.0) / FX * scale           # u = 0
    y_right = (CX - (IMG_W - 1)) / FX * scale  # u = W-1 (negative)
    return x, y_left, y_right


near_x, far_x, near_y, far_y = [], [], [], []
for dp in (-BOT_PITCH_TOL, 0.0, BOT_PITCH_TOL):     # pitch changes the footprint; stay conservative
    xn, yl, yr = row_extent(V_BOTTOM, dp)
    near_x.append(xn); near_y.append(min(abs(yl), abs(yr)))
    xf, yl, yr = row_extent(V_TOP, dp)
    far_x.append(xf); far_y.append(min(abs(yl), abs(yr)))

roi_x_min = max(near_x) + CAM_X_OFFSET + MARGIN_NEAR       # nearest ground seen at ANY pitch
roi_x_max = min(far_x) + CAM_X_OFFSET - MARGIN_FAR         # farthest ground seen at ALL pitches
half_near = min(near_y) - MARGIN_SIDE
half_far = min(far_y) - MARGIN_SIDE

print("// paste into lanegrid::Params")
print(f"roi_x_min       = {roi_x_min:.2f}f;")
print(f"roi_x_max       = {roi_x_max:.2f}f;")
print(f"roi_half_y_near = {half_near:.2f}f;")
print(f"roi_half_y_far  = {half_far:.2f}f;")

if roi_x_max <= roi_x_min:
    print("WARNING: far edge <= near edge. Check pitch sign, intrinsics scale and row range.")