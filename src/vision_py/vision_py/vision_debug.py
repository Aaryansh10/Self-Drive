"""
vision_debug.py

Drop-in replacement for the `vision` node. Same inputs, same outputs (pc_*, detections,
left/right centerline Paths), plus:

  * /cl_debug/<stage>        MarkerArray per pipeline stage (see centerline_debug.STAGE_TOPICS)
  * /cl_debug/seg_overlay    Image: segmentation overlay (raw vs clean_mask) + leading-edge points
  * every CenterlineGenerator hyper-parameter exposed as a live ROS parameter `cl.<name>`

Run INSTEAD of the normal vision node (it registers as node `vision`):
    ros2 run vision_py vision_debug
Tune live:
    ros2 param set /vision cl.centerline_ema_alpha 0.6
    ros2 param set /vision cl.max_gap 1.5          # floats must be given as floats (1.5, 9.0 not 9)
"""
import time

import cv2
import numpy as np
import torch

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from vision_interfaces.msg import DetectionArray, Detection2D

from vision_py.vision import Vision
from vision_py.ipm import project_pixels_torch
from vision_py.model_helpers import preprocess, postprocess_detections, clean_mask
from vision_py.centerline_debug import DebugCenterlineGenerator, build_markers, STAGE_TOPICS

# Defaults copied from CenterlineGenerator.__init__ (unused / disabled args omitted).
CL_DEFAULTS = dict(
    max_gap=1.0, bin_width=0.5, max_bad_frame_streak=20, max_range=8.0,
    nominal_road_width=7.4, min_road_width=6.6, max_road_width=8.2, min_bin_points=1,
    max_width_update=0.5, min_offset_fraction=0.5, yellow_gate_fraction=0.25,
    path_spacing=0.1, path_start=0.2, final_smooth_window=25, near_field_range=3.5,
    min_near_pts=2, width_stale_decay_frames=30, width_stale_decay_alpha=0.05,
    sg_polyorder=2, track_ema_alpha=0.35, width_ema_alpha=0.15, centerline_ema_alpha=0.4,
    width_update_max_x=4.0, resample_step=0.1, position_window_m=1.0, tangent_window_m=2.5,
    far_near_field_range=4.0, lane_change_bezier=True, lane_change_factor=2.5,
    lane_change_min_len=3.0, bezier_handle=0.4,
)
# These change tensor shapes / initial memory, so changing one rebuilds the generator
# (tracked memory is reset). Everything else is applied live without losing state.
STRUCTURAL = {'max_range', 'bin_width', 'path_start', 'path_spacing', 'nominal_road_width'}
INT_PARAMS = {'max_bad_frame_streak', 'min_bin_points', 'final_smooth_window', 'min_near_pts',
              'width_stale_decay_frames', 'sg_polyorder'}


class VisionDebug(Vision):
    def __init__(self):
        super().__init__()

        self.declare_parameter("debug_enable", True)
        self.declare_parameter("debug_overlay", True)
        self.declare_parameter("debug_prefix", "/cl_debug")
        for k, v in CL_DEFAULTS.items():
            self.declare_parameter(f"cl.{k}", v)

        self.debug_enable = self.get_parameter("debug_enable").value
        self.debug_overlay = self.get_parameter("debug_overlay").value
        prefix = self.get_parameter("debug_prefix").value.rstrip('/')

        # Replace the generator that Vision.__init__ created with the instrumented one.
        self.centerline_generator = self._build_generator({})
        self.add_on_set_parameters_callback(self._on_params)

        marker_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.stage_pubs = {
            name: self.create_publisher(MarkerArray, f"{prefix}/{name}", marker_qos)
            for name in STAGE_TOPICS
        }
        self.overlay_pub = self.create_publisher(Image, f"{prefix}/seg_overlay", qos_profile_sensor_data)
        self.get_logger().info(
            f"vision_debug ready. Debug topics under {prefix}/ ; live params: cl.* ; fixed frame: {self.ground_frame}"
        )

    # --- live parameter handling ---

    def _build_generator(self, overrides):
        kw = {}
        for k in CL_DEFAULTS:
            v = overrides.get(k, self.get_parameter(f"cl.{k}").value)
            kw[k] = int(v) if k in INT_PARAMS else v
        return DebugCenterlineGenerator(device=self.device, **kw)

    def _on_params(self, params):
        changes = {p.name[3:]: p.value for p in params if p.name.startswith("cl.")}
        for p in params:
            if p.name == "debug_enable":
                self.debug_enable = p.value
            elif p.name == "debug_overlay":
                self.debug_overlay = p.value
        if not changes:
            return SetParametersResult(successful=True)
        try:
            if STRUCTURAL & changes.keys():
                self.centerline_generator = self._build_generator(changes)
                self.get_logger().warn(
                    f"Rebuilt CenterlineGenerator for {sorted(STRUCTURAL & changes.keys())}: tracked memory reset."
                )
            else:
                g = self.centerline_generator
                for k, v in changes.items():
                    setattr(g, k, int(v) if k in INT_PARAMS else v)
                g.refresh_derived()
                self.get_logger().info(f"Updated live: {changes}")
        except Exception as e:  # keep the node alive on a bad value
            return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    # --- helpers ---

    @staticmethod
    def leading_edge(mask):
        # Same as the closure in Vision.img_cb: lowest (nearest) mask pixel in every image column.
        H, W = mask.shape
        flipped = torch.flip(mask, dims=[0])
        has_edge = flipped.any(dim=0)
        idx = flipped.to(torch.int8).argmax(dim=0)
        v_sub = (H - 1) - idx
        u_sub = torch.arange(W, device=mask.device)
        return v_sub[has_edge], u_sub[has_edge]

    def _project(self, v, u):
        return project_pixels_torch(
            v, u, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

    def make_overlay(self, img, raw, clean, edges):
        H, W = raw[0].shape
        base = cv2.resize(img, (W, H))
        ov = base.copy()
        colors = [(255, 200, 80), (0, 230, 255), (60, 60, 255)]   # BGR: white-lane, yellow, stopline
        for r, c, col in zip(raw, clean, colors):
            r = r.cpu().numpy().astype(bool)
            c = c.cpu().numpy().astype(bool)
            ov[c] = col
            ov[r & ~c] = (255, 0, 255)                 # removed by clean_mask
        out = cv2.addWeighted(ov, self.mask_alpha, base, 1.0 - self.mask_alpha, 0)
        step = self.subsample_step
        for (v, u), col in zip(edges, ((255, 255, 0), (0, 255, 0), (0, 0, 255))):
            vv = np.clip(v.cpu().numpy() * step, 0, H - 2)
            uu = np.clip(u.cpu().numpy() * step, 0, W - 2)
            for dv in (0, 1):
                for du in (0, 1):
                    out[vv + dv, uu + du] = col
        cv2.putText(out, "blue=white lane  yellow=yellow lane  red=stopline  magenta=removed by clean_mask  dots=leading edge",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # --- main callback ---

    def img_cb(self, msg):
        if not self.cam_info_recieved:
            return
        t0 = time.perf_counter()

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.ground_frame
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # ---------- network ----------
        x = preprocess(img, device=self.device, use_fp16=self.use_fp16)
        with torch.no_grad():
            out = self.model(x)
        lane_logits, detections = out["lane_logits"], out["detection"]
        model_boxes, frame_boxes, scores, labels = postprocess_detections(
            detections, (1.0, 1.0), (self.input_width, self.input_height),
            self.score_thresh, self.nms_iou, self.max_detections
        )
        t1 = time.perf_counter()

        # ---------- masks -> leading edge -> ground-plane points ----------
        masks = lane_logits.argmax(1)[0]
        raw = [(masks == c) for c in (1, 2, 3)]
        clean = [clean_mask(r) for r in raw]
        s = self.subsample_step
        edges = [self.leading_edge(c[::s, ::s]) for c in clean]
        pc_white, pc_yellow, pc_stop = (self._project(v, u) for v, u in edges)
        t2 = time.perf_counter()

        self.pc_whitelane_pub_.publish(self.np_to_pc(pc_white.cpu().numpy(), header))
        self.pc_yellowlane_pub_.publish(self.np_to_pc(pc_yellow.cpu().numpy(), header))
        self.pc_stopline_pub_.publish(self.np_to_pc(pc_stop.cpu().numpy(), header))

        # ---------- centerline ----------
        gen = self.centerline_generator
        left_xy, right_xy = gen.generate_centerlines(pc_white, pc_yellow)
        t3 = time.perf_counter()
        left_a = gen.anchor_to_origin(left_xy)
        right_a = gen.anchor_to_origin(right_xy)
        self.left_centerline_pub_.publish(self.xy_to_path(left_a, header))
        self.right_centerline_pub_.publish(self.xy_to_path(right_a, header))

        arr = DetectionArray()
        arr.header = header
        for box, sc, lb in zip(frame_boxes, scores, labels):
            det = Detection2D()
            det.class_id = int(lb)
            det.class_name = self.class_names[int(lb)]
            det.score = float(sc)
            det.x1, det.y1, det.x2, det.y2 = map(float, box)
            arr.detections.append(det)
        self.detections_pub_.publish(arr)

        # ---------- debug outputs (never allowed to break the real pipeline) ----------
        if not self.debug_enable:
            return
        try:
            d = gen.dbg
            d['n_pts'] = (int(pc_white.shape[0]), int(pc_yellow.shape[0]), int(pc_stop.shape[0]))
            d['final'] = {'left': left_a, 'right': right_a}
            d['timing'] = {'net': 1e3 * (t1 - t0), 'mask+ipm': 1e3 * (t2 - t1), 'centerline': 1e3 * (t3 - t2)}
            for name, marr in build_markers(d, gen, header).items():
                self.stage_pubs[name].publish(marr)
            if self.debug_overlay:
                ov = self.make_overlay(img, raw, clean, edges)
                ov_msg = self.bridge.cv2_to_imgmsg(ov, encoding="bgr8")
                ov_msg.header = msg.header
                self.overlay_pub.publish(ov_msg)
        except Exception as e:
            self.get_logger().error(f"debug publishing failed: {e!r}", throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = VisionDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()