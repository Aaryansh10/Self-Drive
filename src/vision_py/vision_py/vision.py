import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, ReliabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, PointField
from vision_interfaces.msg import DetectionArray, Detection2D
from std_msgs.msg import Header
from cv_bridge import CvBridge

from vision_py.architectures.hydra_net.model import build_hydranet
from vision_py.ipm import project_pixels_np
from vision_py.model_helpers import load_weights, preprocess, postprocess_detections, clean_mask_np
from vision_py.thinning import zhang_suen_thinning, order_skeleton_pixels, enable_fused_thinning


class StageProfiler:
    """Per-stage wall-clock profiler. Calls torch.cuda.synchronize() at every mark so that GPU work is
    attributed to the stage that launched it (otherwise it lands on whichever line syncs next).
    Disabled => every method is a no-op, so there is no overhead and no extra syncs."""

    def __init__(self, enabled, period, logger):
        self.enabled, self.period, self.logger = enabled, max(1, period), logger
        self.sums, self.maxs, self.frames = {}, {}, 0
        self.last_start = None
        self.period_sum = 0.0
        self.t = 0.0

    def start(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.t = time.perf_counter()
        if self.last_start is not None:
            self.period_sum += self.t - self.last_start
        self.last_start = self.t
        self.t_start = self.t

    def mark(self, name):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        dt = (now - self.t) * 1e3
        self.sums[name] = self.sums.get(name, 0.0) + dt
        self.maxs[name] = max(self.maxs.get(name, 0.0), dt)
        self.t = now

    def end(self):
        if not self.enabled:
            return
        total = (time.perf_counter() - self.t_start) * 1e3
        self.sums["TOTAL"] = self.sums.get("TOTAL", 0.0) + total
        self.maxs["TOTAL"] = max(self.maxs.get("TOTAL", 0.0), total)
        self.frames += 1
        if self.frames >= self.period:
            n = self.frames
            parts = " | ".join(f"{k} {v / n:6.2f}" for k, v in self.sums.items())
            worst = " | ".join(f"{k} {v:6.2f}" for k, v in self.maxs.items())
            hz = (n - 1) / self.period_sum if self.period_sum > 0 else float("nan")
            self.logger.info(f"[profile] avg ms over {n} frames, callback rate {hz:.1f} Hz\n"
                             f"  avg: {parts}\n  max: {worst}")
            self.sums, self.maxs, self.frames, self.period_sum = {}, {}, 0, 0.0


def leading_edge_extraction(mask):
    """mask: (H, W) bool numpy. For each column with any foreground, the lowest (nearest) pixel."""
    H, W = mask.shape
    flipped = mask[::-1]
    has_edge = flipped.any(axis=0)
    v = (H - 1) - flipped.argmax(axis=0)
    u = np.arange(W)
    return v[has_edge].astype(np.int64), u[has_edge].astype(np.int64)


class Vision(Node):
    def __init__(self):
        super().__init__('vision')

        # --- Parameters ---
        self.declare_parameter("image_topic", "/zed_node/monocamera/image_raw")
        self.declare_parameter("cam_info_topic", "/zed_node/monocamera/camera_info")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("pc_whitelane_topic", "/pc_whitelane")
        self.declare_parameter("pc_yellowlane_topic", "/pc_yellowlane")
        self.declare_parameter("pc_stopline_topic", "/pc_stopline")
        self.declare_parameter("detections_topic", "/detections")

        self.declare_parameter("subsample_step", 1)
        self.declare_parameter("cam_pitch", 0.4)
        self.declare_parameter("cam_height", 1.606)
        self.declare_parameter("max_range", 8.0)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'architectures', 'hydra_net', 'sim_finetuned.pt')
        self.declare_parameter("weights", weights_path)
        self.declare_parameter("model_variant", "base")
        self.declare_parameter("use_fp16", True)         # was False; biggest single win on the model stage
        self.declare_parameter("channels_last", True)    # tensor-core friendly layout (helps fp16 convs)
        self.declare_parameter("input_height", 544)
        self.declare_parameter("input_width", 960)
        self.declare_parameter("num_seg_classes", 4)
        self.declare_parameter("num_obj_classes", 5)
        self.declare_parameter("class_names", ["barrel", "tyre", "pothole", "sign", "pedestrian"])  # optional detection names
        self.declare_parameter("score_thresh", 0.35)
        self.declare_parameter("nms_iou", 0.5)
        self.declare_parameter("max_detections", 50)
        self.declare_parameter("mask_alpha", 0.45)
        self.declare_parameter("fused_thinning", True)   # torch.compile fused thinning (slow ~10-40 s startup, then ~10x faster)
        self.declare_parameter("cpu_threads", 3)         # parallel mask cleaning (OpenCV releases the GIL)
        self.declare_parameter("profile_timing", True)   # per-stage timing log (adds GPU syncs; turn off in production)
        self.declare_parameter("profile_period", 50)     # frames per log line

        self.declare_parameter("ground_frame", "link_base")

        self.image_topic = self.get_parameter("image_topic").value
        self.cam_info_topic = self.get_parameter("cam_info_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.pc_whitelane_topic = self.get_parameter("pc_whitelane_topic").value
        self.pc_yellowlane_topic = self.get_parameter("pc_yellowlane_topic").value
        self.pc_stopline_topic = self.get_parameter("pc_stopline_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.subsample_step = self.get_parameter("subsample_step").value
        self.cam_pitch = self.get_parameter("cam_pitch").value
        self.cam_height = self.get_parameter("cam_height").value
        self.max_range = self.get_parameter("max_range").value
        self.ground_frame = self.get_parameter("ground_frame").value

        self.weights = self.get_parameter("weights").value
        self.model_variant = self.get_parameter("model_variant").value
        self.use_fp16 = self.get_parameter("use_fp16").value
        self.channels_last = self.get_parameter("channels_last").value
        self.input_height = self.get_parameter("input_height").value
        self.input_width = self.get_parameter("input_width").value
        self.num_seg_classes = self.get_parameter("num_seg_classes").value
        self.num_obj_classes = self.get_parameter("num_obj_classes").value
        self.class_names = self.get_parameter("class_names").value
        self.score_thresh = self.get_parameter("score_thresh").value
        self.nms_iou = self.get_parameter("nms_iou").value
        self.max_detections = self.get_parameter("max_detections").value
        self.mask_alpha = self.get_parameter("mask_alpha").value
        self.prof = StageProfiler(self.get_parameter("profile_timing").value,
                                  self.get_parameter("profile_period").value, self.get_logger())
        n_threads = int(self.get_parameter("cpu_threads").value)
        self.pool = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None

        # --- HydraNet model ---
        torch.backends.cudnn.benchmark = True            # autotune conv algos (input shape is fixed)
        self.model = build_hydranet(self.model_variant)
        load_weights(self.model, self.weights)
        self.device = "cuda"
        self.model.to(self.device).eval()
        if self.use_fp16:
            self.model.half()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self._warmup()

        # --- Subscribers ---
        # depth=1: always process the NEWEST frame. With the default sensor-data depth (5) and a
        # callback slower than the camera, the node worked through stale frames and added up to
        # ~5 frame-times of pure queueing latency on top of the compute time.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.img_sub_ = self.create_subscription(Image, self.image_topic, self.img_cb, qos_profile=image_qos)
        self.cam_info_sub_ = self.create_subscription(
            CameraInfo, self.cam_info_topic, self.cam_info_cb, qos_profile=qos_profile_sensor_data
        )
        self.imu_sub_ = self.create_subscription(
            Imu, self.imu_topic, self.imu_cb, qos_profile=qos_profile_sensor_data
        )

        # --- Publishers ---
        self.pc_whitelane_pub_ = self.create_publisher(
            PointCloud2, self.pc_whitelane_topic, qos_profile=qos_profile_sensor_data
        )
        self.pc_yellowlane_pub_ = self.create_publisher(
            PointCloud2, self.pc_yellowlane_topic, qos_profile=qos_profile_sensor_data
        )
        self.pc_stopline_pub_ = self.create_publisher(
            PointCloud2, self.pc_stopline_topic, qos_profile=qos_profile_sensor_data
        )
        detections_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.detections_pub_ = self.create_publisher(DetectionArray, self.detections_topic, detections_qos)

        # --- Data members ---
        self.bridge = CvBridge()
        self.cam_info_recieved = False
        self.pitch_bot = 0.0

        self.get_logger().info("Vision node started successfully.")

    def _warmup(self):
        """Run the model and thinning once at startup so cuDNN autotune, CUDA context setup and
        lazy kernel loading don't land on the first real frames (that was the 578 ms max)."""
        H, W = self.input_height, self.input_width
        dummy = torch.zeros(1, 3, H, W, device=self.device,
                            dtype=torch.float16 if self.use_fp16 else torch.float32)
        if self.channels_last:
            dummy = dummy.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            for _ in range(3):
                self.model(dummy)
        s = self.subsample_step
        hw = ((H + s - 1) // s, (W + s - 1) // s)
        if self.get_parameter("fused_thinning").value:
            self.get_logger().info("Compiling fused thinning kernels (one-off, may take up to a minute)...")
            enable_fused_thinning(self.device, hw, self.get_logger().info)
        zhang_suen_thinning(torch.zeros(2, *hw, dtype=torch.bool, device=self.device))
        torch.cuda.synchronize()

    # --- Callbacks ---

    def img_cb(self, msg):
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.ground_frame

        if not self.cam_info_recieved:
            return
        self.prof.start()
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.prof.mark("bridge")

        x = preprocess(img, device=self.device, use_fp16=self.use_fp16, channels_last=self.channels_last)
        self.prof.mark("preprocess")
        with torch.inference_mode():
            out = self.model(x)
        self.prof.mark("model")

        lane_logits, detections = out["lane_logits"], out["detection"]
        frame_boxes, scores, labels = postprocess_detections(
            detections, (1.0, 1.0), (self.input_width, self.input_height),
            self.score_thresh, self.nms_iou, self.max_detections
        )
        self.prof.mark("post_det")

        # One 0.5 MB transfer of the class map; everything mask-related below is CPU/OpenCV.
        cls = lane_logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
        raw = [cls == 1, cls == 2, cls == 3]
        if self.pool is not None:
            mask_whitelane, mask_yellow, mask_stopline = self.pool.map(clean_mask_np, raw)
        else:
            mask_whitelane, mask_yellow, mask_stopline = map(clean_mask_np, raw)

        s = self.subsample_step
        mask_whitelane = mask_whitelane[::s, ::s]
        mask_yellow = mask_yellow[::s, ::s]
        mask_stopline = mask_stopline[::s, ::s]
        self.prof.mark("masks")

        # Thin both lane masks in one GPU batch, then ONE transfer back for both skeletons.
        skel_in = torch.from_numpy(np.stack([mask_whitelane, mask_yellow])).to(self.device)
        skel = zhang_suen_thinning(skel_in).cpu().numpy()
        self.prof.mark("thinning")

        v_w, u_w = order_skeleton_pixels(skel[0])
        v_y, u_y = order_skeleton_pixels(skel[1])
        v_s, u_s = leading_edge_extraction(mask_stopline)
        self.prof.mark("order_skeleton")

        # Point clouds are tiny (a few thousand points): projecting on the CPU with numpy avoids
        # uploading, ~30 small kernels x3, and downloading again.
        proj = dict(
            subsample_step=s, bot_pitch=self.pitch_bot, cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy,
            cam_pitch=self.cam_pitch, cam_height=self.cam_height, max_range=self.max_range,
        )
        pc_whitelane = project_pixels_np(v_w, u_w, **proj)
        pc_yellow = project_pixels_np(v_y, u_y, **proj)
        pc_stopline = project_pixels_np(v_s, u_s, **proj)
        self.prof.mark("project(ipm)")

        self.pc_whitelane_pub_.publish(self.np_to_pc(pc_whitelane, header))
        self.pc_yellowlane_pub_.publish(self.np_to_pc(pc_yellow, header))
        self.pc_stopline_pub_.publish(self.np_to_pc(pc_stopline, header))
        self.prof.mark("publish_pc")

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
        self.prof.mark("publish_det")
        self.prof.end()

    def imu_cb(self, msg):
        orientation = msg.orientation
        sinp = 2.0 * (orientation.w * orientation.y - orientation.z * orientation.x)
        if abs(sinp) >= 1.0:
            self.pitch_bot = np.copysign(np.pi / 2.0, sinp)
        else:
            self.pitch_bot = np.arcsin(sinp)

    def cam_info_cb(self, msg):
        if not self.cam_info_recieved:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.K = msg.k
            self.cam_info_recieved = True
            self.get_logger().info(f"Camera intrinsics received. fx:{self.fx:.2f}, fy:{self.fy:.2f}, cx:{self.cx:.2f}, cy:{self.cy:.2f}")
            self.destroy_subscription(self.cam_info_sub_)

    # --- Helpers ---

    def np_to_pc(self, pc_np, header):
        pc_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = pc_np.shape[0]
        msg.fields = pc_fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * pc_np.shape[0]
        msg.is_dense = True
        msg.data = pc_np.astype(np.float32, copy=False).tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = Vision()
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