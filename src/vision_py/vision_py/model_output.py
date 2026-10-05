#!/usr/bin/env python3
"""
ROS 2 node: run HydraNet on a camera stream, overlay detections + segmentation
on the camera frame, and record the merged result to a video file.

Overlay:
  * Segmentation: argmax over the logits. Class 0 is treated as background
    (not drawn). Classes 1/2/3 are drawn as red / green / blue translucent masks.
  * Detection: boxes + "class score" labels drawn on the same frame.

Example:
  ros2 run <your_pkg> hydranet_recorder_node --ros-args \
      -p image_topic:=/camera/image_raw \
      -p weights:=/path/to/hydranet.pt \
      -p output_path:=/tmp/hydranet_out.mp4 \
      -p fps:=30.0 \
      -p class_names:="['car','truck','pedestrian','cyclist','stop_sign']"

Requires: rclpy, cv_bridge, opencv-python, torch, torchvision.
The HydraNet package (the folder holding model.py, heads.py, backbone.py,
neck.py, modules.py, utils.py) must be importable, e.g. as `hydranet`.
"""

import os
import time

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Change this import if your package has a different name.
from vision_py.architectures.hydra_net.model import build_hydranet


# BGR colours for OpenCV: seg class 1 -> red, 2 -> green, 3 -> blue
SEG_COLOURS_BGR = {
    1: (0, 0, 255),   # red
    2: (0, 255, 0),   # green
    3: (255, 0, 0),   # blue
}


class HydraNetRecorder(Node):
    def __init__(self):
        super().__init__("hydranet_recorder")

        # ---------------- parameters ----------------
        self.declare_parameter("image_topic", "/zed_node/monocamera/image_raw")
        self.declare_parameter("output_topic", "/hydranet/overlay")  # "" to disable
        self.declare_parameter("output_path", "hydranet_output.mp4")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("fourcc", "mp4v")
        self.declare_parameter("weights", "/home/aaryansh/dev/Self-Drive/src/vision_py/vision_py/architectures/hydra_net/sim_finetuned.pt")
        self.declare_parameter("variant", "base")
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("input_height", 544)
        self.declare_parameter("input_width", 960)
        self.declare_parameter("num_seg_classes", 4)
        self.declare_parameter("num_obj_classes", 5)
        self.declare_parameter("class_names", [""])  # optional detection names
        self.declare_parameter("score_thresh", 0.35)
        self.declare_parameter("nms_iou", 0.5)
        self.declare_parameter("max_dets", 100)
        self.declare_parameter("mask_alpha", 0.45)
        self.declare_parameter("use_fp16", False)
        # Input normalisation: default is plain RGB / 255. Set to match training.
        self.declare_parameter("mean", [0.485, 0.456, 0.406])
        self.declare_parameter("std", [0.229, 0.224, 0.225])

        p = self.get_parameter
        self.input_hw = (int(p("input_height").value), int(p("input_width").value))
        self.num_obj = int(p("num_obj_classes").value)
        self.score_thresh = float(p("score_thresh").value)
        self.nms_iou = float(p("nms_iou").value)
        self.max_dets = int(p("max_dets").value)
        self.alpha = float(p("mask_alpha").value)
        self.fps = float(p("fps").value)
        self.fourcc = str(p("fourcc").value)
        self.output_path = str(p("output_path").value)
        self.device = torch.device(str(p("device").value))
        self.use_fp16 = bool(p("use_fp16").value) and self.device.type == "cuda"

        names = [n for n in p("class_names").value if n]
        self.class_names = names if len(names) == self.num_obj else [f"cls{i}" for i in range(self.num_obj)]

        mean = np.array(p("mean").value, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(p("std").value, dtype=np.float32).reshape(1, 1, 3)
        self.mean, self.std = mean, std

        # ---------------- model ----------------
        self.model = build_hydranet(
            str(p("variant").value),
            input_size=self.input_hw,
            num_seg_classes=int(p("num_seg_classes").value),
            num_obj_classes=self.num_obj,
        )
        self._load_weights(str(p("weights").value))
        self.model.to(self.device).eval()
        if self.use_fp16:
            self.model.half()

        # ---------------- IO ----------------
        self.bridge = CvBridge()
        self.writer = None
        self.frame_count = 0
        self.t_infer = 0.0

        out_dir = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(out_dir, exist_ok=True)

        self.sub = self.create_subscription(
            Image, str(p("image_topic").value), self.image_cb, qos_profile_sensor_data
        )
        out_topic = str(p("output_topic").value)
        self.pub = self.create_publisher(Image, out_topic, 1) if out_topic else None

        self.get_logger().info(
            f"Recording '{p('image_topic').value}' -> {self.output_path} "
            f"(device={self.device}, fp16={self.use_fp16})"
        )

    # ------------------------------------------------------------------
    def _load_weights(self, path):
        if not path:
            self.get_logger().warn("No weights given: running with RANDOM weights.")
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
 
        def is_state_dict(d):
            return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())
 
        state = None
        if isinstance(ckpt, torch.nn.Module):
            state = ckpt.state_dict()
        elif is_state_dict(ckpt):
            state = ckpt
        elif isinstance(ckpt, dict):
            self.get_logger().info(f"Checkpoint top-level keys: {list(ckpt.keys())}")
            preferred = ("ema", "ema_state_dict", "model_state_dict", "state_dict",
                         "model_state", "model", "net", "weights")
            for key in list(preferred) + list(ckpt.keys()):
                v = ckpt.get(key)
                if isinstance(v, torch.nn.Module):
                    v = v.state_dict()
                if is_state_dict(v):
                    self.get_logger().info(f"Using checkpoint entry '{key}'")
                    state = v
                    break
        if state is None:
            raise RuntimeError("Could not find a state_dict in the checkpoint")
 
        # strip DataParallel / DDP prefix if present
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
 
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        msg = f"Loaded weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})"
        if missing or unexpected:
            self.get_logger().error(msg + f" e.g. missing={missing[:3]} unexpected={unexpected[:3]}")
        else:
            self.get_logger().info(msg)


    # ------------------------------------------------------------------
    def preprocess(self, frame_bgr):
        h, w = self.input_hw
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
        x = x.to(self.device)
        return x.half() if self.use_fp16 else x

    @torch.no_grad()
    def postprocess_detections(self, det, scale_xy, frame_wh):
        """FCOS-style: score = sqrt(cls_prob * centerness_prob), then class-wise NMS.
        Returns boxes (n,4) in frame pixels, scores (n,), labels (n,)."""
        cls = det["cls_logits"][0].float().sigmoid()          # (N, C)
        ctr = det["centerness"][0].float().sigmoid()          # (N, 1)
        boxes = det["boxes"][0].float()                       # (N, 4) xyxy, input px

        scores = (cls * ctr).sqrt()                           # (N, C)
        flat = scores.flatten()
        keep = flat > self.score_thresh
        if keep.sum() == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

        idx = keep.nonzero(as_tuple=True)[0]
        if idx.numel() > 1000:  # pre-NMS top-k
            idx = idx[flat[idx].topk(1000).indices]

        anchor_idx = idx // self.num_obj
        labels = idx % self.num_obj
        sel_boxes, sel_scores = boxes[anchor_idx], flat[idx]

        k = batched_nms(sel_boxes, sel_scores, labels, self.nms_iou)[: self.max_dets]
        sel_boxes, sel_scores, labels = sel_boxes[k], sel_scores[k], labels[k]

        sx, sy = scale_xy
        fw, fh = frame_wh
        sel_boxes[:, [0, 2]] = (sel_boxes[:, [0, 2]] * sx).clamp(0, fw - 1)
        sel_boxes[:, [1, 3]] = (sel_boxes[:, [1, 3]] * sy).clamp(0, fh - 1)
        return sel_boxes.cpu().numpy(), sel_scores.cpu().numpy(), labels.cpu().numpy()

    # ------------------------------------------------------------------
    def overlay(self, frame, seg_logits, boxes, scores, labels):
        fh, fw = frame.shape[:2]
        out = frame.copy()

        # --- segmentation: class map -> coloured translucent overlay ---
        cls_map = seg_logits[0].argmax(0).to(torch.uint8).cpu().numpy()
        cls_map = cv2.resize(cls_map, (fw, fh), interpolation=cv2.INTER_NEAREST)

        colour_layer = np.zeros_like(frame)
        for cid, bgr in SEG_COLOURS_BGR.items():
            colour_layer[cls_map == cid] = bgr
        mask = cls_map > 0  # background left untouched
        blended = cv2.addWeighted(frame, 1.0 - self.alpha, colour_layer, self.alpha, 0)
        out[mask] = blended[mask]

        # --- detections on top ---
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, (0, 255, 255), 2)  # yellow so it doesn't clash with mask colours
            text = f"{self.class_names[int(l)]} {s:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(p1[1], th + bl + 2)
            cv2.rectangle(out, (p1[0], ty - th - bl - 2), (p1[0] + tw + 2, ty), (0, 255, 255), -1)
            cv2.putText(out, text, (p1[0] + 1, ty - bl - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        fh, fw = frame.shape[:2]

        t0 = time.perf_counter()
        x = self.preprocess(frame)
        with torch.no_grad():
            out = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        sx, sy = fw / self.input_hw[1], fh / self.input_hw[0]
        boxes, scores, labels = self.postprocess_detections(out["detection"], (sx, sy), (fw, fh))
        result = self.overlay(frame, out["lane_logits"], boxes, scores, labels)
        self.t_infer += time.perf_counter() - t0

        # Lazily create the writer once we know the real frame size.
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                self.output_path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (fw, fh)
            )
            if not self.writer.isOpened():
                self.get_logger().error(f"Could not open VideoWriter for {self.output_path}")
                raise RuntimeError("VideoWriter failed to open")
        self.writer.write(result)

        if self.pub is not None:
            out_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            out_msg.header = msg.header
            self.pub.publish(out_msg)

        self.frame_count += 1
        if self.frame_count % 100 == 0:
            avg = self.t_infer / self.frame_count
            self.get_logger().info(f"{self.frame_count} frames, avg {avg*1000:.1f} ms/frame")

    # ------------------------------------------------------------------
    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.get_logger().info(f"Saved {self.frame_count} frames to {self.output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = HydraNetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()