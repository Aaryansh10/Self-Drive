import os
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
from vision_py.ipm import project_pixels_torch
from vision_py.model_helpers import load_weights, preprocess, postprocess_detections, clean_mask
from vision_py.thinning import zhang_suen_thinning, order_skeleton_pixels

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

        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'architectures', 'hydra_net', 'sim_finetuned.pt')
        self.declare_parameter("weights", weights_path)
        self.declare_parameter("model_variant", "base")
        self.declare_parameter("use_fp16", False)
        self.declare_parameter("input_height", 544)
        self.declare_parameter("input_width", 960)
        self.declare_parameter("num_seg_classes", 4)
        self.declare_parameter("num_obj_classes", 5)
        self.declare_parameter("class_names", ["barrel", "tyre", "pothole", "sign", "pedestrian"])  # optional detection names
        self.declare_parameter("score_thresh", 0.35)
        self.declare_parameter("nms_iou", 0.5)
        self.declare_parameter("max_detections", 50)
        self.declare_parameter("mask_alpha", 0.45)

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
        self.ground_frame = self.get_parameter("ground_frame").value

        self.weights = self.get_parameter("weights").value
        self.model_variant = self.get_parameter("model_variant").value
        self.use_fp16 = self.get_parameter("use_fp16").value
        self.input_height = self.get_parameter("input_height").value
        self.input_width = self.get_parameter("input_width").value
        self.num_seg_classes = self.get_parameter("num_seg_classes").value
        self.num_obj_classes = self.get_parameter("num_obj_classes").value
        self.class_names = self.get_parameter("class_names").value
        self.score_thresh = self.get_parameter("score_thresh").value
        self.nms_iou = self.get_parameter("nms_iou").value
        self.max_detections = self.get_parameter("max_detections").value
        self.mask_alpha = self.get_parameter("mask_alpha").value

        # --- HydraNet model ---
        self.model = build_hydranet(self.model_variant)
        load_weights(self.model, self.weights)
        self.device = "cuda"
        self.model.to(self.device).eval()
        if self.use_fp16:
            self.model.half()

        # --- Subscribers ---
        self.img_sub_ = self.create_subscription(
            Image,
            self.image_topic,
            self.img_cb,
            qos_profile=qos_profile_sensor_data
        )
        self.cam_info_sub_= self.create_subscription(
            CameraInfo,
            self.cam_info_topic,
            self.cam_info_cb,
            qos_profile=qos_profile_sensor_data
        )
        self.imu_sub_ = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_cb,
            qos_profile=qos_profile_sensor_data
        )

        # --- Publishers ---
        self.pc_whitelane_pub_ = self.create_publisher(
            PointCloud2,
            self.pc_whitelane_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.pc_yellowlane_pub_ = self.create_publisher(
            PointCloud2,
            self.pc_yellowlane_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.pc_stopline_pub_ = self.create_publisher(
            PointCloud2,
            self.pc_stopline_topic,
            qos_profile=qos_profile_sensor_data
        )
        detections_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.detections_pub_ = self.create_publisher(
            DetectionArray,
            self.detections_topic,
            detections_qos
        )

        # --- Data members ---
        self.bridge = CvBridge()
        self.mean = np.array([0.485, 0.456, 0.406] ,dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.cam_info_recieved = False
        self.pitch_bot = 0.0

        self.get_logger().info("Vision node started successfully.")

    # --- Callbacks ---

    def img_cb(self, msg):
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.ground_frame
        
        if self.cam_info_recieved == False:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        x = preprocess(img, device=self.device, use_fp16=self.use_fp16)
        with torch.no_grad():
            out = self.model(x)

        lane_logits, detections = out["lane_logits"], out["detection"]
        model_boxes, frame_boxes, scores, labels = postprocess_detections(
            detections, (1.0, 1.0), (self.input_width, self.input_height),
            self.score_thresh, self.nms_iou, self.max_detections
        )

        masks = lane_logits.argmax(1)[0]
        mask_whitelane = clean_mask((masks == 1))
        mask_yellow = clean_mask((masks == 2))
        mask_stopline = clean_mask((masks == 3))

        mask_whitelane = mask_whitelane[::self.subsample_step, ::self.subsample_step]
        mask_yellow = mask_yellow[::self.subsample_step, ::self.subsample_step]
        mask_stopline = mask_stopline[::self.subsample_step, ::self.subsample_step]

        def leading_edge_extraction(mask):
            H, W = mask.shape
            flipped_mask = torch.flip(mask, dims=[0])
            has_edge = flipped_mask.any(dim=0)
            idx = flipped_mask.to(torch.int8).argmax(dim=0)
            v_sub = (H - 1) - idx
            u_sub = torch.arange(W, device=mask.device)
            return v_sub[has_edge], u_sub[has_edge]

        skel = zhang_suen_thinning(torch.stack([mask_whitelane, mask_yellow]))
        v_sub_whitelane, u_sub_whitelane = order_skeleton_pixels(skel[0])
        v_sub_yellow, u_sub_yellow = order_skeleton_pixels(skel[1])

        pc_whitelane = project_pixels_torch(
            v_sub_whitelane, u_sub_whitelane, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

        pc_yellow = project_pixels_torch(
            v_sub_yellow, u_sub_yellow, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

        v_sub_stopline, u_sub_stopline = leading_edge_extraction(mask_stopline)
        pc_stopline = project_pixels_torch(
            v_sub_stopline, u_sub_stopline, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

        self.pc_whitelane_pub_.publish(self.np_to_pc(pc_whitelane.cpu().numpy(), header)) 
        self.pc_yellowlane_pub_.publish(self.np_to_pc(pc_yellow.cpu().numpy(), header))
        self.pc_stopline_pub_.publish(self.np_to_pc(pc_stopline.cpu().numpy(), header))

        arr = DetectionArray()
        arr.header = header
        for box, s, l in zip(frame_boxes, scores, labels):
            det = Detection2D()
            det.class_id = int(l)
            det.class_name = self.class_names[int(l)]
            det.score = float(s)
            det.x1, det.y1, det.x2, det.y2 = map(float, box)
            arr.detections.append(det)
        self.detections_pub_.publish(arr)

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
        msg.data = pc_np.astype(np.float32).tobytes()
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