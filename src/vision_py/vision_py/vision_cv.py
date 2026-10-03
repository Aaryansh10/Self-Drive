import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, PointField
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

from vision_py import hsv_masking, ipm

class Vision(Node):
    def __init__(self):
        super().__init__('vision')
        self.declare_parameter('subsample_step', 1)
        self.declare_parameter('cam_pitch', 0.4)
        self.declare_parameter('cam_height', 1.606)

        self.cam_info_sub_ = self.create_subscription(
            CameraInfo,
            '/zed_node/monocamera/camera_info',
            self.cam_info_cb,
            qos_profile=qos_profile_sensor_data
        )
        self.img_sub_ = self.create_subscription(
            Image,
            '/zed_node/monocamera/image_raw',
            self.img_cb, 
            qos_profile=qos_profile_sensor_data
        )
        self.imu_sub_ = self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            qos_profile=qos_profile_sensor_data
        )

        self.pc_white_pub_ = self.create_publisher(
            PointCloud2,
            '/pc/white_lane',
            qos_profile=qos_profile_sensor_data
        )
        self.pc_yellow_pub_ = self.create_publisher(
            PointCloud2,
            '/pc/yellow_lane',
            qos_profile=qos_profile_sensor_data
        )
        self.pc_stopline_pub_ = self.create_publisher(
            PointCloud2,
            '/pc/stop_line',
            qos_profile=qos_profile_sensor_data
        )

        self.cam_info_recieved = False
        self.pitch_bot = 0.0
        self.header = Header()
        self.header.frame_id = 'link_base'

        self.cv_bridge = CvBridge()

        self.subsample_step = self.get_parameter('subsample_step').value
        self.cam_pitch = self.get_parameter('cam_pitch').value
        self.cam_height = self.get_parameter('cam_height').value

        self.get_logger().info(f"Vision node started successfully.")

    def img_cb(self, msg):
        if self.cam_info_recieved == False:
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        mask = hsv_masking.generate_mask(img)
        mask_whitelane = mask[0]
        mask_yellow = mask[1]
        mask_stopline = mask[2]

        mask_whitelane[:] = mask_whitelane[::self.subsample_step]
        mask_yellow[:] = mask_yellow[::self.subsample_step]
        mask_stopline[:] = mask_stopline[::self.subsample_step]

        def leading_edge_extraction(mask):
            H, W = mask.shape
            flipped_mask = mask[::-1, :]
            has_edge = np.any(flipped_mask, axis=0)
            idx = np.argmax(flipped_mask, axis=0)
            v_sub = (H - 1) - idx
            u_sub = np.arange(W)
            return v_sub[has_edge], u_sub[has_edge]

        v_sub_whitelane, u_sub_whitelane = leading_edge_extraction(mask_whitelane)
        pc_whitelane = ipm.project_pixels_np(
            v_sub_whitelane, u_sub_whitelane, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

        v_sub_yellow, u_sub_yellow = np.nonzero(mask_yellow)
        pc_yellow = ipm.project_pixels_np(
            v_sub_yellow, u_sub_yellow, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

        v_sub_stopline, u_sub_stopline = leading_edge_extraction(mask_stopline)
        pc_stopline = ipm.project_pixels_np(
            v_sub_stopline, u_sub_stopline, self.subsample_step, self.pitch_bot,
            self.cx, self.cy, self.fx, self.fy, self.cam_pitch, self.cam_height
        )

        self.pc_white_pub_.publish(self.np_to_pc(pc_whitelane)) 
        self.pc_yellow_pub_.publish(self.np_to_pc(pc_yellow))
        self.pc_stopline_pub_.publish(self.np_to_pc(pc_stopline))

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

    def np_to_pc(self, pc_np):
        pc_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg = PointCloud2()
        msg.header = self.header
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