import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

import hsv_masking


class Vision(Node):
    def __init__(self):
        super().__init__('vision')

        self.cam_info_sub_ = self.create_subscription(
            CameraInfo,
            '/zed_node/monocamera/camera_info',
            self.cam_info_cb,
            qos_overriding_options=qos_profile_sensor_data
        )
        self.img_sub_ = self.create_subscription(
            Image,
            '/zed_node/monocamera/image_raw',
            self.img_cb, 
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

        self.cv_bridge = CvBridge()

        self.get_logger().info(f"Vision node started successfully.")



    def img_cb(self, msg):
        if self.cam_info_recieved == False:
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        mask = hsv_masking.generate_mask(img)







        

    def cam_info_cb(self, msg):
        if not self.cam_info_recieved:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.fy = msg.k[4]
            self.cy = msg.k[5]
            self.K = msg.k
            self.cam_info_recieved = True
            self.get_logger().info(f"Camera intrinsics received. fx:{self.fx:.2f}, fy:{self.fy:.2f}, cx:{self.cx:.2f}, cy:{self.cy:.2f}")
            self.destroy_subscription(self.camera_info_sub_)

