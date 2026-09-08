import math

import rclpy

from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

class SensorCovarianceRepublisher(Node):
    def __init__(self):
        super().__init__('sensor_covariance_republisher')
        
        # --- ODOMETRY SETUP ---
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom/with_covariance', qos_profile_sensor_data)
        
        # 36-value matrix. 1e9 locks the Y and Z axes to prevent sideways slip in the math.
        self.odom_cov = [
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1e9, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1e9, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1e9, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.1
        ]

        # --- IMU SETUP ---
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, '/imu/with_covariance', qos_profile_sensor_data)
        
        # 9-value matrices for the IMU
        self.imu_orientation_cov = [
            0.05, 0.0, 0.0,
            0.0, 0.05, 0.0,
            0.0, 0.0, 0.05
        ]
        
        self.imu_angular_vel_cov = [
            0.05, 0.0, 0.0,
            0.0, 0.05, 0.0,
            0.0, 0.0, 0.05
        ]
        
        self.imu_linear_accel_cov = [
            0.05, 0.0, 0.0,
            0.0, 0.05, 0.0,
            0.0, 0.0, 0.05
        ]

    def odom_callback(self, msg):
        # The ackermann plugin publishes twist.linear in the world/odom
        # frame while labeling child_frame_id as the body frame (link_base)
        # — a REP 105 violation. Rotate it by -yaw here so everything
        # downstream (this node's own consumers, the EKF, controller_server)
        # gets a true body-frame vx/vy instead of a heading-dependent one.
        q = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        wx = msg.twist.twist.linear.x
        wy = msg.twist.twist.linear.y
        msg.twist.twist.linear.x = cos_yaw * wx + sin_yaw * wy
        msg.twist.twist.linear.y = -sin_yaw * wx + cos_yaw * wy

        # Inject the matrix into Odometry
        msg.pose.covariance = self.odom_cov
        msg.twist.covariance = self.odom_cov
        self.odom_pub.publish(msg)

    def imu_callback(self, msg):
        # Inject the matrices into IMU
        msg.orientation_covariance = self.imu_orientation_cov
        msg.angular_velocity_covariance = self.imu_angular_vel_cov
        msg.linear_acceleration_covariance = self.imu_linear_accel_cov
        self.imu_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorCovarianceRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()