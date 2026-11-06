from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("parking"), "config", "slot_detector.yaml")
    return LaunchDescription([
        Node(package="parking", executable="slot_detector", name="slot_detector",
             parameters=[cfg, {"use_sim_time": True}], output="screen"),
    ])
