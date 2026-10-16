# Starts lane_change_server + lane_path_mux ONLY. Vision, Nav2 and path_bridge are launched separately.
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lane_change'), 'config', 'lane_change.yaml')
    sim = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='lane_change', executable='lane_change_server', name='lane_change_server',
             output='screen', parameters=[cfg, {'use_sim_time': sim}]),
        Node(package='lane_change', executable='lane_path_mux', name='lane_path_mux',
             output='screen', parameters=[cfg, {'use_sim_time': sim}]),
    ])