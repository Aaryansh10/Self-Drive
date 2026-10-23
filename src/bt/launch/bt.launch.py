# Starts the behavior tree executor only. Vision, Nav2, path_bridge and lane_system are launched separately.
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('bt')
    sim = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='bt', executable='bt_executor', name='bt', output='screen',
             parameters=[os.path.join(share, 'config', 'bt.yaml'),
                         {'tree_file': os.path.join(share, 'trees', 'main.xml'), 'use_sim_time': sim}]),
    ])