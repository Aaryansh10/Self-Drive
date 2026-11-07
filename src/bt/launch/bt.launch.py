# Starts the BT, stop server and intersection server. Vision, Nav2, path_bridge and lane_system remain separate.
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('bt')
    sim = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('stop_action'), 'launch', 'stop_server.launch.py')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
        Node(package='intersection_cpp', executable='intersection_server', output='screen',
             parameters=[{'use_sim_time': sim}]),
        Node(package='bt', executable='bt_executor', name='bt', output='screen',
             parameters=[os.path.join(share, 'config', 'bt.yaml'),
                         {'tree_file': os.path.join(share, 'trees', 'main.xml'), 'use_sim_time': sim,
                         'detections_topic': '/tracked_obstacles'}]),
    ])