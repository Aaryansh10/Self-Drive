import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lane_change'), 'config', 'lane_change.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true when the simulator publishes /clock'),
        Node(package='lane_change', executable='lane_change_server', name='lane_change_server',
             output='screen',
             parameters=[cfg, {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}]),
    ])

