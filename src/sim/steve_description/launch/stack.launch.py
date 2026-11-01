import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # --- Launch Configurations ---
    use_sim_time = LaunchConfiguration('use_sim_time')

    # --- Package Directories ---
    steve_dir = get_package_share_directory('steve_description')

    # --- Launch Arguments ---
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation clock if true'
    )

    # --- Perception Nodes ---
    vision_node = Node(
        package='vision_py', 
        executable='vision', 
        name='vision',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'profile_timing': False
        }]
    )

    lidar_node = Node(
        package='lidar_perception_cpp',
        executable='lidar_processor',
        name='lidar_processor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )

    centerline_node = Node(
        package='path_generation_cpp',
        executable='centerline_node',
        name='centerline_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    mapping_node = Node(
        package='mapping_cpp',
        executable='mapping_node',
        name='mapping_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,

        vision_node,
        lidar_node,
        centerline_node,
        mapping_node
    ])

