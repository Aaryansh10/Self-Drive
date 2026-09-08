import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'steve_description'

    local_ekf_config = os.path.join(
        get_package_share_directory(pkg_name), 
        'config', 
        'local_ekf.yaml'
    )

    sensor_repub_node = Node(
        package='utils_py',
        executable='sensor_repub', 
        name='sensor_covariance_republisher',
        output='screen'
    )

    local_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[local_ekf_config, {'use_sim_time': True}],
        remappings=[
            ('odometry/filtered', 'odometry/local') 
        ]
    )

    return LaunchDescription([
        sensor_repub_node,
        local_ekf_node,
    ])