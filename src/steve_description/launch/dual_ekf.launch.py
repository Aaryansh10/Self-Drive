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
    
    global_ekf_config = os.path.join(
        get_package_share_directory(pkg_name), 
        'config', 
        'global_ekf.yaml'
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

    global_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_map',
        output='screen',
        parameters=[global_ekf_config, {'use_sim_time': True}],
        remappings=[
            ('odometry/filtered', 'odometry/global') 
        ]
    )

    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[global_ekf_config, {'use_sim_time': True}],
        remappings=[
            ('imu/data', '/imu'),           
            ('gps/fix', '/gps/fix'),        
            ('odometry/filtered', 'odometry/global'), 
            ('odometry/gps', '/odometry/gps') 
        ]
    )

    return LaunchDescription([
        local_ekf_node,
        global_ekf_node,
        navsat_transform_node
    ])