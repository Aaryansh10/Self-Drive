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
    mapping_dir = get_package_share_directory('mapping_cpp')
    bt_dir = get_package_share_directory('bt')

    nav2_params_file = os.path.join(mapping_dir, 'config', 'nav2_params.yaml')

    # --- Launch Arguments ---
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation clock if true'
    )

    # --- Perception Nodes ---
    vision_node = Node(
        package='vision_cpp', 
        executable='vision_node', 
        name='vision',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'onnx_path': "/home/aaryansh/dev/Self-Drive/src/vision_cpp/models/hydranet.onnx"
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

    path_mux_node = Node(
        package='lane_change',
        executable='lane_path_mux',
        name='lane_path_mux',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        remappings=[('cmd_vel', 'cmd_vel_nav')],
        parameters=[nav2_params_file, {'use_sim_time': use_sim_time}]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[nav2_params_file, {'use_sim_time': use_sim_time}]
    )

    velocity_smoother_node = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')],
        parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
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

    map_skeleton_node = Node(
        package='mapping_cpp',
        executable='map_skeleton_node',
        name='map_skeleton_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    intersection_node = Node(
        package='intersection_cpp',
        executable='intersection_server',
        name='intersection_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    path_brige_node = Node(
        package='mapping_cpp',
        executable='path_bridge',
        name='path_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    lidar_cam_fusion_node = Node(
        package='lidar_cam_fusion',
        executable='fusion_node',
        name='fusion_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    in_lane_filter_node = Node(
        package='lidar_cam_fusion',
        executable='in_lane_filter',
        name='in_lane_filter',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    obstacle_memory_node = Node(
        package='lidar_cam_fusion',
        executable='obstacle_memory',
        name='obstacle_memory',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    lane_change_server = Node(
        package='lane_change',
        executable='lane_change_server',
        name='lane_change_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    bt = Node(
        package='bt',
        executable='bt_executor',
        name='bt',
        output='screen',
        parameters=[
            os.path.join(bt_dir, 'config', 'bt.yaml'),
            {
                'tree_file': os.path.join(bt_dir, 'trees', 'main.xml'), 
                'use_sim_time': use_sim_time,
                'detections_topic': '/tracked_obstacles'
            }
        ]
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,

        vision_node,
        lidar_node,
        centerline_node,
        path_mux_node,
        mapping_node,
        map_skeleton_node,
        controller_server_node,
        lifecycle_manager_node,
        velocity_smoother_node,
        intersection_node,
        path_brige_node,
        lidar_cam_fusion_node,
        in_lane_filter_node,
        obstacle_memory_node,
        lane_change_server,
        bt

    ])

