import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # --- Launch Configurations ---
    use_sim_time = LaunchConfiguration('use_sim_time')

    # --- Package Directories ---
    steve_dir = get_package_share_directory('steve_description')

    # --- Configuration File Paths ---
    local_ekf_config = os.path.join(steve_dir, 'config', 'local_ekf.yaml')
    global_ekf_config = os.path.join(steve_dir, 'config', 'global_ekf.yaml')
    nav2_params_file = os.path.join(steve_dir, 'config', 'nav2_params.yaml')

    # --- Launch Arguments ---
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # --- Perception Nodes ---
    vision_node = Node(
        package='vision_perception_py',
        executable='vision',
        name='vision',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'camera_height': 1.606,        
            'camera_pitch_deg': 22.92,    
            'subsample_step': 1,
            'max_projection_range': 7.5,
            'centerline_bin_size': 0.5,
            'half_lane_width': 1.5,
            'min_bin_points': 1,
            'min_valid_bins_per_side': 5   
        }]
    )

    path_relay_node = Node(
        package='vision_perception_py',
        executable='path_relay',
        name='path_relay',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    gps_navigator_node = Node(
        package='utils_py',
        executable='gps_navigator',
        name='gps_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # --- Localization Nodes ---
    local_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[local_ekf_config, {'use_sim_time': use_sim_time}],
        remappings=[
            ('odometry/filtered', 'odometry/local') 
        ]
    )

    global_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_map',
        output='screen',
        parameters=[global_ekf_config, {'use_sim_time': use_sim_time}],
        remappings=[
            ('odometry/filtered', 'odometry/global') 
        ]
    )

    sensor_repub_node = Node(
        package='utils_py',
        executable='sensor_repub', 
        name='sensor_covariance_republisher',
        output='screen'
    )

    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[global_ekf_config, {'use_sim_time': use_sim_time}],
        remappings=[
            ('imu/data', '/imu/with_covariance'),           
            ('gps/fix', '/gps/fix'),        
            ('odometry/filtered', 'odometry/global'), 
            ('odometry/gps', '/odometry/gps') 
        ]
    )

    # --- Optimized Nav2 Component Container ---
    nav2_container = ComposableNodeContainer(
        name='nav2_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded to handle concurrent execution safely
        arguments=['--ros-args', '--params-file', nav2_params_file],
        composable_node_descriptions=[
            ComposableNode(
                package='nav2_controller',
                plugin='nav2_controller::ControllerServer',
                name='controller_server',
                parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='nav2_behaviors',
                plugin='behavior_server::BehaviorServer',
                name='behavior_server',
                parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='nav2_velocity_smoother',
                plugin='nav2_velocity_smoother::VelocitySmoother',
                name='velocity_smoother',
                parameters=[nav2_params_file, {'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='lidar_perception_cpp',
                plugin='LidarProcessor',
                name='lidar_processor',
                parameters=[{'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
        parameters=[nav2_params_file, {'use_sim_time': use_sim_time}]
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[nav2_params_file, {'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        
        # Perception
        vision_node,
        path_relay_node,
        gps_navigator_node,
        
        # Localization
        local_ekf_node,
        global_ekf_node,
        sensor_repub_node,
        navsat_transform_node,
        
        # Navigation (Optimized Container + Manager)
        nav2_container,
        lifecycle_manager_node
    ])