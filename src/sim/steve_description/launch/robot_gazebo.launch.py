# Build Workspace to see changes in Launch Files output
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from scripts import GazeboRosPaths
def generate_launch_description():
    package_share_dir = get_package_share_directory("steve_description")
    urdf = os.path.join(package_share_dir, "urdf", "steve.urdf")
    with open(urdf, "r") as f:
        robot_description = f.read()
    # gazebo_ros_package = get_package_share_directory('gazebo_ros')
    # gazebo_launch_file = os.path.join(gazebo_ros_package, 'launch', 'gazebo.launch.py')
    # meshes = package_share_dir+'/meshes'
    #os.environ['ROBOT_DESCRIPTION_MESHES'] = meshes
    return LaunchDescription([
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(gazebo_launch_file),
        # ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=["-entity","steve","-file",urdf,"-x","0","-y","0","-z","0.02","-unpause"]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,  
            }]
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': True}] 
        ),
    ])