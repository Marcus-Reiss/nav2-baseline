from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('nav2_baseline')
    gazebo_pkg = get_package_share_directory('gazebo_ros')

    default_world = 'corridor_3x10_dynamic_v0.world'
    world_name = LaunchConfiguration('world_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            name='world_name',
            default_value=default_world,
            description='Name of the .world file in nav2_baseline/worlds'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                gazebo_pkg, 'launch', 'gazebo.launch.py'
            )),
            launch_arguments={
                'world': PathJoinSubstitution([pkg, 'worlds', world_name])
            }.items()
        ),
    ])
