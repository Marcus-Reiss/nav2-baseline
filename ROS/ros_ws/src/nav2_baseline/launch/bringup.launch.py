from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('nav2_baseline')
    turtlebot3_pkg = get_package_share_directory('turtlebot3_bringup')

    default_world = 'corridor_3x10_dynamic_v0.world'
    world_name = LaunchConfiguration('world_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            name='world_name',
            default_value=default_world
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'gazebo_custom_world.launch.py')
            ),
            launch_arguments={
                'world_name': world_name
            }.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'spawn_robot.launch.py')
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_pkg, 'launch', 'robot.launch.py')
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'nav2_navigation.launch.py')
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'collision_monitor.launch.py')
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'goal_pose.launch.py')
            )
        ),
    ])
