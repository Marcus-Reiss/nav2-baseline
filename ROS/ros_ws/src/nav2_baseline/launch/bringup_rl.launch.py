from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('nav2_baseline')

    rl_bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'rl_bridge.launch.py')
            )
        )

    almost_bringup = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'almost_bringup.launch.py')
            )
        )
    
    goal_pose = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'goal_pose.launch.py')
            )
        )

    return LaunchDescription([
        rl_bridge,
        TimerAction(period=5.0, actions=[almost_bringup]),
        TimerAction(period=12.0, actions=[goal_pose])
    ])
