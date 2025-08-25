from launch import LaunchDescription
from launch_ros.actions import Node
from math import pi
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    model_file = os.path.join(
        get_package_share_directory('nav2_baseline'),
        'model_bumper', 'model.sdf')
    
    # Original model.sdf:
    # '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf'

    # Turtlebot3 node
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3',
            '-file', model_file,
            '-x', '4.5',   # 4.5; 9.0
            '-y', '0.0',  # -4.5; 0.0
            '-z', '0.01',
            '-Y', f'{pi}'
        ],
        output='screen'
    )

    return LaunchDescription([
        spawn_robot
    ])