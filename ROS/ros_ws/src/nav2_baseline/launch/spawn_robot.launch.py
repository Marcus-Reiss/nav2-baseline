from launch import LaunchDescription
from launch_ros.actions import Node
from math import pi


def generate_launch_description():

    # Turtlebot3 node
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3',
            '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf',
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