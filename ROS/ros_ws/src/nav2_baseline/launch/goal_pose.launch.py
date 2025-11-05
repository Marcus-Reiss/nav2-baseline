from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_baseline',
            executable='goal_pub_node',
            name='goal_pub_node',
            output='screen',
            parameters=[{
                # Ajuste se quiser
                'target_x': -1.5,
                'target_y': -1.5,
                'target_yaw': 0.0,
                'frame_id': 'map',
                'once': True,
                'check_period': 0.5,
            }]
        )
    ])
