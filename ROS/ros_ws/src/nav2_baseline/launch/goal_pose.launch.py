import random
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # goal_candidates = [
    #     {'x': -2.2, 'y': -2.2},
    #     {'x': 0.0, 'y': -2.2},
    #     {'x': 0.0, 'y': 2.2},
    #     {'x': -2.0, 'y': 1.5},
    #     {'x': 0.0, 'y': 0.0}
    # ]

    #random.choice(goal_candidates)['x']

    return LaunchDescription([
        Node(
            package='nav2_baseline',
            executable='goal_pub_node',
            name='goal_pub_node',
            output='screen',
            parameters=[{
                # Ajuste se quiser
                'target_x': -4.0,  # 2.75
                'target_y': 0.0,   # 2.5
                'target_yaw': 0.0,
                'frame_id': 'map',
                'once': True,
                'check_period': 0.5,
            }]
        )
    ])
