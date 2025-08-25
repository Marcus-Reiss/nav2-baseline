from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_baseline',
            executable='collision_monitor',
            name='collision_monitor',
            output='screen',
            parameters=[
                {'input_topic': '/bumper_states'},
                {'debounce_time': 1.5}
            ]
        )
    ])
