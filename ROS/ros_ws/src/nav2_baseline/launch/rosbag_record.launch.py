from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', 'nav2_benchmark_bag',
                '/odom', '/tf', '/tf_static', '/scan',
                '/amcl_pose', '/initialpose', '/goal_pose', 
                '/cmd_vel', '/map', '/plan', '/local_plan',
                '/navigate_to_pose/_action/goal',
                '/navigate_to_pose/_action/feedback',
                '/navigate_to_pose/_action/result',
                '/collision_event', '/collision_count',
            ],
            output='screen'
        )
    ])
