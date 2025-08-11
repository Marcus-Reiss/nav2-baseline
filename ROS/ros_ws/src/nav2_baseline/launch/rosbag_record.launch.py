from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', 'nav2_benchmark_bag',
                '/odom', '/tf', '/collision', '/tf_static', '/scan',
                '/amcl_pose', '/goal_pose', '/cmd_vel'
            ],
            output='screen'
        )
    ])
