from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    goal_pub_node = Node(
        package='nav2_baseline',
        executable='goal_pub_node',
        name='goal_pub_node',
        output='screen'
    )

    return LaunchDescription([
        goal_pub_node
    ])
