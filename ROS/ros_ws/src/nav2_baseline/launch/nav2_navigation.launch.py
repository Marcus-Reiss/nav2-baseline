from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_pkg = get_package_share_directory('nav2_baseline')
    nav2_params_file = os.path.join(
        nav2_pkg, 'config', 'nav2_params.yaml'
    )
    map_file = os.path.join(
        nav2_pkg, 'maps', 'map_maze.yaml'
    )

    # # Map server node
    # map_server = Node(
    #     package='nav2_map_server',
    #     executable='map_server',
    #     name='map_server',
    #     output='screen',
    #     parameters=[{'yaml_filename': map_file}]
    # )

    # # AMCL node
    # amcl = Node(
    #     package='nav2_amcl',
    #     executable='amcl',
    #     name='amcl',
    #     output='screen',
    #     parameters=[nav2_params_file]
    # )

    # Nav2 bringup (bt_navigator, controller, planner, etc.)
    nav2_core = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map': map_file,
                'params_file': nav2_params_file
            }.items()
        )

    return LaunchDescription([
        # amcl,
        # map_server,
        nav2_core
    ])
