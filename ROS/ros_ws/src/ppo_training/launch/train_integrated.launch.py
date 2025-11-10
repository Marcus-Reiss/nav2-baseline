# ppo_training/launch/train_integrated.launch.py (corrigido)
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # --- package paths ---
    pkg_share = get_package_share_directory("ppo_training")
    gazebo_share = get_package_share_directory("gazebo_ros")
    nav2_baseline_pkg = get_package_share_directory("nav2_baseline")
    nav2_bringup_pkg = get_package_share_directory("nav2_bringup")

    # --- args ---
    stage_arg = DeclareLaunchArgument(
        name='stage',
        default_value='1',
        description='Training stage: 1 = empty, 2 = static obstacles, 3 = static + dynamic'
    )
    stage = LaunchConfiguration('stage')

    world_expr = PythonExpression([
        "'", os.path.join(nav2_baseline_pkg, 'worlds', 'empty_bkp.world'), "' if '", stage, "' == '1' else '",
        os.path.join(pkg_share, 'worlds', 'static_v1.world'), "' if '", stage, "' == '2' else '",
        os.path.join(pkg_share, 'worlds', 'corridor_3x10_static.world'), "'"
    ])

    # --- gazebo include ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_expr,
            'use_sim_time': 'true'
        }.items()
    )

    # --- spawn robot ---
    tb3_file = os.path.join(pkg_share, 'model_bumper', 'model.sdf')
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'turtlebot3', '-file', tb3_file, '-x', '0.5', '-y', '0.5', '-z', '0.01'],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # --- goal spawner ---
    goal_spawner = Node(
        package='ppo_training',
        executable='goal_spawner',
        name='goal_spawner_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # --- Nav2 file paths ---
    nav2_params_file = os.path.join(nav2_baseline_pkg, 'config', 'nav2_params.yaml')
    custom_bt_xml_file = os.path.join(nav2_baseline_pkg, 'config', 'plan_only.xml')
    map_file = os.path.join(nav2_baseline_pkg, 'maps', 'empty_bkp.yaml')

    # --- include bringup (do nav2_bringup) ---
    # autostart false: vamos controlar ativação explicitamente com o lifecycle manager (evita corrida)
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_pkg, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_file,
            'use_sim_time': 'true',
            'autostart': 'true',
            'params_file': nav2_params_file,
            'default_bt_xml_filename': custom_bt_xml_file
        }.items()
    )

    # --- trainer node ---
    trainer_node = Node(
        package='ppo_training',
        executable='train',
        name='ppo_trainer_node',
        output='screen',
        arguments=['--stage', stage],
        parameters=[{'use_sim_time': True}]
    )

    delayed_nav2 = TimerAction(
        period=8.0,
        actions=[nav2_bringup]
    )

    # --- assemble launch description ---
    return LaunchDescription([
        stage_arg,
        gazebo,
        spawn_robot,
        goal_spawner,
        delayed_nav2,
        trainer_node
    ])
