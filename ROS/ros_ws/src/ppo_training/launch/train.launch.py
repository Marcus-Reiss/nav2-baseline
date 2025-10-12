import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("ppo_training")
    gazebo_share = get_package_share_directory("gazebo_ros")

    # Stage argument: 1 = empty, 2 = static obstacles, 3 = static + dynamic
    stage_arg = DeclareLaunchArgument(
        name='stage',
        default_value='1',
        description='Training stage: 1 = empty, 2 = static obstacles, 3 = static + dynamic'
    )
    stage = LaunchConfiguration('stage')

    # Seleciona o arquivo .world conforme stage (usando PythonExpression)
    world_expr = PythonExpression([
        "'", os.path.join(pkg_share, 'worlds', 'empty.world'), "' if '", stage, "' == '1' else '",
        os.path.join(pkg_share, 'worlds', 'static_v1.world'), "' if '", stage, "' == '2' else '",
        os.path.join(pkg_share, 'worlds', 'corridor_3x10_static.world'), "'"
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_expr}.items()
    )

    # Spawn do turtlebot3 burger sem imu
    tb3_file = os.path.join(pkg_share, 'model_bumper', 'model.sdf')

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3',
            '-file', tb3_file,
            '-x', '4.0',   # 4.0, 0.0
            '-y', '0.0',  # -4.0, 0.0
            '-z', '0.01'
        ],
        output='screen'
    )

    # robot_spawner = Node(
    #     package='ppo_training',
    #     executable='robot_spawner',
    #     name='robot_spawner_node',
    #     output='screen'
    # )

    # Goal spawner node (serviço spawn_new_goal)
    goal_spawner = Node(
        package='ppo_training',
        executable='goal_spawner',
        name='goal_spawner_node',
        output='screen'
    )

    # Trainer node (training_node), passa --stage arg para que o trainer saiba o stage
    trainer = Node(
        package='ppo_training',
        executable='train',
        name='ppo_training_node',
        output='screen',
        arguments=['--stage', stage]
    )

    return LaunchDescription([
        stage_arg,
        gazebo,
        spawn_robot,
        goal_spawner,
        trainer,
    ])
