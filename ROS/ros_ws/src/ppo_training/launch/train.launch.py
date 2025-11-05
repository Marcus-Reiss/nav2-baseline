import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import TextSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("ppo_training")
    gazebo_share = get_package_share_directory("gazebo_ros")

    # Paths from nav2_baseline
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_pkg = get_package_share_directory('nav2_baseline')
    nav2_params_file = os.path.join(nav2_pkg, 'config', 'nav2_params.yaml')
    map_file = os.path.join(nav2_pkg, 'maps', 'empty_bkp.yaml')  # map_corr_3x10

    # Tb3 pkg
    turtlebot3_pkg = get_package_share_directory('turtlebot3_bringup')

    # Stage argument: 1 = empty, 2 = static obstacles, 3 = static + dynamic
    stage_arg = DeclareLaunchArgument(
        name='stage',
        default_value='1',
        description='Training stage: 1 = empty, 2 = static obstacles, 3 = static + dynamic'
    )
    stage = LaunchConfiguration('stage')

    # Seleciona o arquivo .world conforme stage (usando PythonExpression)
    world_expr = PythonExpression([
        "'", os.path.join(nav2_pkg, 'worlds', 'empty_bkp.world'), "' if '", stage, "' == '1' else '",
        os.path.join(pkg_share, 'worlds', 'static_v1.world'), "' if '", stage, "' == '2' else '",
        os.path.join(pkg_share, 'worlds', 'corridor_3x10_static.world'), "'"
    ])

    use_sim_time_arg = DeclareLaunchArgument(
    'use_sim_time',
    default_value='true',
    description='Use simulated clock (Gazebo /clock topic)'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_expr,
            'use_sim_time': use_sim_time
        }.items()
    )

    robot_spawner_node = Node(
        package='ppo_training',
        executable='robot_spawner',
        name='robot_spawner',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    tb3_features = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_pkg, 'launch', 'robot.launch.py')
        )
    )

    nav2_custom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'bringup_custom.launch.py')
        ),
        launch_arguments={
            'map': map_file,
            'params_file': nav2_params_file,
            'use_sim_time': use_sim_time
        }.items()
    )

    # Goal spawner node (serviço spawn_new_goal)
    goal_spawner = Node(
        package='ppo_training',
        executable='goal_spawner',
        name='goal_spawner_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Trainer node (training_node), passa --stage arg para que o trainer saiba o stage
    trainer = Node(
        package='ppo_training',
        executable='train',
        name='ppo_training_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['--stage', stage]
    )

    return LaunchDescription([
        stage_arg,
        use_sim_time_arg,
        gazebo,
        robot_spawner_node,
        tb3_features,
        nav2_custom,
        goal_spawner,
        trainer
    ])
