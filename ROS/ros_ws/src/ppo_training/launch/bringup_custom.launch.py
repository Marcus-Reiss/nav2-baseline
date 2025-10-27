import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml, ReplaceString


def generate_launch_description():
    bringup_dir = get_package_share_directory('nav2_bringup')
    launch_dir = os.path.join(bringup_dir, 'launch')

    # Custom navigation file (with cmd_vel remap)
    ppo_training_dir = get_package_share_directory('ppo_training')
    navigation_custom_path = os.path.join(ppo_training_dir, 'launch', 'navigation_custom.launch.py')

    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    slam = LaunchConfiguration('slam')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    remappings = [
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
        # ### PATCH PPO: redirecionar todos cmd_vel internos do Nav2
        ('/cmd_vel', '/nav2_cmd_vel'),
    ]

    param_substitutions = {
        'use_sim_time': use_sim_time,
        'yaml_filename': map_yaml_file
    }

    params_file = ReplaceString(
        source_file=params_file,
        replacements={'<robot_namespace>': ('/', namespace)},
        condition=IfCondition(use_namespace)
    )

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True),
        allow_substs=True
    )

    stdout_linebuf_envvar = SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument('namespace', default_value='')
    declare_use_namespace_cmd = DeclareLaunchArgument('use_namespace', default_value='false')
    declare_slam_cmd = DeclareLaunchArgument('slam', default_value='False')
    declare_map_yaml_cmd = DeclareLaunchArgument('map')
    declare_use_sim_time_cmd = DeclareLaunchArgument('use_sim_time', default_value='false')
    declare_params_file_cmd = DeclareLaunchArgument('params_file', default_value=os.path.join(bringup_dir, 'params', 'nav2_params.yaml'))
    declare_autostart_cmd = DeclareLaunchArgument('autostart', default_value='true')
    declare_use_composition_cmd = DeclareLaunchArgument('use_composition', default_value='True')
    declare_use_respawn_cmd = DeclareLaunchArgument('use_respawn', default_value='False')
    declare_log_level_cmd = DeclareLaunchArgument('log_level', default_value='info')

    bringup_cmd_group = GroupAction([
        PushRosNamespace(condition=IfCondition(use_namespace), namespace=namespace),

        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[configured_params, {'autostart': autostart}],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings,
            output='screen'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'slam_launch.py')),
            condition=IfCondition(slam),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'use_respawn': use_respawn,
                'params_file': params_file
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'localization_launch.py')),
            condition=IfCondition(PythonExpression(['not ', slam])),
            launch_arguments={
                'namespace': namespace,
                'map': map_yaml_file,
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'params_file': params_file,
                'use_composition': use_composition,
                'use_respawn': use_respawn,
                'container_name': 'nav2_container'
            }.items()
        ),

        # ### PATCH PPO: use navigation_custom.launch.py (instead of nav2's original)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(navigation_custom_path),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'params_file': params_file,
                'use_composition': use_composition,
                'use_respawn': use_respawn,
                'container_name': 'nav2_container'
            }.items()
        ),
    ])

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_namespace_cmd)
    ld.add_action(declare_slam_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(bringup_cmd_group)
    return ld
