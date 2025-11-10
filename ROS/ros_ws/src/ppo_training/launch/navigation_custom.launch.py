# navigation_custom.launch.py (planner + bt_navigator)
# ROS 2 Humble – versão compatível com PPO training
# VERSÃO CORRIGIDA FINAL
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
import time # Importa 'time' que estava faltando no seu original


def generate_launch_description():
    bringup_dir = get_package_share_directory('nav2_bringup')

    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    map_yaml = LaunchConfiguration('map')                # <-- novo argumento 'map'
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    container_name_full = (namespace, '/', container_name)
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    # ======================================================
    # CORREÇÃO 1: Remove 'local_costmap' da lista de nós
    # ======================================================
    lifecycle_nodes = [
        'map_server',
        'global_costmap',
        # 'local_costmap', # <-- REMOVIDO (não é lançado)
        'planner_server',
        'bt_navigator'
    ]

    remappings = [
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
        ('/cmd_vel', '/nav2_cmd_vel'),
        ('/cmd_vel_smoothed', '/nav2_smoothed_cmd_vel'),
    ]

    # Substituições (o 'autostart' do seu params.yaml (false) será usado)
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'autostart': autostart,
        'yaml_filename': map_yaml,
    }

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites=param_substitutions,
            convert_types=True,
        ),
        allow_substs=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    # argumentos padrão (agora com 'map')
    args = [
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(bringup_dir, 'params', 'nav2_params.yaml'),
        ),
        DeclareLaunchArgument('map', default_value=''),             # <-- novo
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('use_composition', default_value='False'),
        DeclareLaunchArgument('container_name', default_value='nav2_container'),
        DeclareLaunchArgument('use_respawn', default_value='False'),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    # modo standalone
    load_nodes = GroupAction(
        actions=[
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                parameters=[configured_params],
                remappings=remappings,
            ),
            
            # ======================================================
            # CORREÇÃO 2: Adiciona o nó 'global_costmap' que faltava
            # ======================================================
            Node(
                package='nav2_costmap_2d',
                executable='nav2_costmap_2d', # O executável que você encontrou
                name='global_costmap',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings + [('map', '/map'), ('scan', '/scan')]
            ),
            # ======================================================
            
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            ),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings,
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    # ======================================================
                    # CORREÇÃO 3: Força 'autostart: False' no manager
                    # para impedir conflito com a função '_activate_nodes'
                    # ======================================================
                    {'autostart': False},
                    {'node_names': lifecycle_nodes},
                ],
            ),
        ]
    )

    # publicador TF estático map->odom (mantido)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_odom_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # Função de ativação manual (do seu arquivo original)
    def _activate_nodes(context, *args, **kwargs):
        import rclpy
        from rclpy.node import Node as RclNode
        
        # Só ativa se o launch argument 'autostart' for 'true'
        if LaunchConfiguration('autostart').perform(context) == 'false':
            return

        if not rclpy.ok():
            rclpy.init()
        tmp = RclNode('planner_bt_activator')
        client = tmp.create_client(ChangeState, '/lifecycle_manager_navigation/change_state')

        tmp.get_logger().info('Ativador manual esperando pelo serviço do lifecycle_manager...')
        
        if not client.wait_for_service(timeout_sec=20.0):
            tmp.get_logger().error('SERVIÇO DO LIFECYCLE MANAGER NÃO ENCONTRADO (timeout).')
            tmp.destroy_node()
            return
            
        tmp.get_logger().info('Serviço do manager encontrado.')

        def call_transition(tid, name):
            req = ChangeState.Request()
            trans = Transition()
            trans.id = tid
            req.transition = trans
            
            tmp.get_logger().info(f'Solicitando transição: {name}...')
            fut = client.call_async(req)
            rclpy.spin_until_future_complete(tmp, fut, timeout_sec=15.0)
            if fut.done():
                try:
                    result = fut.result()
                    if result.success:
                         tmp.get_logger().info(f'Transição {name} BEM SUCEDIDA.')
                    else:
                         tmp.get_logger().error(f'Transição {name} FALHOU (success=false).')
                except Exception as e:
                    tmp.get_logger().error(f'Transição {name} FALHOU com exceção: {e}')
            else:
                tmp.get_logger().error(f'Transição {name} FALHOU (timeout).')

        # primeiro configure, depois activate (aguarda entre transições)
        call_transition(Transition.TRANSITION_CONFIGURE, 'CONFIGURE')
        time.sleep(1.0)
        call_transition(Transition.TRANSITION_ACTIVATE, 'ACTIVATE')

        tmp.get_logger().info('Ativação manual concluída.')
        tmp.destroy_node()

    activate = OpaqueFunction(function=_activate_nodes)

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    for a in args:
        ld.add_action(a)
    ld.add_action(load_nodes)
    ld.add_action(static_tf)
    ld.add_action(activate) # <-- Sua função de ativação manual está aqui
    return ld
