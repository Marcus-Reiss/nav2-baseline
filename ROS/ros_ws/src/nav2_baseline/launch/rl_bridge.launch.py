from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ppo_models_dir = get_package_share_directory("ppo_training")
    ppo_models_path = os.path.join(
        ppo_models_dir, "trained_models", "ppo_model"
    )

    rl_infer_node = Node(
        package='ppo_training',
        executable='rl_infer_node',
        name='rl_infer',
        output='screen',
        parameters=[{
            'model_path': ppo_models_path,
            'model_type': 'sb3'
        }]
    )

    return LaunchDescription([
        rl_infer_node
    ])
