from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ppo_training'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'model_bumper'), glob('model_bumper/*.config')),
        (os.path.join('share', package_name, 'model_bumper'), glob('model_bumper/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros_ws',
    maintainer_email='ros_ws@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = ppo_training.training_node:main',
            'robot_spawner = ppo_training.robot_spawner:main',
            'goal_spawner = ppo_training.goal_spawner:main',
            'evaluate = ppo_training.evaluator:main',
            'test_reward = ppo_training.test_reward:main',
            'rl_infer_node = ppo_training.rl_infer_node:main'
        ],
    },
)
