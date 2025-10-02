import os
import random

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty


class RobotSpawnerNode(Node):
    def __init__(self):
        super().__init__("robot_spawner_node")

        # Caminho do modelo do turtlebot3 modificado
        self.entity_dir_path = get_package_share_directory("ppo_training")

        self.entity_path = os.path.join(
            self.entity_dir_path, 'model_bumper', 'model.sdf'
        )
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = "turtlebot3"

        self.pos_candidates = [
            {"x": 4.0, "y": -4.0},
            {"x": 1.5, "y": -3.0},
            {"x": 4.0, "y": -0.5},
            {"x": 2.0, "y": 2.0},
            {"x": -2.5, "y": 1.5},
            {"x": -1.5, "y": -2.5}
        ]

        # Serviços internos do Gazebo
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')

        # Serviço exposto: chamado pelo reset() do env_wrapper
        self.spawn_robot_srv = self.create_service(
            Empty, 'spawn_new_robot', self.spawn_new_robot_callback
        )

        self.get_logger().info('Serviço spawn_new_robot disponível')

        # Primeiro spawn
        self.spawn_new_robot_callback(None, Empty.Response())

    def spawn_new_robot_callback(self, request, response):
        self.delete_entity()
        self.generate_new_robot_pose()

        pose = self.robot_pose
        self.spawn_entity(pose)

        self.get_logger().info(f'Novo robot spawnado em ({pose.position.x:.2f}, {pose.position.y:.2f})')
        return response

    def generate_new_robot_pose(self):
        pos = random.choice(self.pos_candidates)

        pose = Pose()
        pose.position.x = pos["x"]
        pose.position.y = pos["y"]
        pose.position.z = 0.01

        self.robot_pose = pose

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        if self.delete_entity_client.wait_for_service(timeout_sec=2.0):
            future = self.delete_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        else:
            self.get_logger().info("Serviço delete_entity não disponível")

    def spawn_entity(self, pose: Pose):
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.robot_namespace = ""
        req.initial_pose = pose
        req.reference_frame = "world"

        if self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
            future = self.spawn_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        else:
            self.get_logger().info("Serviço spawn_entity não disponível")

def main(args=None):
    rclpy.init(args=args)
    node = RobotSpawnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
