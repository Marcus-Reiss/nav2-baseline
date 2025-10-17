import os
import random
import time

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty


# Renomeado para RobotSpawner para consistência
class RobotSpawner(Node):
    def __init__(self):
        # Nome do nó ajustado
        super().__init__("robot_spawner")

        self.entity_dir_path = get_package_share_directory("ppo_training")
        self.entity_path = os.path.join(
            self.entity_dir_path, 'model_bumper', 'model.sdf'
        )
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = "turtlebot3_burger" # Nome completo do modelo no Gazebo

        self.pos_candidates = [
            {"x": 4.0, "y": 0.0}
        ]

        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')

        # O serviço que será chamado pelo env_wrapper
        self.respawn_robot_service = self.create_service(
            Empty, 'respawn_robot', self.respawn_robot_callback
        )

        self.get_logger().info('Serviço "respawn_robot" pronto.')
        
        # Realiza o primeiro spawn para garantir que o robô exista
        # Damos um pequeno tempo para o Gazebo iniciar completamente
        time.sleep(5)
        self.respawn_robot_callback(None, Empty.Response())


    def respawn_robot_callback(self, request, response):
        # A lógica é a mesma: deleta o robô antigo, cria um novo
        self.delete_entity()
        
        # Pequeno delay para garantir que a deleção foi processada
        time.sleep(0.5)

        new_pose = self.generate_new_robot_pose()
        self.spawn_entity(new_pose)

        self.get_logger().info(f'Robô spawnado em ({new_pose.position.x:.2f}, {new_pose.position.y:.2f})')
        return response

    def generate_new_robot_pose(self):
        pos = random.choice(self.pos_candidates)
        pose = Pose()
        pose.position.x = pos["x"]
        pose.position.y = pos["y"]
        # Orientação aleatória (yaw)
        yaw = random.uniform(-3.14, 3.14)
        pose.orientation.z = yaw
        return pose

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        if self.delete_entity_client.wait_for_service(timeout_sec=2.0):
            future = self.delete_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        else:
            self.get_logger().warn("Serviço /delete_entity não disponível.")

    def spawn_entity(self, pose: Pose):
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = pose
        req.reference_frame = "world"

        if self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
            future = self.spawn_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        else:
            self.get_logger().warn("Serviço /spawn_entity não disponível.")


def main(args=None):
    rclpy.init(args=args)
    # Instancia a classe corrigida
    node = RobotSpawner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()