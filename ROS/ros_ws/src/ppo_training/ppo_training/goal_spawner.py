# ppo_training/ppo_training/goal_spawner.py
import os
import random
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty


class GoalSpawnerNode(Node):
    def __init__(self):
        super().__init__('goal_spawner_node')

        # Caminho do modelo de goal
        self.entity_dir_path = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'models/turtlebot3_dqn_world/goal_box'
        )
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()

        self.entity_name = "goal"

        # Posições fixas exemplo
        # static_bkp.world (verificar)
        # self.goal_candidates = [
        #     {"x": -3.0, "y": -2.0},
        #     {"x": -1.5, "y": 0.0},
        #     {"x": -3.0, "y": 3.0},
        #     {"x": 3.0,  "y": 3.0},
        #     {"x": 3.0,  "y": 0.0},
        #     {"x": 3.0,  "y": -3.0},
        #     {"x": 2.0,  "y": 1.0},
        #     {"x": -2.0, "y": -1.5},
        #     {"x": -1.0, "y": -2.5},
        #     {"x": -0.5, "y": -2.5},
        #     {"x": -4.0, "y": -3.5},
        #     {"x": 4.5,  "y": 4.5},
        #     {"x": 3.5,  "y": -4.5},
        #     {"x": 0.0,  "y": 4.0}
        # ]
        
        # static.world
        # self.goal_candidates = [
        #     {"x": 0.5, "y": 0.0},
        #     {"x": 2.0, "y": 0.3},
        #     {"x": 2.0, "y": -3.0},
        #     {"x": 2.0, "y": 3.0},
        #     {"x": 0.0, "y": 1.5},
        #     {"x": 0.0, "y": 2.0},
        #     {"x": -2.0, "y": 0.0},
        #     {"x": -0.5, "y": -1.5},
        #     {"x": 0.75, "y": -1.75},
        #     {"x": -1.5, "y": -3.0},
        #     {"x": -3.5, "y": -1.5},
        #     {"x": -3.0, "y": 0.0},
        #     {"x": -4.0, "y": 0.0}
        # ]

        # # atrás dos obstáculos + outros normais
        # self.goal_candidates = [
        #     {"x": 1.5, "y": 2.0},
        #     {"x": 1.5, "y": 1.75},
        #     {"x": -1.5, "y": 3.0},
        #     {"x": -2.0, "y": 4.0},
        #     {"x": -3.5, "y": 0.0},
        #     {"x": -3.0, "y": 0.15},
        #     {"x": -1.0, "y": -3.0},
        #     {"x": 1.5, "y": -4.0},
        #     {"x": -2.0, "y": 1.5},
        #     {"x": 0.0, "y": 3.0},
        #     {"x": 0.75, "y": -1.25},
        #     {"x": -4.0, "y": -2.5},
        #     {"x": 4.5, "y": 3.0},
        #     {"x": 2.0, "y": 3.0},
        #     {"x": -3.0, "y": 2.0},
        #     {"x": -0.5, "y": -3.5},
        #     {"x": 1.0, "y": 3.0},
        #     {"x": 2.5, "y": -3.0},
        #     {"x": 3.0, "y": 3.0}
        # ]

        # corridor_3x10_static.world
        self.goal_candidates = [
            {"x": -4.5, "y": 0.0},
            {"x": -4.5, "y": 0.5},
            {"x": -4.5, "y": -0.5}
        ]

        # Publisher de goal (para os envs)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # serviços do gazebo
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')

        # pause/unpause if available
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

        # expose a service spawn_new_goal that envs call at reset
        self.spawn_goal_srv = self.create_service(Empty, '/spawn_new_goal', self.spawn_new_goal_callback)

        self.get_logger().info("GoalSpawnerNode pronto. Serviço '/spawn_new_goal' disponível.")

    # ---------------------------------------------------------------------------------------------
    def _call_service_sync(self, client, request, timeout=5.0):
        if client is None:
            return False, "no_client"
        if not client.wait_for_service(timeout_sec=timeout):
            return False, "wait_for_service_timeout"
        fut = client.call_async(request)
        try:
            rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        except Exception as e:
            return False, e
        if not fut.done():
            return False, "call_timeout"
        try:
            return True, fut.result()
        except Exception as e:
            return False, e

    def _pause_physics(self, timeout=2.0):
        req = Empty.Request()
        ok, res = self._call_service_sync(self.pause_client, req, timeout=timeout)
        if ok:
            self.get_logger().debug("Physics paused for spawn")
        return ok

    def _unpause_physics(self, timeout=2.0):
        req = Empty.Request()
        ok, res = self._call_service_sync(self.unpause_client, req, timeout=timeout)
        if ok:
            self.get_logger().debug("Physics unpaused after spawn")
        return ok

    def spawn_new_goal_callback(self, request, response):
        """Callback chamado quando o ambiente pede um novo goal."""
        self.get_logger().debug("spawn_new_goal solicitado; pausando física e (re)spawnando goal.")
        # pause physics to avoid artefacts
        self._pause_physics(timeout=2.0)

        # delete old visual (if present)
        self.delete_entity()

        # generate new pose
        self.generate_goal_pose()

        goal_pose = PoseStamped()
        goal_pose.pose.position.x = self.goal_pose["x"]
        goal_pose.pose.position.y = self.goal_pose["y"]
        goal_pose.pose.position.z = 0.01

        # Publica para que o env_wrapper saiba do novo goal
        self.goal_pose_pub.publish(goal_pose)

        # Spawna entidade visual no Gazebo (síncrono)
        self.spawn_entity(goal_pose)

        self.get_logger().info(f"Novo goal spawnado em ({self.goal_pose['x']:.2f}, {self.goal_pose['y']:.2f})")

        # unpause physics
        self._unpause_physics(timeout=2.0)

        return response

    def generate_goal_pose(self):
        """Gera posição aleatória do goal em alguma das posições pré-definidas"""
        self.goal_pose = random.choice(self.goal_candidates)

    def delete_entity(self):
        """Remove entidade antiga (se existir)."""
        req = DeleteEntity.Request()
        req.name = self.entity_name
        ok, res = self._call_service_sync(self.delete_entity_client, req, timeout=2.0)
        if not ok:
            self.get_logger().debug("delete_entity: serviço não disponível ou falhou (pode ser que entidade não exista).")
        else:
            self.get_logger().debug("Entidade anterior removida (se existia).")

    def spawn_entity(self, pose: PoseStamped):
        """Spawna entidade visual no Gazebo de forma síncrona."""
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        # initial_pose é um geometry_msgs/Pose. Setamos posição e mantemos orientação padrão
        req.initial_pose = pose.pose
        ok, res = self._call_service_sync(self.spawn_entity_client, req, timeout=5.0)
        if not ok:
            self.get_logger().warn(f"spawn_entity: serviço não disponível ou falhou: {res}")
        else:
            self.get_logger().debug("spawn_entity concluído com sucesso.")

def main(args=None):
    rclpy.init(args=args)
    node = GoalSpawnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
