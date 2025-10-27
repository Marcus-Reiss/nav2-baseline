# ppo_training/ppo_training/goal_spawner.py
import os
import random
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

# Action
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


class GoalSpawnerNode(Node):
    def __init__(self):
        super().__init__('goal_spawner_node')

        # Caminho do modelo de goal (ajuste conforme seu pacote / modelo)
        self.entity_dir_path = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'models/turtlebot3_dqn_world/goal_box'
        )
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        try:
            with open(self.entity_path, 'r') as f:
                self.entity = f.read()
        except Exception as e:
            self.get_logger().warn(f"Falha ao ler model.sdf em {self.entity_path}: {e}")
            self.entity = ''

        self.entity_name = "goal"

        # Posições possíveis
        self.goal_candidates = [
            {"x": 0.0, "y": 0.0},
            {"x": -2.2, "y": -2.2},
            {"x": -2.2, "y": 2.2},
            {"x": 2.2, "y": -2.2},
            {"x": 2.2, "y": 2.2},
            {"x": 0.0, "y": 2.0},
            {"x": 0.0, "y": -2.0},
            {"x": -2.0, "y": 0.0},
            {"x": 2.0, "y": 0.0}
        ]

        # Publisher de goal (para os envs)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # serviços do gazebo
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')

        # pause/unpause if available
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

        # Action client para Nav2 (NavigateToPose)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # expose a service spawn_new_goal that envs call at reset
        self.spawn_goal_srv = self.create_service(Empty, '/spawn_new_goal', self.spawn_new_goal_callback)

        self.get_logger().info("GoalSpawnerNode pronto. Serviço '/spawn_new_goal' disponível.")
        # self.get_logger().debug("Aguardando action server /navigate_to_pose (conexão será tentada no envio).")
        # no GoalSpawner.__init__ ou training_node init:
        self.get_logger().info("Aguardando navigate_to_pose...")
        if self._action_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().info("navigate_to_pose action server disponível.")
        else:
            self.get_logger().warn("navigate_to_pose NÃO disponível após timeout estendido.")

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

        # Cria PoseStamped com tempo atual (importante para nav2/tf timing)
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = self.goal_pose["x"]
        goal_pose.pose.position.y = self.goal_pose["y"]
        goal_pose.pose.position.z = 0.01
        # identidade quaternion
        # goal_pose.pose.orientation.w = 1.0

        # Publica para que o env_wrapper saiba do novo goal (rosbag / wrapper)
        self.goal_pose_pub.publish(goal_pose)
        self.get_logger().info(f"Novo goal spawnado e enviado ao Nav2 em ({self.goal_pose['x']:.2f}, {self.goal_pose['y']:.2f})")

        # Spawna entidade visual no Gazebo (síncrono)
        self.spawn_entity(goal_pose)

        # Agora tenta enviar action NavigateToPose (assíncrono)
        self._send_navigate_action(goal_pose)

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
        if not self.entity:
            self.get_logger().debug("spawn_entity: xml da entidade vazio — pulando spawn visual.")
            return
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

    # ----------------- Action senders / callbacks -------------------------------------------------
    def _send_navigate_action(self, pose_stamped: PoseStamped, server_wait_sec: float = 5.0):
        """Tenta enviar a ação NavigateToPose. Não bloqueante a longo prazo."""
        # espera por server (até server_wait_sec)
        # if not self._action_client.wait_for_server(timeout_sec=server_wait_sec):
        #     self.get_logger().warn(f"Action server 'navigate_to_pose' não disponível após {server_wait_sec}s. Goal NÃO enviado ao Nav2.")
        #     return
        # tenta aguardar por mais tempo e com retries curtas
        total_wait = max(5.0, server_wait_sec)
        interval = 1.0
        waited = 0.0
        while not self._action_client.wait_for_server(timeout_sec=interval) and waited < total_wait:
            waited += interval
            self.get_logger().info(f"Waiting for navigate_to_pose action server... {waited:.0f}/{total_wait:.0f}s")
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn(f"Action server 'navigate_to_pose' não disponível após {total_wait}s. Goal NÃO enviado ao Nav2.")
            return

        # Monta o goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        self.get_logger().info("Enviando goal para navigate_to_pose...")
        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._on_feedback
        )
        # adiciona callback para tratar resposta
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().warn(f"Erro ao enviar goal: {e}")
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Goal REJEITADO pelo servidor. (navigate_to_pose)')
            return

        self.get_logger().info('Goal ACEITO. Aguardando resultado...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg):
        # feedback_msg é um GoalHandle feedback container
        # Manter silencioso para não poluir logs excesivamente
        pass

    def _on_result(self, future):
        try:
            result = future.result()
            self.get_logger().info(f'Resultado da navegação: status={result.status}')
        except Exception as e:
            self.get_logger().warn(f'Erro ao obter resultado da ação: {e}')


# ---------------------------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = GoalSpawnerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
