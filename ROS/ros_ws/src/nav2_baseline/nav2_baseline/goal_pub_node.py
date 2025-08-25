import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

# Se você preferir usar yaw em radianos:
try:
    from tf_transformations import quaternion_from_euler
except Exception:
    quaternion_from_euler = None


class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__('goal_pub_node')

        # Parâmetros simples para não “engessar” o objetivo
        self.declare_parameter('target_x', 9.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_yaw', 0.0)  # rad
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('once', True)       # envia só 1 goal
        self.declare_parameter('check_period', 0.5)  # s: frequência de checagem do servidor

        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

        self._goal_sent = False
        self._logged_waiting = False

        # Timer leve que re-tenta até o servidor estar pronto; depois envia 1x
        period = float(self.get_parameter('check_period').value)
        self._timer = self.create_timer(period, self._tick)

    # --- Helpers ---
    def _build_pose(self) -> PoseStamped:
        x = float(self.get_parameter('target_x').value)
        y = float(self.get_parameter('target_y').value)
        yaw = float(self.get_parameter('target_yaw').value)
        frame = str(self.get_parameter('frame_id').value)

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = frame
        ps.pose.position.x = x
        ps.pose.position.y = y

        if quaternion_from_euler is not None:
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
        else:
            # Quat identidade caso lib não esteja disponível
            ps.pose.orientation.w = 1.0

        return ps

    # --- Timer ---
    def _tick(self):
        # Já enviou e está configurado para "once"? nada a fazer.
        if self._goal_sent and bool(self.get_parameter('once').value):
            return

        # Espera o servidor levantar (Nav2 ativo e ação disponível)
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            if not self._logged_waiting:
                self.get_logger().info('Aguardando o servidor de ação navigate_to_pose...')
                self._logged_waiting = True
            return

        # Monta o goal, publica cópia em /goal_pose (entra no rosbag) e envia a ação
        pose = self._build_pose()
        self._goal_pub.publish(pose)

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info('Enviando goal para navigate_to_pose...')
        send_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._on_feedback
        )
        send_future.add_done_callback(self._on_goal_response)
        self._goal_sent = True

    # --- Callbacks da ação ---
    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal REJEITADO pelo servidor. Voltando a tentar...')
            self._goal_sent = False  # permite re-tentar no próximo tick
            return

        self.get_logger().info('Goal ACEITO. Aguardando resultado...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg):
        # Mantém silencioso para não poluir log (o feedback já aparecerá nos tópicos da ação)
        pass

    def _on_result(self, future):
        result = future.result()
        self.get_logger().info(f'Resultado da navegação: status={result.status}')
        # Se quiser re-enviar automaticamente quando terminar, defina once:=False
        if not bool(self.get_parameter('once').value):
            self._goal_sent = False  # permite novo envio no próximo tick


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
