#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from gazebo_msgs.msg import ContactsState
from std_msgs.msg import Bool, UInt32, Empty

class CollisionMonitor(Node):
    """
    Lê /bumper_states (gazebo_msgs/ContactsState) e:
      - Publica /collision (Bool) com o estado instantâneo de contato.
      - Publica /collision_event (Empty) APENAS na transição sem->com contato.
      - Publica /collision_count (UInt32) com o total APENAS quando incrementa.
    Parâmetros:
      - input_topic (string): tópico de entrada. Default: '/bumper_states'
      - collision_topic (string): tópico Bool. Default: '/collision'
      - event_topic (string): tópico de evento. Default: '/collision_event'
      - count_topic (string): tópico de contagem. Default: '/collision_count'
      - debounce_time (double, s): tempo mínimo entre eventos contados. Default: 0.2
      - publish_bool_always (bool): se True, publica Bool a cada mensagem; se False, só em mudança. Default: True
    """
    def __init__(self):
        super().__init__('collision_monitor')

        # Parâmetros
        self.input_topic = self.declare_parameter('input_topic', '/bumper_states').get_parameter_value().string_value
        self.collision_topic = self.declare_parameter('collision_topic', '/collision').get_parameter_value().string_value
        self.event_topic = self.declare_parameter('event_topic', '/collision_event').get_parameter_value().string_value
        self.count_topic = self.declare_parameter('count_topic', '/collision_count').get_parameter_value().string_value
        self.debounce_time = self.declare_parameter('debounce_time', 0.2).get_parameter_value().double_value
        self.publish_bool_always = self.declare_parameter('publish_bool_always', True).get_parameter_value().bool_value

        # Publishers
        self.pub_collision = self.create_publisher(Bool, self.collision_topic, 10)
        # Count com QoS "latched"
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_count = self.create_publisher(UInt32, self.count_topic, latched_qos)
        self.pub_event = self.create_publisher(Empty, self.event_topic, 10)

        # Subscriber
        self.sub = self.create_subscription(ContactsState, self.input_topic, self._cb, 10)

        # Estado interno
        self.in_collision_prev = False
        self.count = 0
        # Publica contagem inicial (latched)
        init_cnt = UInt32()
        init_cnt.data = self.count
        self.pub_count.publish(init_cnt)

        # Para debounce
        self.last_event_time = self.get_clock().now() - Duration(seconds=self.debounce_time)

        self.get_logger().info(
            f"CollisionMonitor ouvindo {self.input_topic} | "
            f"publicando {self.collision_topic}, {self.event_topic}, {self.count_topic} | "
            f"debounce={self.debounce_time:.3f}s"
        )

    def _cb(self, msg: ContactsState):
        now = self.get_clock().now()
        in_collision = len(msg.states) > 0

        # Publica Bool conforme configurado
        if self.publish_bool_always or (in_collision != self.in_collision_prev):
            b = Bool()
            b.data = in_collision
            self.pub_collision.publish(b)

        # Detecção de borda de subida + debounce
        if in_collision and not self.in_collision_prev:
            if (now - self.last_event_time) >= Duration(seconds=self.debounce_time):

                # Verifica se existe algum contato que NÃO seja com o ground_plane
                valid_contact = any(
                    state.collision2_name != "ground_plane::link::collision"
                    and state.collision1_name != "ground_plane::link::collision"
                    for state in msg.states
                )

                if valid_contact:
                    self.count += 1
                    self.last_event_time = now

                    # Publica evento (pulso) e a nova contagem (latched)
                    self.pub_event.publish(Empty())
                    c = UInt32()
                    c.data = self.count
                    self.pub_count.publish(c)

        self.in_collision_prev = in_collision


def main():
    rclpy.init()
    node = CollisionMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
