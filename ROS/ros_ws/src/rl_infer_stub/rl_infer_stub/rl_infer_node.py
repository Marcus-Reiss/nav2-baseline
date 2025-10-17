import rclpy
from rclpy.node import Node
from nav2_rl_controller.srv import RLInfer

class RLInferNode(Node):
    def __init__(self):
        super().__init__('rl_infer_node')
        self.srv = self.create_service(RLInfer, '/rl_infer', self.infer_callback)
        self.get_logger().info("RLInfer stub ready at /rl_infer")

    def infer_callback(self, request, response):
        self.get_logger().info(f"Received obs: {len(request.obs)} values")
        response.linear_x = 0.15
        response.angular_z = 0.0
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RLInferNode()
    rclpy.spin(node)
    rclpy.shutdown()
