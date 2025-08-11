import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from math import pi
from tf_transformations import quaternion_from_euler


class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__('goal_pub_node')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        timer_period = 5.0
        self.timer = self.create_timer(timer_period, self.send_goal)

    def send_goal(self):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('NavigateToPose action server not available yet')
            return
        
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = -9.0
        goal.pose.pose.position.y = 0.0

        q = quaternion_from_euler(0, 0, pi)
        goal.pose.pose.orientation.w = q[3]
        
        self.get_logger().info('Sending goal...')
        self._action_client.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
