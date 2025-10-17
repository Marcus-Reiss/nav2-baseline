#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav2_rl_controller.srv import RLInfer

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


class RLInferNode(Node):
    def __init__(self):
        super().__init__('rl_infer_node')

        self.declare_parameter('model_path', 'models/ppo_model.zip')
        self.model_path = self.get_parameter('model_path').value

        self.declare_parameter('n_sectors', 36)
        self.n_sectors = self.get_parameter('n_sectors').value

        self.sectors = np.full((self.n_sectors,), 1.0, dtype=np.float32)
        self.min_obst = 1.0
        self.front_min = 1.0
        self.path_min = 1.0
        self.robot_pose = np.zeros(2, dtype=np.float32)
        self.robot_yaw = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.scan_angle_min = -math.pi
        self.scan_angle_max = math.pi

        # Subs
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)

        self.srv = self.create_service(RLInfer, 'rl_infer', self.cb_infer)

        if PPO is None:
            self.get_logger().error("Stable-Baselines3 não instalado.")
            self.model = None
        else:
            try:
                self.model = PPO.load(self.model_path)
                self.get_logger().info(f"Modelo PPO carregado: {self.model_path}")
            except Exception as e:
                self.get_logger().error(f"Falha ao carregar modelo: {e}")
                self.model = None

    # ====== Callbacks ======
    def scan_cb(self, msg: LaserScan):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan[np.isinf(scan)] = 10.0
        scan[np.isnan(scan)] = 10.0
        n = self.n_sectors
        L = len(scan)
        sector_len = max(1, L // n)
        sectors = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            start = i * sector_len
            end = start + sector_len if i < n - 1 else L
            seg = scan[start:end]
            val = np.min(seg) if seg.size > 0 else 10.0
            sectors[i] = val
        self.sectors = np.clip(sectors, 0.0, 10.0) / 10.0
        self.min_obst = float(np.min(scan)) / 10.0
        self.scan_angle_min = float(msg.angle_min)
        self.scan_angle_max = float(msg.angle_max)

        # calcular front_min e path_min (usando último yaw/goal)
        n = self.n_sectors
        angle_min = self.scan_angle_min
        angle_max = self.scan_angle_max
        front_frac = (0.0 - angle_min) / (angle_max - angle_min + 1e-9)
        front_idx = int(round(front_frac * (n - 1)))
        w = max(1, int(n * 60 / 360))
        start = max(0, front_idx - w//2)
        end = min(n, front_idx + w//2 + 1)
        self.front_min = np.min(sectors[start:end])

        goal_vec = self.goal - self.robot_pose
        dist = np.linalg.norm(goal_vec)
        curr_angle = 0.0 if dist == 0 else normalize_angle(math.atan2(goal_vec[1], goal_vec[0]) - self.robot_yaw)
        frac = (curr_angle - angle_min) / (angle_max - angle_min + 1e-9)
        sector_idx = int(round(frac * (n - 1)))
        window = max(1, int(n * 20 / 360))
        s0 = max(0, sector_idx - window)
        s1 = min(n, sector_idx + window + 1)
        self.path_min = np.min(sectors[s0:s1])

    def odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pose = np.array([pos.x, pos.y], dtype=np.float32)
        self.robot_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def goal_cb(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)

    # ====== Infer ======
    def cb_infer(self, request, response):
        obs = self.compose_obs()
        if self.model is None:
            response.linear_x = 0.0
            response.angular_z = 0.0
            return response
        try:
            action, _ = self.model.predict(obs, deterministic=True)
            response.linear_x = float(action[0])
            response.angular_z = float(action[1])
        except Exception as e:
            self.get_logger().error(f"Model predict failed: {e}")
            response.linear_x = 0.0
            response.angular_z = 0.0
        return response

    def compose_obs(self):
        goal_vec = self.goal - self.robot_pose
        dist = np.linalg.norm(goal_vec)
        curr_angle = 0.0 if dist == 0 else normalize_angle(math.atan2(goal_vec[1], goal_vec[0]) - self.robot_yaw)
        angle_sin = math.sin(curr_angle)
        angle_cos = math.cos(curr_angle)
        obs = np.concatenate([
            self.sectors,
            np.array([
                dist,
                angle_sin,
                angle_cos,
                self.min_obst,
                np.clip(self.front_min, 0.0, 10.0) / 10.0,
                np.clip(self.path_min, 0.0, 10.0) / 10.0
            ], dtype=np.float32)
        ])
        return obs.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = RLInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
