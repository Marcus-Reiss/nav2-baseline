#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
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

        # sectors are normalized to [0,1] matching env_wrapper
        self.sectors = np.full((self.n_sectors,), 1.0, dtype=np.float32)
        # min_obst stored as normalized (0..1) in scan_cb (matches env_wrapper min_norm)
        self.min_obst = 1.0
        # front_min and path_min are also normalized (0..1)
        self.front_min = 1.0
        self.path_min = 1.0

        # poses / yaw
        self.robot_pose = np.zeros(2, dtype=np.float32)
        self.robot_yaw = 0.0
        self.goal = np.zeros(2, dtype=np.float32)

        # scan angle limits
        self.scan_angle_min = -math.pi
        self.scan_angle_max = math.pi

        # lookahead and plan
        self.lookahead_rel = np.zeros(2, dtype=np.float32)
        self.lookahead_distance = 0.8  # must match env_wrapper
        self.current_plan = None

        # tuning constant used for dist normalization (must match env_wrapper's max_dist_norm)
        self.max_dist_norm = 8.5

        # Subs
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(Path, '/plan', self.plan_cb, 1)

        self.srv = self.create_service(RLInfer, 'rl_infer', self.cb_infer)

        # Load model
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
    def plan_cb(self, msg: Path):
        if msg is not None and len(msg.poses) > 0:
            self.current_plan = msg
        else:
            self.current_plan = None

    def compute_lookahead(self):
        if self.current_plan is None or len(self.current_plan.poses) == 0:
            self.lookahead_rel = np.zeros(2, dtype=np.float32)
            return self.lookahead_rel

        # find closest point on plan
        min_idx = 0
        min_dist = float('inf')
        for i, pose_stamped in enumerate(self.current_plan.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d = math.hypot(px - self.robot_pose[0], py - self.robot_pose[1])
            if d < min_dist:
                min_idx = i
                min_dist = d

        # choose lookahead point at >= lookahead_distance ahead
        look_pt = self.current_plan.poses[-1].pose
        for j in range(min_idx, len(self.current_plan.poses)):
            ps = self.current_plan.poses[j].pose
            dx = ps.position.x - self.robot_pose[0]
            dy = ps.position.y - self.robot_pose[1]
            if math.hypot(dx, dy) >= self.lookahead_distance:
                look_pt = ps
                break

        # transform to robot frame
        dx = look_pt.position.x - self.robot_pose[0]
        dy = look_pt.position.y - self.robot_pose[1]
        yaw = self.robot_yaw
        x_rel = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        y_rel = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        self.lookahead_rel = np.array([x_rel, y_rel], dtype=np.float32)
        return self.lookahead_rel

    def scan_cb(self, msg: LaserScan):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan[np.isinf(scan)] = 10.0
        scan[np.isnan(scan)] = 10.0

        n = self.n_sectors
        L = len(scan)
        if L == 0:
            return

        sector_len = max(1, L // n)
        sectors = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            start = i * sector_len
            end = start + sector_len if i < n - 1 else L
            seg = scan[start:end]
            val = np.min(seg) if seg.size > 0 else 10.0
            sectors[i] = val

        # normalize sectors to [0,1] (10.0 -> 1.0)
        self.sectors = np.clip(sectors, 0.0, 10.0) / 10.0

        # min obstacle normalized
        self.min_obst = float(np.min(scan)) / 10.0

        # angles
        self.scan_angle_min = float(getattr(msg, "angle_min", -math.pi))
        self.scan_angle_max = float(getattr(msg, "angle_max", math.pi))

        # compute front_min and path_min using same mapping as env_wrapper
        angle_min = self.scan_angle_min
        angle_max = self.scan_angle_max
        ang_range = angle_max - angle_min if (angle_max - angle_min) != 0.0 else (2.0 * math.pi)

        # front_min (use sectors which are already normalized)
        front_frac = (0.0 - angle_min) / (ang_range + 1e-9)
        front_idx = int(round(front_frac * (n - 1)))
        front_idx = max(0, min(n - 1, front_idx))
        w = max(1, int(n * 60 / 360))
        start = max(0, front_idx - w//2)
        end = min(n, front_idx + w//2 + 1)
        self.front_min = float(np.min(self.sectors[start:end])) if self.sectors.size > 0 else 1.0

        # path_min (towards goal)
        goal_vec = self.goal - self.robot_pose
        dist = np.linalg.norm(goal_vec)
        curr_angle = 0.0 if dist == 0.0 else normalize_angle(math.atan2(goal_vec[1], goal_vec[0]) - self.robot_yaw)
        frac = (curr_angle - angle_min) / (ang_range + 1e-9)
        sector_idx = int(round(frac * (n - 1)))
        sector_idx = max(0, min(n - 1, sector_idx))
        window = max(1, int(n * 20 / 360))
        s0 = max(0, sector_idx - window)
        s1 = min(n, sector_idx + window + 1)
        self.path_min = float(np.min(self.sectors[s0:s1])) if self.sectors.size > 0 else 1.0

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

        # compute some useful diagnostics to log (use normalized versions)
        # dist normalized the same as env_wrapper
        goal_vec = self.goal - self.robot_pose
        dist = np.linalg.norm(goal_vec)
        dist_norm = np.clip(dist / (self.max_dist_norm + 1e-9), 0.0, 1.0)
        la = getattr(self, "lookahead_rel", np.zeros(2, dtype=np.float32))
        la_x_norm = float(np.clip(la[0] / (self.lookahead_distance + 1e-9), -1.0, 1.0))

        self.get_logger().info(
            f"[RLInfer] obs.shape={obs.shape}, dist_norm={dist_norm:.3f}, "
            f"front_min={self.front_min:.3f}, path_min={self.path_min:.3f}, "
            f"la_x_norm={la_x_norm:.3f}"
        )

        try:
            action, _ = self.model.predict(obs, deterministic=True)

            self.get_logger().info(
                f"[RLInfer] model.predict → lin={action[0]:.3f}, ang={action[1]:.3f}"
            )

            response.linear_x = float(action[0])
            response.angular_z = float(action[1])
        except Exception as e:
            self.get_logger().error(f"Model predict failed: {e}")
            response.linear_x = 0.0
            response.angular_z = 0.0
        return response

    def compose_obs(self):
        """
        Build observation with the SAME ordering and normalization as env_wrapper.get_state():
        [sectors (n), dist_norm, angle_sin, angle_cos, min_norm, front_min_norm, path_min_norm, lookahead_x_norm, lookahead_y_norm]
        """
        # sectors already normalized 0..1
        sectors = self.sectors.copy()

        goal_vec = self.goal - self.robot_pose
        dist = float(np.linalg.norm(goal_vec))
        # normalize dist to same scale used in env_wrapper
        dist_norm = np.clip(dist / (self.max_dist_norm + 1e-9), 0.0, 1.0)

        if dist > 0.0:
            raw_angle = math.atan2(goal_vec[1], goal_vec[0]) - float(self.robot_yaw)
            curr_angle = normalize_angle(raw_angle)
        else:
            curr_angle = 0.0
        angle_sin = math.sin(curr_angle)
        angle_cos = math.cos(curr_angle)

        # min_obst already stored normalized (0..1)
        min_norm = float(np.clip(self.min_obst, 0.0, 1.0))

        # front_min and path_min already normalized (0..1)
        front_min_norm = float(np.clip(self.front_min, 0.0, 1.0))
        path_min_norm = float(np.clip(self.path_min, 0.0, 1.0))

        # lookahead in robot frame (meters) -> normalize by lookahead_distance
        lookahead_rel = self.compute_lookahead()
        lookahead_x_norm = float(np.clip(lookahead_rel[0] / (self.lookahead_distance + 1e-9), -1.0, 1.0))
        lookahead_y_norm = float(np.clip(lookahead_rel[1] / (self.lookahead_distance + 1e-9), -2.0, 2.0))

        obs = np.concatenate([
            sectors,
            np.array([
                dist_norm,
                angle_sin,
                angle_cos,
                min_norm,
                front_min_norm,
                path_min_norm,
                lookahead_x_norm,
                lookahead_y_norm
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
