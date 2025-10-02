# ppo_training/ppo_training/env_wrapper.py
import math
import time
from time import sleep

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
import gymnasium as gym
from gymnasium import spaces
from std_srvs.srv import Empty

# Try to import ContactsState for bumper collision detection (optional)
try:
    from gazebo_msgs.msg import ContactsState
    HAVE_BUMPER_MSG = True
except Exception:
    HAVE_BUMPER_MSG = False


def quaternion_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compress_scan(scan, n_sectors=24, default_value=10.0):
    if scan is None or scan.size == 0:
        return np.full((n_sectors,), default_value, dtype=np.float32)

    L = scan.size
    if L < n_sectors and L > 0:
        repeats = int(np.ceil(n_sectors / L))
        scan2 = np.tile(scan, repeats)[:n_sectors]
        return scan2.astype(np.float32)

    sector_len = max(1, L // n_sectors)
    sectors = np.zeros((n_sectors,), dtype=np.float32)
    for i in range(n_sectors):
        start = i * sector_len
        end = start + sector_len if i < n_sectors - 1 else L
        seg = scan[start:end]
        val = np.min(seg) if seg.size > 0 else default_value
        sectors[i] = float(val)
    return sectors


class PPOEnvironment(Node, gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 n_sectors: int = 24,
                 max_linear: float = 0.7,
                 max_w: float = 1.0,
                 min_goal_dist: float = 0.2,
                 collision_dist: float = 0.2,
                 max_steps: int = 2000,
                 step_wait_time: float = 0.1):
        Node.__init__(self, 'ppo_env')
        gym.Env.__init__(self)

        # Params
        self.n_sectors = int(n_sectors)
        self.max_lin = float(max_linear)
        self.max_w = float(max_w)
        self.min_goal_dist = float(min_goal_dist)
        self.collision_dist = float(collision_dist)
        self.max_steps = int(max_steps)
        self.step_wait_time = float(step_wait_time)

        # State
        self.scan = np.array([], dtype=np.float32)
        self.scan_compressed = np.full((self.n_sectors,), 10.0, dtype=np.float32)
        self.previous_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.previous_yaw = 0.0
        self.robot_yaw = 0.0
        self.goal = np.array([0.0, 0.0], dtype=np.float32)

        self.step_number = 0
        self.min_obst_dist = 10.0
        self.done = False

        # collision flag updated by bumper callback (if available)
        self.collided = False

        # Reward shaping weights (tune later)
        self.k_align = 0.6
        self.k_angle = 0.2

        # ROS2 pubs/subs
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # optional bumper contact subscription
        if HAVE_BUMPER_MSG:
            self.bumper_sub = self.create_subscription(ContactsState, '/bumper_states', self.bumper_callback, 10)
            self.get_logger().info("ContactsState subscriber enabled for /bumper_states")
        else:
            self.bumper_sub = None
            self.get_logger().info("ContactsState not available; bumper-based collision detection disabled")

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_w], dtype=np.float32),
            high=np.array([self.max_lin, self.max_w], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = self.n_sectors + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # small helper state to detect new goals
        self._last_goal_stamp = 0.0

        self.get_logger().info("PPOEnvironment initialized")

    # ------------------ Callbacks ------------------
    def scan_callback(self, msg: LaserScan):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan[np.isinf(scan)] = 10.0
        scan[np.isnan(scan)] = 10.0
        self.scan = scan
        self.scan_compressed = compress_scan(scan, n_sectors=self.n_sectors, default_value=10.0)
        if scan.size > 0:
            self.min_obst_dist = float(np.min(scan))

    def odom_callback(self, msg: Odometry):
        self.previous_pose = self.robot_pose.copy()
        self.previous_yaw = float(self.robot_yaw)

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pose = np.array([pos.x, pos.y], dtype=np.float32)
        self.robot_yaw = float(quaternion_to_yaw(q.x, q.y, q.z, q.w))

    def goal_callback(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)
        # remember last update time so reset can wait for new goal
        self._last_goal_stamp = time.time()

    def bumper_callback(self, msg):
        # If any contact reported, flag collision (message contains a list of contacts)
        try:
            if hasattr(msg, "states") and len(msg.states) > 0:
                # If there is at least one contact with non-zero force -> collision
                for st in msg.states:
                    if st.total_wrench.force.x != 0.0 or st.total_wrench.force.y != 0.0 or st.total_wrench.force.z != 0.0:
                        self.collided = True
                        return
                # fallback: if states present but zero force, still consider collision
                self.collided = True
            else:
                # no contact
                return
        except Exception:
            # conservative fallback
            self.collided = True

    # ------------------ Gym API ------------------
    def get_state(self):
        if self.scan_compressed is None:
            sectors = np.full((self.n_sectors,), 10.0, dtype=np.float32)
        else:
            sectors = self.scan_compressed.astype(np.float32)

        goal_vec = self.goal - self.robot_pose
        dist = float(np.linalg.norm(goal_vec))
        if dist > 0.0:
            raw_angle = math.atan2(goal_vec[1], goal_vec[0]) - float(self.robot_yaw)
            angle = normalize_angle(raw_angle)
        else:
            angle = 0.0
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)

        obs = np.concatenate([sectors, np.array([dist, angle_sin, angle_cos], dtype=np.float32)])
        return obs.astype(np.float32)

    def _publish_stop(self, n=5):
        """Publish zero velocity a few times to ensure robot stops moving."""
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        for _ in range(n):
            self.cmd_pub.publish(stop)
            sleep(0.02)

    def _wait_for_initial_sensors(self, timeout=2.0):
        """Wait until we have at least one scan and a non-zero goal (or timeout)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            # require: scan has some readings AND goal has been updated recently
            if self.scan.size > 0 and (np.linalg.norm(self.goal) > 0.001 or (time.time() - self._last_goal_stamp) < 1.0):
                return True
            sleep(0.05)
        return False

    def step(self, action):
        # publish clipped action
        action = np.asarray(action, dtype=np.float32).flatten()
        lin = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        ang = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))

        vel_msg = Twist()
        vel_msg.linear.x = lin
        vel_msg.angular.z = ang
        self.cmd_pub.publish(vel_msg)

        self.step_number += 1

        # wait a small time for sensors to update (executor thread must be running)
        prev_pose = self.robot_pose.copy()
        prev_min_obst = float(self.min_obst_dist)
        t0 = time.time()
        while time.time() - t0 < self.step_wait_time:
            # if odom changed by threshold or min obstacle changed -> proceed
            if np.linalg.norm(self.robot_pose - prev_pose) > 1e-4 or self.min_obst_dist != prev_min_obst or self.collided:
                break
            sleep(0.01)

        # Compute reward using current (hopefully updated) state
        reward = self.get_reward((lin, ang))

        # termination flags
        terminated = False
        truncated = False

        # collision: prefer bumper flag if available, else laser-based min distance
        if self.collided or (self.min_obst_dist < self.collision_dist):
            terminated = True
        elif float(np.linalg.norm(self.goal - self.robot_pose)) < self.min_goal_dist:
            terminated = True
        elif self.step_number >= self.max_steps:
            truncated = True

        obs = self.get_state()
        info = {"collision": bool(self.collided)}
        return obs, float(reward), terminated, truncated, info

    def get_reward(self, action):
        prev_dist = float(np.linalg.norm(self.goal - self.previous_pose))
        curr_dist = float(np.linalg.norm(self.goal - self.robot_pose))
        rg = prev_dist - curr_dist

        # prev angle
        prev_vec = self.goal - self.previous_pose
        if np.linalg.norm(prev_vec) > 0:
            prev_raw_angle = math.atan2(prev_vec[1], prev_vec[0]) - float(self.previous_yaw)
            prev_angle = normalize_angle(prev_raw_angle)
        else:
            prev_angle = 0.0

        # curr angle
        curr_vec = self.goal - self.robot_pose
        if np.linalg.norm(curr_vec) > 0:
            curr_raw_angle = math.atan2(curr_vec[1], curr_vec[0]) - float(self.robot_yaw)
            curr_angle = normalize_angle(curr_raw_angle)
        else:
            curr_angle = 0.0

        delta_angle = (abs(prev_angle) - abs(curr_angle))
        # alignment: forward speed times cos(curr_angle) -> positive when pointing to goal
        alignment = 0.0
        try:
            alignment = float(action[0]) * math.cos(curr_angle)
        except Exception:
            alignment = 0.0

        rc = -10.0 if (self.collided or (self.min_obst_dist < self.collision_dist)) else 0.0
        rw = -0.1 * abs(action[1]) if abs(action[1]) > self.max_w else 0.0
        rt = -0.01

        reward = rg + self.k_align * alignment + self.k_angle * delta_angle + rc + rw + rt
        return float(reward)

    def reset(self, seed=None, options=None):
        # gymnasium reset API (obs, info)
        super().reset(seed=seed)

        # Clear state
        self.scan = np.array([], dtype=np.float32)
        self.scan_compressed = np.full((self.n_sectors,), 10.0, dtype=np.float32)
        self.previous_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.previous_yaw = 0.0
        self.robot_yaw = 0.0
        self.goal = np.array([0.0, 0.0], dtype=np.float32)

        self.step_number = 0
        self.min_obst_dist = 10.0
        self.done = False
        self.collided = False

        # stop robot just in case
        self._publish_stop()

        # reset Gazebo world
        reset_client = self.create_client(Empty, '/reset_world')
        req = Empty.Request()
        if reset_client.wait_for_service(timeout_sec=2.0):
            try:
                reset_client.call_async(req)
            except Exception as e:
                self.get_logger().warn(f"/reset_world call failed: {e}")
        else:
            self.get_logger().warn("/reset_world service not available")

        # Request a new goal from the GoalSpawner (the goal_spawner node should publish /goal_pose)
        # We will wait for the '/goal_pose' callback to update self.goal (using _last_goal_stamp)
        goal_client = self.create_client(Empty, 'spawn_new_goal')
        req2 = Empty.Request()
        if goal_client.wait_for_service(timeout_sec=2.0):
            try:
                goal_client.call_async(req2)
            except Exception as e:
                self.get_logger().warn(f"spawn_new_goal call failed: {e}")
        else:
            self.get_logger().warn("spawn_new_goal service not available")

        # Wait briefly until sensors and goal are updated (or timeout)
        got = self._wait_for_initial_sensors(timeout=2.0)
        if not got:
            self.get_logger().warn("Timeout waiting for initial sensors/goal after reset")

        # extra short wait to stabilize
        sleep(max(0.1, self.step_wait_time))

        obs = self.get_state()
        info = {}
        return obs, info

    def render(self, mode="human"):
        pass
