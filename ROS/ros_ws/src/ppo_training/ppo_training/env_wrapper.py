# ppo_training/ppo_training/env_wrapper.py
import math
import time
from time import sleep

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
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
                 n_sectors: int = 36,
                 max_linear: float = 0.7,
                 max_w: float = 1.0,
                 min_goal_dist: float = 0.2,
                 collision_dist: float = 0.2,
                 max_steps: int = 2000,
                 step_wait_time: float = 0.1,
                 lookahead_distance: float = 0.8):
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
        self.lookahead_distance = float(lookahead_distance)

        # State
        self.scan = np.array([], dtype=np.float32)
        self.scan_compressed = np.full((self.n_sectors,), 10.0, dtype=np.float32)
        self.previous_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_pose = np.array([0.0, 0.0], dtype=np.float32)
        self.previous_yaw = 0.0
        self.robot_yaw = 0.0
        self.goal = np.array([0.0, 0.0], dtype=np.float32)
        self.current_plan = None
        self.lookahead_rel = np.array([0.0, 0.0], dtype=np.float32)

        self.step_number = 0
        self.min_obst_dist = 10.0
        self.done = False

        # collision flag updated by bumper callback (if available)
        self.collided = False

        # reward scalars (suggested)
        self.k_goal = 2.0         # progress to goal (was maybe similar)
        self.k_head = 0.6         # heading alignment weight (was larger before)
        self.k_plan_align = 0.8   # reward for following lookahead / plan direction

        # safety / proximity
        self.k_prox = 4.0
        self.prox_thresh = 1.0
        self.k_front = 8.0
        self.front_thresh = 1.0
        self.last_front_min = 10.0  # valor grande (sem obstáculo)

        # smoothness / action penalties
        self.k_omega = 0.12       # penalize large angular commands (not zero)
        self.k_step = 0.01        # small time penalty

        # terminal rewards (keep large)
        self.collision_reward = -6.0
        self.goal_reward = 6.0

        # clipping
        self.reward_clip = 20.0

        # Smoothing and spin detection parameters
        self.smooth_alpha = 0.6         # smoothing factor for angular velocity (0=no smoothing, 1=keep previous)
        self.prev_action = (0.0, 0.0)   # last applied (lin, ang)
        self.spin_v_thresh = 0.05       # below this linear velocity, robot is effectively spinning in place
        self.spin_ang_thresh = 0.5      # above this angular velocity, considered spinning
        self.spin_counter = 0           # count consecutive spins

        # ROS2 pubs/subs
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Cliente para o serviço de respawn (robot_spawner)
        self.respawn_robot_client = self.create_client(Empty, 'respawn_robot')
        while not self.respawn_robot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Serviço "respawn_robot" não disponível, esperando...')

        # optional bumper contact subscription
        if HAVE_BUMPER_MSG:
            self.bumper_sub = self.create_subscription(ContactsState, '/bumper_states', self.bumper_callback, 10)
            self.get_logger().info("ContactsState subscriber enabled for /bumper_states")
        else:
            self.bumper_sub = None
            self.get_logger().info("ContactsState not available; bumper-based collision detection disabled")

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-0.3, -self.max_w], dtype=np.float32),
            high=np.array([self.max_lin, self.max_w], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = self.n_sectors + 6 + 2
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

        # new, 08/10 17h37
        # store scan angle limits for correct sector mapping later
        try:
            self.scan_angle_min = float(msg.angle_min)
            self.scan_angle_max = float(msg.angle_max)
        except Exception:
            # fallback to full circle if not available
            self.scan_angle_min = -math.pi
            self.scan_angle_max = math.pi

        # --- compute last_front_min: find the sectors that correspond to a frontal angular window ---
        # map angle -> sector index
        try:
            # number of sectors in compressed scan
            n = int(self.n_sectors) if hasattr(self, 'n_sectors') else len(self.scan_compressed)

            # normalize angles (ensure scan_angle_min < scan_angle_max)
            ang_min = float(self.scan_angle_min)
            ang_max = float(self.scan_angle_max)
            ang_range = ang_max - ang_min
            if ang_range == 0.0:
                # fallback: take full scan
                center_idx = n // 2
            else:
                # index corresponding to angle 0 (front)
                center_idx = int(round((0.0 - ang_min) / ang_range * (n - 1)))

            # choose a small window around front (10% of sectors, at least 1)
            half_w = max(1, int(round(0.1 * n)))
            start = max(0, center_idx - half_w)
            end = min(n, center_idx + half_w + 1)

            if self.scan_compressed.size > 0:
                self.last_front_min = float(np.min(self.scan_compressed[start:end]))
            else:
                self.last_front_min = float(self.min_obst_dist)
        except Exception:
            # conservative fallback if anything goes wrong
            self.last_front_min = float(self.min_obst_dist)

    def odom_callback(self, msg: Odometry):
        # Keep previous pose/yaw for reward computation
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

    def plan_callback(self, msg: Path):
        """Recebe o caminho do planner global (Nav2)."""
        if msg is not None and len(msg.poses) > 0:
            self.current_plan = msg
            self.get_logger().info(f"Received plan with {len(msg.poses)} poses.")
        else:
            self.get_logger().warn(f"Received empty plan!")  # dentro do else do plan_callback
            self.current_plan = None

    def compute_lookahead(self):
        """Seleciona ponto no caminho a `lookahead_distance` e converte para o referencial do robô."""
        if self.current_plan is None or len(self.current_plan.poses) == 0:
            return np.zeros(2, dtype=np.float32)

        # encontra ponto do plano mais próximo do robô
        min_idx = 0
        min_dist = float('inf')
        for i, pose_stamped in enumerate(self.current_plan.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d = math.hypot(px - self.robot_pose[0], py - self.robot_pose[1])
            if d < min_dist:
                min_idx = i
                min_dist = d

        # procura ponto a lookahead_distance adiante
        look_pt = self.current_plan.poses[-1].pose
        for j in range(min_idx, len(self.current_plan.poses)):
            ps = self.current_plan.poses[j].pose
            dx = ps.position.x - self.robot_pose[0]
            dy = ps.position.y - self.robot_pose[1]
            d = math.hypot(dx, dy)
            if d >= self.lookahead_distance:
                look_pt = ps
                break

        # converte ponto global em coordenadas relativas ao robô
        dx = look_pt.position.x - self.robot_pose[0]
        dy = look_pt.position.y - self.robot_pose[1]
        yaw = self.robot_yaw
        x_rel = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        y_rel = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        self.lookahead_rel = np.array([x_rel, y_rel], dtype=np.float32)
        self.get_logger().debug(f"Lookahead point: {look_pt.position.x:.2f}, {look_pt.position.y:.2f}")
        return self.lookahead_rel

    # ------------------ Gym API ------------------
    def get_state(self):
        # normalize sectors to [0,1] (10.0 is treated as max range)
        if self.scan_compressed is None:
            sectors = np.full((self.n_sectors,), 10.0, dtype=np.float32)
        else:
            sectors = self.scan_compressed.astype(np.float32)
        sectors = np.clip(sectors, 0.0, 10.0) / 10.0

        goal_vec = self.goal - self.robot_pose
        dist = float(np.linalg.norm(goal_vec))
        if dist > 0.0:
            raw_angle = math.atan2(goal_vec[1], goal_vec[0]) - float(self.robot_yaw)
            curr_angle = normalize_angle(raw_angle)
        else:
            curr_angle = 0.0

        angle_sin = math.sin(curr_angle)
        angle_cos = math.cos(curr_angle)
        min_norm = np.clip(self.min_obst_dist, 0.0, 10.0) / 10.0

        # === NOVO: cálculo explícito de front_min e path_min ===
        n = self.n_sectors
        angle_min = getattr(self, "scan_angle_min", -math.pi)
        angle_max = getattr(self, "scan_angle_max", math.pi)

        # --- front_min ---
        front_frac = (0.0 - angle_min) / (angle_max - angle_min + 1e-9)
        front_idx = int(round(front_frac * (n - 1)))
        front_idx = max(0, min(n - 1, front_idx))
        w = max(1, int(n * 60 / 360))  # janela ±30°
        start = max(0, front_idx - w//2)
        end = min(n, front_idx + w//2 + 1)
        front_min = float(np.min(self.scan_compressed[start:end])) if self.scan_compressed.size > 0 else 10.0

        # --- path_min (em direção ao goal) ---
        frac = (curr_angle - angle_min) / (angle_max - angle_min + 1e-9)
        sector_idx = int(round(frac * (n - 1)))
        sector_idx = max(0, min(n - 1, sector_idx))
        window = max(1, int(n * 20 / 360))  # ±10°
        s0 = max(0, sector_idx - window)
        s1 = min(n, sector_idx + window + 1)
        path_min = float(np.min(self.scan_compressed[s0:s1])) if self.scan_compressed.size > 0 else 10.0

        # --- normalização para [0,1] ---
        front_min_norm = np.clip(front_min, 0.0, 10.0) / 10.0
        path_min_norm = np.clip(path_min, 0.0, 10.0) / 10.0

        # new: alcular ponto lookahead relativo e adicionar ao obs
        lookahead_rel = self.compute_lookahead()
        lookahead_x_rel, lookahead_y_rel = lookahead_rel[0], lookahead_rel[1]
        # store for reward computation
        self.lookahead_rel = np.array([lookahead_x_rel, lookahead_y_rel], dtype=np.float32)

        # --- montagem da observação ---
        obs = np.concatenate([
            sectors,
            np.array([
                dist, 
                angle_sin, 
                angle_cos, 
                min_norm, 
                front_min_norm, 
                path_min_norm,
                lookahead_x_rel,
                lookahead_y_rel
            ], dtype=np.float32)
        ])
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

        # Apply simple smoothing to angular command to avoid abrupt changes
        smoothed_ang = float(self.smooth_alpha * float(self.prev_action[1]) + (1.0 - self.smooth_alpha) * ang)
        vel_msg = Twist()
        vel_msg.linear.x = lin
        vel_msg.angular.z = smoothed_ang
        self.cmd_pub.publish(vel_msg)

        # Update prev_action used for next smoothing and reward osc calc
        current_action_applied = (lin, smoothed_ang)

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
        reward = self.get_reward(current_action_applied)

        # update spin counter detection
        if lin < self.spin_v_thresh and abs(smoothed_ang) > self.spin_ang_thresh:
            self.spin_counter += 1
        else:
            self.spin_counter = 0

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

        # update prev_action after reward calc so delta uses previous applied action
        self.prev_action = current_action_applied

        obs = self.get_state()
        info = {"collision": bool(self.collided)}
        return obs, float(reward), terminated, truncated, info

    def get_reward(self, action):
        """
        Simplified, robust reward:
        - R_goal: progress towards goal (positive if distance decreases)
        - R_step: small negative per-step (time penalty)
        - R_heading: reward for facing the goal (cosine of heading)
        - R_clear: penalty for being too close to obstacles (front_min / min_obst)
        - R_plan_align: small reward if lookahead point lies roughly in front (follow planner)
        - R_collision / R_success: terminal large penalties/rewards handled elsewhere (collision flags / goal reward)
        """
        # --- distances & progress ---
        prev_dist = float(np.linalg.norm(self.goal - self.previous_pose))
        curr_dist = float(np.linalg.norm(self.goal - self.robot_pose))
        # progress reward (positive when closer)
        r_goal = self.k_goal * (prev_dist - curr_dist)

        # small lazy penalty if not improving at all (discourage standing still)
        if curr_dist >= prev_dist - 1e-6:
            r_lazy = -0.01
        else:
            r_lazy = 0.0

        # --- heading (align with goal) ---
        goal_vec = self.goal - self.robot_pose
        if np.linalg.norm(goal_vec) > 1e-6:
            raw_angle = math.atan2(goal_vec[1], goal_vec[0]) - float(self.robot_yaw)
            curr_angle = normalize_angle(raw_angle)
        else:
            curr_angle = 0.0
        r_heading = self.k_head * math.cos(curr_angle)  # in [-k_head, k_head]

        # --- collision / goal (terminal rewards kept large and explicit) ---
        r_col = self.collision_reward if (self.collided or (self.min_obst_dist < self.collision_dist)) else 0.0
        r_goal_done = self.goal_reward if (curr_dist < self.min_goal_dist) else 0.0

        # --- proximity / clearance: frontal emphasis ---
        curr_min = float(self.min_obst_dist)
        front_min = getattr(self, "last_front_min", curr_min)  # compute in callback or earlier; fallback curr_min
        # soft penalty if within prox_thresh
        if curr_min < self.prox_thresh:
            r_prox = - self.k_prox * (1.0 - curr_min / self.prox_thresh)
        else:
            r_prox = 0.0
        # additional frontal penalty (stronger if directly in front)
        if front_min < self.front_thresh:
            r_front = - self.k_front * (1.0 - front_min / self.front_thresh)
        else:
            r_front = 0.0

        # --- plan-following via lookahead: reward when lookahead is generally forward (x positive) and not behind ---
        # lookahead_rel is [x_rel, y_rel] in robot frame (already stored in self.lookahead_rel in get_state)
        la = getattr(self, "lookahead_rel", np.array([0.0, 0.0], dtype=np.float32))
        la_x = float(la[0])
        la_y = float(la[1])
        # angle to lookahead point (small if forward)
        la_angle = math.atan2(la_y, la_x) if (abs(la_x) + abs(la_y)) > 1e-6 else 0.0
        r_plan = self.k_plan_align * max(0.0, math.cos(la_angle))  # reward ∈ [0, k_plan_align]

        # --- smoothness: penalize big angular commands (to reduce oscillation) ---
        ang_cmd = float(action[1])
        r_smooth = - self.k_omega * abs(ang_cmd)

        # --- step/time penalty ---
        r_step = - self.k_step

        # sum components
        reward = (r_goal + r_lazy + r_heading + r_plan + r_smooth + r_step
                + r_prox + r_front + r_col + r_goal_done)

        # clip to avoid exploding values
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        return reward

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

        # reset smoothing & spin counters
        self.prev_action = (0.0, 0.0)
        self.spin_counter = 0

        # stop robot just in case
        self._publish_stop()

        # reset Gazebo world
        # reset_client = self.create_client(Empty, '/reset_world')
        # req = Empty.Request()
        # if reset_client.wait_for_service(timeout_sec=2.0):
        #     try:
        #         reset_client.call_async(req)
        #     except Exception as e:
        #         self.get_logger().warn(f"/reset_world call failed: {e}")
        # else:
        #     self.get_logger().warn("/reset_world service not available")
        req = Empty.Request()
        future = self.respawn_robot_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info("Robô respawnado via serviço.")

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