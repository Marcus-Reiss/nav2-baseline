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
                 n_sectors: int = 36,
                 max_linear: float = 0.7,
                 max_w: float = 1.0,
                 min_goal_dist: float = 0.2,
                 collision_dist: float = 0.25,
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
        # New structured weights (can be tuned)
        self.k_goal = 2.5     # progress toward goal (prev_dist - curr_dist)
        self.k_head = 1.0     # heading alignment reward (cos error)
        self.k_omega = 0.15   # penalty on angular speed magnitude
        self.k_osc = 0.15     # penalty on change in angular speed (oscillation)
        self.k_step = 0.001   # small time-step penalty
        self.goal_reward = 10.0
        self.collision_reward = -10.0

        # path-clear reward
        self.k_path_clear = 5.0

        self.reward_clip = 50.0

        # forward motion shaping
        self.k_forward = 2.0
        self.decel_start = 0.05
        self.decel_beta = 10.0

        # smoothing & spin detection
        self.smooth_alpha = 0.5
        self.prev_action = (0.0, 0.0)
        self.spin_counter = 0
        self.spin_v_thresh = 0.05
        self.spin_ang_thresh = 0.5
        self.spin_penalty = -0.5
        self.spin_counter_max = 10

        # ---- NEW: safety / clearance shaping ----
        self.k_prox = 5.0          # penalidade por proximidade
        self.prox_thresh = 1.2     # m â€” comeÃ§a a penalizar
        self.k_clear = 0.8         # recompensa por aumento de clearance
        self.k_ttc = 1.0           # penalidade se TTC for baixo
        self.ttc_thresh = 1.2      # s â€” TTC abaixo disso penaliza
        self.k_front = 30.0         # penalidade para obstÃ¡culo frontal 8.0
        self.front_thresh = 2.5    # m â€” faixa frontal crÃ­tica
        self.prev_min_obst_for_reward = 10.0  # inicializaÃ§Ã£o

        # novo: peso de bloqueio e penalidade linear quando bloqueado
        self.k_block = 100.0        # penalidade base por bloqueio 20.0
        self.k_block_lin = 300.0    # penalidade por mover-se (lin_cmd) enquanto bloqueado 35.0
        self.k_commit = 25.0  # teimosia

        # permitir curvas mais agressivas (menos puniÃ§Ã£o em ang)
        self.k_omega = 0.01
        self.k_osc = 0.01
        # reduÃ§Ã£o de suavizaÃ§Ã£o angular para permitir respostas mais imediatas
        self.smooth_alpha = 0.2

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
            low=np.array([-0.3, -self.max_w], dtype=np.float32),
            high=np.array([self.max_lin, self.max_w], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = self.n_sectors + 6
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

        # --- montagem da observação ---
        obs = np.concatenate([
            sectors,
            np.array([dist, angle_sin, angle_cos, min_norm, front_min_norm, path_min_norm], dtype=np.float32)
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

    # def get_reward(self, action):
    #     """Shaped reward combining goal progress, alignment, and obstacle avoidance."""
    #     # DistÃ¢ncias
    #     prev_dist = float(np.linalg.norm(self.goal - self.previous_pose))
    #     curr_dist = float(np.linalg.norm(self.goal - self.robot_pose))
    #     rg = self.k_goal * (prev_dist - curr_dist)
    #     r_lazy = -0.05 if curr_dist >= prev_dist - 1e-3 else 0.0

    #     # Ã‚ngulos relativos
    #     prev_vec = self.goal - self.previous_pose
    #     prev_angle = normalize_angle(math.atan2(prev_vec[1], prev_vec[0]) - self.previous_yaw) if np.linalg.norm(prev_vec) > 0 else 0.0
    #     curr_vec = self.goal - self.robot_pose
    #     curr_angle = normalize_angle(math.atan2(curr_vec[1], curr_vec[0]) - self.robot_yaw) if np.linalg.norm(curr_vec) > 0 else 0.0

    #     # Alinhamento e aÃ§Ã£o aplicada
    #     r_heading = self.k_head * (math.cos(curr_angle) - 0.5)
    #     lin_cmd, ang_cmd = float(action[0]), float(action[1])

    #     # Penalidades de rotaÃ§Ã£o e oscilaÃ§Ã£o
    #     r_omega = - self.k_omega * abs(ang_cmd)
    #     prev_ang = float(self.prev_action[1]) if hasattr(self, "prev_action") else 0.0
    #     r_osc = - self.k_osc * abs(ang_cmd - prev_ang)
    #     r_step = - self.k_step

    #     # Penalidades fortes de evento
    #     r_col = self.collision_reward if (self.collided or (self.min_obst_dist < self.collision_dist)) else 0.0
    #     r_goal = self.goal_reward if (curr_dist < self.min_goal_dist) else 0.0

    #     # Incentivo de avanÃ§o quando alinhado
    #     r_forward = 4.0 * lin_cmd if abs(curr_angle) < math.radians(30) else 0.0

    #     # ------------------------------------------------------------------
    #     # === NOVOS TERMOS: desvio e seguranÃ§a ===
    #     # Clearance atual e anterior
    #     curr_min = float(self.min_obst_dist)
    #     prev_min = float(getattr(self, "prev_min_obst_for_reward", 10.0))

    #     # Penalidade de proximidade (linear normalizada)
    #     r_prox = 0.0
    #     if curr_min < self.prox_thresh:
    #         r_prox = - self.k_prox * ((self.prox_thresh - curr_min) / (self.prox_thresh + 1e-6))

    #     # Recompensa por aumento de distÃ¢ncia ao obstÃ¡culo
    #     r_clear = self.k_clear * (curr_min - prev_min)

    #     # Penalidade por TTC baixo
    #     r_ttc = 0.0
    #     if lin_cmd > 1e-3:
    #         ttc = curr_min / (abs(lin_cmd) + 1e-6)
    #         if ttc < self.ttc_thresh:
    #             r_ttc = - self.k_ttc * (1.0 - (ttc / self.ttc_thresh))

    #     # Penalidade frontal: mais forte para obstÃ¡culos Ã  frente
    #     n = self.n_sectors
    #     w = max(1, int(n * 60 / 360))  # janela Â±30Â°
    #     center = n // 2
    #     start = max(0, center - w//2)
    #     end = min(n, center + w//2 + 1)
    #     front_sectors = self.scan_compressed[start:end]
    #     front_min = float(np.min(front_sectors)) if front_sectors.size > 0 else curr_min

    #     r_front = 0.0
    #     if front_min < self.front_thresh:
    #         r_front = - self.k_front * ((self.front_thresh - front_min) / (self.front_thresh + 1e-6))

    #     # ------------------ NEW: obstacle *in path* detection ------------------
    #     # Map curr_angle to sector index (0..n-1). We assume front is at center index.
    #     # sector_width = 2.0 * math.pi / float(n)
    #     # # sector offset from center (rounded)
    #     # sector_offset = int(round(curr_angle / sector_width))
    #     # sector_idx = (center + sector_offset) % n

    #     # new: 08/10 17h41 -------------
    #     # compute sector index corresponding to bearing curr_angle (robot frame)
    #     n = self.n_sectors
    #     angle_min = getattr(self, "scan_angle_min", -math.pi)
    #     angle_max = getattr(self, "scan_angle_max", math.pi)
    #     # curr_angle is relative to robot front in [-pi, pi]
    #     # convert curr_angle into the scan frame which ranges [angle_min, angle_max]
    #     # fraction = (curr_angle - angle_min) / (angle_max - angle_min)
    #     frac = (curr_angle - angle_min) / (angle_max - angle_min + 1e-9)
    #     sector_idx = int(round(frac * (n - 1)))
    #     # clamp
    #     sector_idx = max(0, min(n - 1, sector_idx))
    #     # ------------------------------

    #     # define small window around the sector_idx to check obstruction along goal bearing
    #     window = max(1, int(n * 20 / 360))  # ~Â±10Â°
    #     s0 = max(0, sector_idx - window)
    #     s1 = min(n, sector_idx + window + 1)
    #     path_sectors = self.scan_compressed[s0:s1] if s1 > s0 else np.array([10.0], dtype=np.float32)
    #     path_min = float(np.min(path_sectors))

    #     # condition: obstacle is between robot and goal if min in that bearing is significantly
    #     # smaller than distance to goal (i.e., obstacle closer than the goal), with margin eps
    #     eps = 0.05
    #     blocking_factor = 0.0
    #     if path_min < max(0.01, curr_dist - eps) and path_min < self.prox_thresh:
    #         # normalized factor âˆˆ (0,1]
    #         blocking_factor = max(0.0, (self.prox_thresh - path_min) / (self.prox_thresh + 1e-6))

    #     # Penalidade explÃ­cita por ter obstÃ¡culo bloqueando o caminho direto
    #     k_block = 4.0  # peso do bloqueio (tune)
    #     r_block = - k_block * blocking_factor

    #     # Added 08/10 00:08
    #     if front_min < 0.6 and abs(curr_angle) < math.radians(20):
    #         r_block += -2.0 * (0.6 - front_min)

    #     # Se caminho bloqueado, reduzimos recompensa de heading/forward (forÃ§ar quebra do alinhamento)
    #     if blocking_factor > 0.0:
    #         # reduz heading reward e forward reward proporcionalmente
    #         r_heading = r_heading * (1.0 - 0.9 * blocking_factor)
    #         r_forward = r_forward * (1.0 - 0.95 * blocking_factor)

    #     # Atualiza min_obst anterior
    #     self.prev_min_obst_for_reward = curr_min

    #     # ------------------------------------------------------------------
    #     # Soma total
    #     reward = (
    #         rg + r_heading + r_omega + r_osc + r_step +
    #         r_col + r_goal + r_forward + r_lazy +
    #         r_prox + r_clear + r_ttc + r_front + r_block
    #     )

    #     # Penalidade por spin prolongado
    #     if self.spin_counter > self.spin_counter_max:
    #         reward += self.spin_penalty

    #     # Limite de magnitude
    #     reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

    #     # new: 08/10 17h38
    #     # --- build debug summary for external logging / inspection ---
    #     try:
    #         self.last_debug = {
    #             "front_min": float(front_min),
    #             "path_min": float(path_min),
    #             "blocking_factor": float(blocking_factor),
    #             "r_block": float(r_block),
    #             "r_front": float(r_front),
    #             "r_forward": float(r_forward),
    #             "r_prox": float(r_prox),
    #             "r_clear": float(r_clear),
    #             "r_ttc": float(r_ttc),
    #             "lin_cmd": float(lin_cmd),
    #             "ang_cmd": float(ang_cmd),
    #             "curr_angle": float(curr_angle),
    #             "curr_dist": float(curr_dist),
    #             "curr_min": float(curr_min),
    #             "reward": float(reward)
    #         }
    #     except Exception:
    #         # safe fallback so reward always returns
    #         self.last_debug = None

    #     return reward

    def get_reward(self, action):
        """Refined reward: block-linear penalty + path-clear reward + attenuated forward incentive."""
        # distances
        prev_dist = float(np.linalg.norm(self.goal - self.previous_pose))
        curr_dist = float(np.linalg.norm(self.goal - self.robot_pose))
        rg = self.k_goal * (prev_dist - curr_dist)
        r_lazy = -0.05 if curr_dist >= prev_dist - 1e-3 else 0.0

        # angles
        prev_vec = self.goal - self.previous_pose
        prev_angle = normalize_angle(math.atan2(prev_vec[1], prev_vec[0]) - self.previous_yaw) if np.linalg.norm(prev_vec) > 0 else 0.0
        curr_vec = self.goal - self.robot_pose
        curr_angle = normalize_angle(math.atan2(curr_vec[1], curr_vec[0]) - self.robot_yaw) if np.linalg.norm(curr_vec) > 0 else 0.0

        # heading
        r_heading = self.k_head * (math.cos(curr_angle) - 0.5)

        # actions
        lin_cmd = float(action[0])
        ang_cmd = float(action[1])

        # rotation penalties (small, to allow aggressive turns)
        r_omega = - self.k_omega * abs(ang_cmd)
        prev_ang = float(self.prev_action[1]) if hasattr(self, "prev_action") else 0.0
        r_osc = - self.k_osc * abs(ang_cmd - prev_ang)

        r_step = - self.k_step
        r_col = self.collision_reward if (self.collided or (self.min_obst_dist < self.collision_dist)) else 0.0
        r_goal = self.goal_reward if (curr_dist < self.min_goal_dist) else 0.0

        # ---------- proximity / clearance ----------
        curr_min = float(self.min_obst_dist)
        prev_min = float(getattr(self, "prev_min_obst_for_reward", 10.0))

        r_prox = 0.0
        if curr_min < self.prox_thresh:
            # r_prox = - self.k_prox * ((self.prox_thresh - curr_min) / (self.prox_thresh + 1e-6))
            r_prox = -self.k_prox * math.sqrt(1.0 - (curr_min / self.prox_thresh))

        r_clear = self.k_clear * (curr_min - prev_min)

        r_ttc = 0.0
        if lin_cmd > 1e-3:
            ttc = curr_min / (abs(lin_cmd) + 1e-6)
            if ttc < self.ttc_thresh:
                r_ttc = - self.k_ttc * (1.0 - (ttc / self.ttc_thresh))

        # ---------- frontal window and front_min ----------
        # n = self.n_sectors
        # w = max(1, int(n * 60 / 360))
        # center = n // 2
        # start = max(0, center - w//2)
        # end = min(n, center + w//2 + 1)
        # front_sectors = self.scan_compressed[start:end]
        # front_min = float(np.min(front_sectors)) if front_sectors.size > 0 else curr_min

        # # --- substituir o bloco 'front_sectors' por este ---
        n = self.n_sectors
        angle_min = getattr(self, "scan_angle_min", -math.pi)
        angle_max = getattr(self, "scan_angle_max", math.pi)
        # índice na varredura correspondente a ângulo 0.0 (frente)
        front_frac = (0.0 - angle_min) / (angle_max - angle_min + 1e-9)
        front_idx = int(round(front_frac * (n - 1)))
        front_idx = max(0, min(n - 1, front_idx))

        w = max(1, int(n * 60 / 360))  # janela ±30°
        start = max(0, front_idx - w//2)
        end = min(n, front_idx + w//2 + 1)
        front_sectors = self.scan_compressed[start:end]
        front_min = float(np.min(front_sectors)) if front_sectors.size > 0 else curr_min
        # #

        r_front = 0.0
        if front_min < self.front_thresh:
            r_front = - self.k_front * ((self.front_thresh - front_min) / (self.front_thresh + 1e-6))

        # ---------- path sector (direction to goal) ----------
        angle_min = getattr(self, "scan_angle_min", -math.pi)
        angle_max = getattr(self, "scan_angle_max", math.pi)
        frac = (curr_angle - angle_min) / (angle_max - angle_min + 1e-9)
        sector_idx = int(round(frac * (n - 1)))
        sector_idx = max(0, min(n - 1, sector_idx))

        window = max(1, int(n * 20 / 360))
        s0 = max(0, sector_idx - window)
        s1 = min(n, sector_idx + window + 1)
        path_sectors = self.scan_compressed[s0:s1] if s1 > s0 else np.array([10.0], dtype=np.float32)
        path_min = float(np.min(path_sectors))

        # NEW blocking definition: use ratio 1 - path_min/curr_dist (more continuous)
        # blocking_factor = 0.0
        # if curr_dist > 0.05:
        #     ratio = max(0.0, 1.0 - (path_min / (curr_dist + 1e-6)))
        #     # only consider blocking if path_min smaller than some fraction of curr_dist or small absolute
        #     if path_min < curr_dist * 0.9 or path_min < self.prox_thresh:
        #         blocking_factor = float(min(1.0, ratio))

        blocking_factor = 0.0
        # Limiar de distância para considerar um bloqueio sério (ex: 0.7 metros)
        is_obstacle_in_path = (abs(path_min - curr_min) < 0.1 * curr_min)
        is_dangerously_close = (path_min < curr_dist and path_min < self.front_thresh)
        if is_obstacle_in_path and is_dangerously_close:
            ratio = 1.0 - (path_min / self.front_thresh)
            blocking_factor = min(1.0, math.sqrt(math.sqrt(math.sqrt(ratio))))

        # ADDED
        rg *= (1.0 - blocking_factor)

        # block penalties
        r_block = - self.k_block * blocking_factor
        r_lin_block = - self.k_block_lin * lin_cmd * blocking_factor

        # reward for increasing path clearance (encoraja virar cedo)
        prev_path = float(getattr(self, "prev_path_min_for_reward", 10.0))
        r_path_clear = self.k_path_clear * (path_min - prev_path)
        self.prev_path_min_for_reward = path_min

        # attenuate forward reward by blocking_factor (smooth)
        heading_cos = max(0.0, math.cos(curr_angle))
        block_scale = min(1.0, blocking_factor * 1.4)
        r_forward = self.k_forward * lin_cmd * heading_cos * (1.0 - block_scale)
        if front_min < 0.5:
            r_forward *= max(0.0, (front_min / 0.5))

        # update prev min
        self.prev_min_obst_for_reward = curr_min

        # --- Penalidade de Estagnação ---
        r_stall = 0.0
        if front_min < self.prox_thresh and abs(lin_cmd) < 0.05:
            r_stall = -0.1  # Pequena penalidade constante por estar parado perto de um obstáculo

        # --- Penalidade de Teimosia ---
        r_commit = 0.0
        # Se o caminho está significativamente bloqueado E o agente insiste em ir reto e rápido...
        if blocking_factor > 0.75 and abs(ang_cmd) < 0.2 and lin_cmd > 0.2:
            # ...aplique uma forte penalidade proporcional ao bloqueio e à velocidade.
            r_commit = -self.k_commit * blocking_factor * lin_cmd

        # sum
        reward = (
            rg + r_heading + r_omega + r_osc + r_step +
            r_col + r_goal + r_forward + r_lazy +
            r_prox + r_clear + r_ttc + r_front + r_block + r_lin_block + 
            r_path_clear + r_stall + r_commit
        )

        if blocking_factor > 0.6 and lin_cmd > 0.05 and abs(ang_cmd) < 0.25:
            # penalidade forte para "ir reto" quando caminho bloqueado
            reward += - (self.k_block_lin * 2.0) * blocking_factor * lin_cmd

        # incentivo à manobra: recompensar virar quando caminho bloqueado
        if blocking_factor > 0.2:
            # recompensa proporcional ao componente angular que reduz o bloqueio (heurística)
            # aqui usamos magnitude absoluta do ang_cmd (o agente deve virar)
            r_turn_encourage = 0.6 * blocking_factor * min(abs(ang_cmd), 1.0)
            reward += r_turn_encourage

        # spin penalty
        if self.spin_counter > self.spin_counter_max:
            reward += self.spin_penalty

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # debug info
        try:
            self.last_debug = {
                "front_min": float(front_min),
                "path_min": float(path_min),
                "blocking_factor": float(blocking_factor),
                "r_block": float(r_block),
                "r_lin_block": float(r_lin_block),
                "r_front": float(r_front),
                "r_forward": float(r_forward),
                "r_prox": float(r_prox),
                "r_path_clear": float(r_path_clear),
                "lin_cmd": float(lin_cmd),
                "ang_cmd": float(ang_cmd),
                "curr_angle": float(curr_angle),
                "curr_dist": float(curr_dist),
                "curr_min": float(curr_min),
                "reward": float(reward)
            }
        except Exception:
            self.last_debug = None

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