import math
import time
from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_srvs.srv import Empty
import gymnasium as gym
from gymnasium import spaces
import tf2_ros

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
                 n_sectors=36,
                 max_linear=0.7,
                 max_w=1.0,
                 min_goal_dist=0.2,
                 collision_dist=0.2,
                 max_steps=2000,
                 step_wait_time=0.1,
                 lookahead_distance=0.8):
        Node.__init__(self, 'ppo_env')
        gym.Env.__init__(self)

        # force /clock sync
        try:
            self.declare_parameter('use_sim_time', True)
            from rclpy.parameter import Parameter
            try:
                self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
            except Exception:
                self.get_logger().debug("Could not set use_sim_time parameter programmatically.")
        except Exception:
            pass

        # === Parameters ===
        self.n_sectors = int(n_sectors)
        self.max_lin = float(max_linear)
        self.max_w = float(max_w)
        self.min_goal_dist = float(min_goal_dist)
        self.collision_dist = float(collision_dist)
        self.max_steps = int(max_steps)
        self.step_wait_time = float(step_wait_time)
        self.lookahead_distance = float(lookahead_distance)

        # === State ===
        self.scan = np.array([], dtype=np.float32)
        self.scan_compressed = np.full((self.n_sectors,), 10.0, dtype=np.float32)
        self.previous_pose = np.zeros(2, dtype=np.float32)
        self.robot_pose = np.zeros(2, dtype=np.float32)
        self.previous_yaw = 0.0
        self.robot_yaw = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.current_plan = None
        self.lookahead_rel = np.zeros(2, dtype=np.float32)
        self.step_number = 0
        self.min_obst_dist = 10.0
        self.done = False
        self.collided = False
        self.delta_d = 0.0

        # thresholds
        self.front_thresh = 0.8  # metros
        self.prox_thresh = 0.5   # metros

        # === Reward Weights (Versão Consolidada) ===
        self.k_goal_far = 20.0       # Recompensa padrão (FAR ZONE)
        self.k_goal_near = 40.0      # Recompensa MAIOR (NEAR ZONE)
        self.goal_near_thresh = 1.5  # Limite para "perto" (metros)

        self.k_head = 0.1            # Recompensa por alinhamento puro (baixa)
        self.k_forward_align = 8.0   # Recompensa por mover ALINHADO
        self.k_plan_align = 0.8
        self.k_lat = 0.5

        self.k_reverse = 2.0         # Penalidade por usar ré
        self.k_stuck_aligned = 5.0   # Penalidade por estar alinhado mas PARADO
        self.k_prox = 3.0
        self.k_front = 6.0
        self.k_omega = 0.20
        self.k_step = 0.002

        self.stuck_align_angle = 0.25
        self.stuck_progress_thresh = 0.001

        self.k_head_near = 2.5       # Recompensa FORTE por alinhamento fino (NEAR ZONE)
        self.max_lin_near = 0.3      # Velocidade linear máxima desejada (NEAR ZONE)
        self.k_slow_down = 5.0       # Penalidade por exceder max_lin_near (NEAR ZONE)

        self.reward_clip = 30.0
        self.collision_reward = -20.0
        self.goal_reward = 30.0

        # === Movement smoothing ===
        self.smooth_alpha_lin = 0.1
        self.smooth_alpha_ang = 0.5
        self.prev_action = (0.0, 0.0)

        # === TF Buffer ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # === ROS2 I/O ===
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- CORREÇÃO: Clientes ROS2 no __init__ ---
        self.respawn_robot_client = self.create_client(Empty, 'respawn_robot')
        self.get_logger().info("Aguardando serviço 'respawn_robot'...")
        while not self.respawn_robot_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Serviço "respawn_robot" não disponível, esperando...')
            rclpy.spin_once(self, timeout_sec=0.5) 
        self.get_logger().info("✅ Serviço 'respawn_robot' encontrado.")
        
        self.spawn_new_goal_client = self.create_client(Empty, 'spawn_new_goal')
        self.get_logger().info("Aguardando serviço 'spawn_new_goal'...")
        while not self.spawn_new_goal_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Serviço "spawn_new_goal" não disponível, esperando...')
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info("✅ Serviço 'spawn_new_goal' encontrado.")
        # --- FIM DA CORREÇÃO ---

        if HAVE_BUMPER_MSG:
            self.bumper_sub = self.create_subscription(ContactsState, '/bumper_states', self.bumper_callback, 10)
        else:
            self.bumper_sub = None

        # === Spaces ===
        obs_dim = self.n_sectors + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-0.3, -self.max_w], dtype=np.float32),
            high=np.array([self.max_lin, self.max_w], dtype=np.float32),
            dtype=np.float32
        )

        # --- CORREÇÃO: Timestamps para espera ---
        self._last_goal_stamp = 0.0
        self._last_odom_stamp = 0.0
        self._last_scan_stamp = 0.0
        self.last_front_min = 10.0
        self.get_logger().info("✅ PPOEnvironment initialized")

    # === Callbacks ===
    def scan_callback(self, msg: LaserScan):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan[np.isinf(scan)] = 10.0
        scan[np.isnan(scan)] = 10.0
        self.scan = scan
        self.scan_compressed = compress_scan(scan, self.n_sectors)
        self.min_obst_dist = float(np.min(scan)) if scan.size > 0 else 10.0
        self.scan_angle_min = getattr(msg, "angle_min", -math.pi)
        self.scan_angle_max = getattr(msg, "angle_max", math.pi)

        n = self.n_sectors
        ang_min, ang_max = self.scan_angle_min, self.scan_angle_max
        ang_range = ang_max - ang_min
        center_idx = int(round((0.0 - ang_min) / ang_range * (n - 1)))
        half_w = max(1, int(round(0.1 * n)))
        start, end = max(0, center_idx - half_w), min(n, center_idx + half_w + 1)
        self.last_front_min = float(np.min(self.scan_compressed[start:end]))

        self._last_scan_stamp = time.time() # Atualiza o timestamp

    def odom_callback(self, msg):
        # --- CORREÇÃO: Não atualiza 'previous_pose' aqui ---
        # self.previous_pose = self.robot_pose.copy()
        # self.previous_yaw = float(self.robot_yaw)

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_pose = np.array([pos.x, pos.y], dtype=np.float32)
        self.robot_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        self._last_odom_stamp = time.time() # Atualiza o timestamp

    def goal_callback(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)
        self._last_goal_stamp = time.time() # Atualiza o timestamp

    def bumper_callback(self, msg):
        if hasattr(msg, "states") and len(msg.states) > 0:
            self.collided = True

    def plan_callback(self, msg: Path):
        if msg and len(msg.poses) > 0:
            self.current_plan = msg
        else:
            self.current_plan = None

    # === Lookahead Computation ===
    def compute_lookahead(self):
        if self.current_plan is None or len(self.current_plan.poses) == 0:
            return np.zeros(2, dtype=np.float32)
        min_idx, min_dist = 0, float('inf')
        for i, pose in enumerate(self.current_plan.poses):
            px, py = pose.pose.position.x, pose.pose.position.y
            d = math.hypot(px - self.robot_pose[0], py - self.robot_pose[1])
            if d < min_dist:
                min_idx, min_dist = i, d
        look_pt = self.current_plan.poses[-1].pose
        for j in range(min_idx, len(self.current_plan.poses)):
            ps = self.current_plan.poses[j].pose
            dx, dy = ps.position.x - self.robot_pose[0], ps.position.y - self.robot_pose[1]
            if math.hypot(dx, dy) >= self.lookahead_distance:
                look_pt = ps
                break
        dx, dy = look_pt.position.x - self.robot_pose[0], look_pt.position.y - self.robot_pose[1]
        yaw = self.robot_yaw
        x_rel = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        y_rel = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        self.lookahead_rel = np.array([x_rel, y_rel], dtype=np.float32)
        return self.lookahead_rel

    # === get_state ===
    def get_state(self):
        sectors = np.clip(self.scan_compressed, 0.0, 10.0) / 10.0

        goal_vec = self.goal - self.robot_pose
        dist = float(np.linalg.norm(goal_vec))
        max_dist_norm = 8.5
        dist_norm = np.clip(dist / max_dist_norm, 0.0, 1.0)

        curr_angle = 0.0
        if dist > 1e-6:
            curr_angle = normalize_angle(math.atan2(goal_vec[1], goal_vec[0]) - float(self.robot_yaw))
        angle_sin = math.sin(curr_angle)
        angle_cos = math.cos(curr_angle)

        min_norm = np.clip(self.min_obst_dist, 0.0, 10.0) / 10.0

        # front and path normalization
        n = self.n_sectors
        angle_min = getattr(self, "scan_angle_min", -math.pi)
        angle_max = getattr(self, "scan_angle_max", math.pi)
        ang_range = angle_max - angle_min if (angle_max - angle_min) != 0.0 else (2.0 * math.pi)
        front_frac = (0.0 - angle_min) / (ang_range + 1e-9)
        front_idx = int(round(front_frac * (n - 1)))
        front_idx = max(0, min(n - 1, front_idx))
        w = max(1, int(n * 60 / 360))
        start = max(0, front_idx - w//2)
        end = min(n, front_idx + w//2 + 1)
        front_min = float(np.min(self.scan_compressed[start:end])) if self.scan_compressed.size > 0 else 10.0
        front_min_norm = np.clip(front_min, 0.0, 10.0) / 10.0

        frac = (curr_angle - angle_min) / (ang_range + 1e-9)
        sector_idx = int(round(frac * (n - 1)))
        sector_idx = max(0, min(n - 1, sector_idx))
        window = max(1, int(n * 20 / 360))
        s0 = max(0, sector_idx - window)
        s1 = min(n, sector_idx + window + 1)
        path_min = float(np.min(self.scan_compressed[s0:s1])) if self.scan_compressed.size > 0 else 10.0
        path_min_norm = np.clip(path_min, 0.0, 10.0) / 10.0

        la = self.compute_lookahead()
        lookahead_x_norm = np.clip(la[0] / (self.lookahead_distance + 1e-6), -1.0, 1.0)
        lookahead_y_norm = np.clip(la[1] / (self.lookahead_distance + 1e-6), -2.0, 2.0)

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

    # === step ===
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        lin = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        ang = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])

        lin = self.smooth_alpha_lin * self.prev_action[0] + (1 - self.smooth_alpha_lin) * lin
        ang = self.smooth_alpha_ang * self.prev_action[1] + (1 - self.smooth_alpha_ang) * ang

        twist = Twist()
        twist.linear.x, twist.angular.z = lin, ang
        self.cmd_pub.publish(twist)
        self.get_logger().debug(
            f"[CMD_PUB] lin={lin:.3f}, ang={ang:.3f} → sent to /cmd_vel"
        )
        self.prev_action = (lin, ang)
        self.step_number += 1 # O step 1 agora é 1

        t0 = time.time()
        while time.time() - t0 < self.step_wait_time:
            rclpy.spin_once(self, timeout_sec=0.01)

        reward = self.get_reward((lin, ang))
        terminated, truncated = False, False

        # --- INÍCIO DA CORREÇÃO ---
        dist_to_goal = np.linalg.norm(self.goal - self.robot_pose)

        if self.collided or self.min_obst_dist < self.collision_dist:
            terminated = True
            reward += self.collision_reward
            self.get_logger().debug(f"Episódio terminado por colisão. Step: {self.step_number}")

        elif dist_to_goal < self.min_goal_dist:
            # SÓ termine se NÃO for o primeiro step (para evitar o bug do 0,0)
            if self.step_number > 1:
                terminated = True
                reward += self.goal_reward
                self.get_logger().debug(f"Episódio terminado por GOAL. Step: {self.step_number}")
            else:
                # É o step 1 e já "estamos no goal" (bug de 0,0)
                # Ignore a terminação e continue o episódio.
                self.get_logger().debug(
                    "Condição de goal ignorada no Step 1 para evitar loop de reset."
                )
        
        elif self.step_number >= self.max_steps:
            truncated = True
            self.get_logger().debug(f"Episódio truncado por max_steps. Step: {self.step_number}")
        # --- FIM DA CORREÇÃO ---
            
        obs = self.get_state()
        info = {"collision": self.collided}

        # Atualiza o previous_pose para o PRÓXIMO step
        self.previous_pose = self.robot_pose.copy()
        self.previous_yaw = float(self.robot_yaw)

        if self.step_number % 50 == 0:
            self.get_logger().info(
                f"[TRAIN] step={self.step_number} Δd={self.delta_d:.3f} lin={lin:.3f} ang={ang:.3f}"
            )

        return obs, reward, terminated, truncated, info

    # === reward ===
    def get_reward(self, action):
        """
        Função de recompensa consolidada (Near Zone, penalidades, etc.)
        """
        lin_applied = float(action[0])
        ang_cmd = float(action[1])

        # Com a correção no 'step', 'self.previous_pose' é do início do step
        # e 'self.robot_pose' é do fim do step.
        prev_dist = float(np.linalg.norm(self.goal - self.previous_pose))
        curr_dist = float(np.linalg.norm(self.goal - self.robot_pose))

        goal_vec = self.goal - self.robot_pose
        if np.linalg.norm(goal_vec) > 1e-6:
            raw_angle = math.atan2(goal_vec[1], goal_vec[0]) - float(self.robot_yaw)
            curr_angle = normalize_angle(raw_angle)
        else:
            curr_angle = 0.0

        # Progresso Real (Projeção)
        disp_vec = self.robot_pose - self.previous_pose
        goal_vec_prev = self.goal - self.previous_pose
        goal_dist_prev = np.linalg.norm(goal_vec_prev)

        if goal_dist_prev > 1e-6:
            goal_unit = goal_vec_prev / goal_dist_prev
            delta_proj = float(np.dot(disp_vec, goal_unit))
        else:
            delta_proj = 0.0
        self.delta_d = float(delta_proj) # Salva para log

        # Sensor
        front_min = getattr(self, "last_front_min", self.min_obst_dist)

        # Recompensa de Progresso
        r_goal = 0.0
        if curr_dist < self.goal_near_thresh:
            r_goal = self.k_goal_near * delta_proj # Ponderação alta
        else:
            r_goal = self.k_goal_far * delta_proj  # Ponderação padrão

        # Lógica de Recompensa Dividida (Far vs Near)
        if curr_dist > self.goal_near_thresh:
            # --- ZONA "LONGE" (FAR ZONE) ---
            r_forward_align = self.k_forward_align * (lin_applied / max(1e-3, self.max_lin)) * max(0.0, math.cos(curr_angle))

            la = getattr(self, "lookahead_rel", np.array([0.0, 0.0], dtype=np.float32))
            la_x_norm = np.clip(la[0] / (self.lookahead_distance + 1e-6), -1.0, 1.0)
            r_plan = self.k_plan_align * max(0.0, la_x_norm) * (max(0.0, lin_applied) / max(1e-3, self.max_lin))

            front_clear = 1.0 if front_min > max(0.5, self.front_thresh) else (front_min / max(0.001, self.front_thresh))
            r_speed = 1.0 * (lin_applied / max(1e-3, self.max_lin)) * front_clear
            
            r_lat = -self.k_lat * abs(la[1])
            r_heading = self.k_head * math.cos(curr_angle)
            
            r_stuck_aligned = 0.0
            is_well_aligned = abs(curr_angle) < self.stuck_align_angle
            is_not_progressing = delta_proj < self.stuck_progress_thresh
            if is_well_aligned and is_not_progressing and lin_applied > 0.05:
                r_stuck_aligned = -self.k_stuck_aligned
            
            r_slow_down = 0.0

        else:
            # --- ZONA "PERTO" (NEAR ZONE) ---
            r_forward_align = 0.0 
            r_plan = 0.0
            r_speed = 0.0
            r_lat = 0.0
            r_stuck_aligned = 0.0 

            alignment_bonus = (math.cos(curr_angle) + 1.0) / 2.0
            r_heading = self.k_head_near * (alignment_bonus - 0.5)

            r_slow_down = 0.0
            if lin_applied > self.max_lin_near:
                r_slow_down = -self.k_slow_down * (lin_applied - self.max_lin_near)

        # --- Penalidades (Sempre Ativas) ---
        r_reverse = -self.k_reverse * abs(lin_applied) if lin_applied < -0.01 else 0.0

        curr_min = float(self.min_obst_dist)
        r_prox = -self.k_prox * (1.0 - curr_min / self.prox_thresh) if curr_min < self.prox_thresh else 0.0
        r_front = -self.k_front * (1.0 - front_min / self.front_thresh) if front_min < self.front_thresh else 0.0

        r_smooth = -self.k_omega * abs(ang_cmd)
        r_spin = -1.0 if abs(lin_applied) < 0.05 and abs(ang_cmd) > 0.7 else 0.0
        r_step = -self.k_step

        # Recompensas terminais (adicionadas no 'step', mas podemos zerar aqui)
        r_col = 0.0 # self.collision_reward if ...
        r_goal_done = 0.0 # self.goal_reward if ...

        # Soma final
        reward = (r_goal + r_forward_align + r_plan + r_heading + r_speed +
                  r_stuck_aligned + r_lat + r_slow_down +
                  r_reverse + r_smooth + r_prox + r_front + r_spin + r_step +
                  r_col + r_goal_done)

        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

    # === reset ===
    def reset(self, seed=None, options=None):
        """
        [VERSÃO CONSOLIDADA E CORRIGIDA]
        """
        super().reset(seed=seed)
        self.collided = False
        self.step_number = 0
        self.get_logger().debug("Iniciando reset do ambiente...")
        self._publish_stop() 

        t_reset_start = time.time()

        # --- Chamada de Serviço "respawn_robot" (Robusta) ---
        req = Empty.Request()
        spawn_success = False
        while not spawn_success:
            self.get_logger().info("Tentando spawnar o robô...")
            future = self.respawn_robot_client.call_async(req)
            
            t_wait_start_inner = time.time()
            timeout_sec = 10.0
            
            while time.time() - t_wait_start_inner < timeout_sec:
                rclpy.spin_once(self, timeout_sec=0.01)
                if future.done():
                    try:
                        future.result() 
                        spawn_success = True 
                        self.get_logger().info("✅ Robô spawnado com sucesso.")
                    except Exception as e:
                        self.get_logger().warn(f"Serviço 'respawn_robot' falhou: {e}. Tentando novamente em 2s...")
                        t_wait_err = time.time()
                        while time.time() - t_wait_err < 2.0:
                            rclpy.spin_once(self, timeout_sec=0.01)
                    break 
            else:
                self.get_logger().warn(f"Timeout de {timeout_sec}s esperando 'respawn_robot'. Tentando novamente...")
                future.cancel()

        # --- CORREÇÃO: Chamada "Fire-and-Forget" para "spawn_new_goal" ---
        self.get_logger().info("Disparando 'spawn_new_goal'...")
        self.spawn_new_goal_client.call_async(req)
        # Não esperamos pelo 'future' aqui. A função _wait_for_initial_sensors_
        # vai esperar pelo *resultado* (o goal_callback ser chamado).
        # --- FIM DA CORREÇÃO ---

        # --- Espera por Sensores (Corrigida) ---
        self.get_logger().debug("Aguardando sensores (goal, odom, scan)...")
        if not self._wait_for_initial_sensors(t_reset_start, timeout=5.0):
            self.get_logger().error("FALHA CRÍTICA: Timeout esperando sensores (odom, scan ou goal).")
            # Mesmo em falha, retornamos um estado, mas o log de erro indicará o problema.
        
        # --- Espera de estabilização ---
        t_wait_stable = time.time()
        while time.time() - t_wait_stable < 0.2: 
            rclpy.spin_once(self, timeout_sec=0.01)

        # Seta o 'previous_pose' inicial para o 'reset'
        self.previous_pose = self.robot_pose.copy()
        self.previous_yaw = float(self.robot_yaw)

        # --- Espera por TF ---
        tf_wait_t0 = time.time()
        tf_timeout = 8.0
        tf_ok = False
        self.get_logger().debug("Aguardando TF map->base_link...")
        while time.time() - tf_wait_t0 < tf_timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            try:
                self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                tf_ok = True
                break
            except Exception as e:
                pass 

        if not tf_ok:
            self.get_logger().warn("Timeout em TF map->base_link no reset.")
        else:
            self.get_logger().debug("TF disponível.")

        self.get_logger().info("✅ Reset do ambiente concluído. Iniciando novo episódio.")
        time.sleep(2.0)
        return self.get_state(), {}

    def _wait_for_initial_sensors(self, t_reset_start, timeout=8.0):
        t0 = time.time()
        self.get_logger().debug("Aguardando sensores iniciais (Scan, Goal, Odom)...")
        
        scan_ok = False
        goal_ok = False
        odom_ok = False
        
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)

            # Verifica se os timestamps dos callbacks são MAIS NOVOS que o início do reset
            if not scan_ok and self._last_scan_stamp > t_reset_start:
                scan_ok = True
                self.get_logger().debug("... Scan OK.")
            
            if not goal_ok and self._last_goal_stamp > t_reset_start:
                goal_ok = True
                self.get_logger().debug("... Goal OK.")
            
            if not odom_ok and self._last_odom_stamp > t_reset_start:
                odom_ok = True
                self.get_logger().debug("... Odom OK.")

            if scan_ok and goal_ok and odom_ok:
                self.get_logger().debug("Sensores OK.")
                return True
        
        self.get_logger().warn(f"Timeout em _wait_for_initial_sensors. Status: [Scan: {scan_ok}, Goal: {goal_ok}, Odom: {odom_ok}]")
        return False

    def _publish_stop(self, n=3):
        msg = Twist()
        for _ in range(n):
            self.cmd_pub.publish(msg)
            t_wait_start = time.time()
            while time.time() - t_wait_start < 0.02: # Espera 20ms
                rclpy.spin_once(self, timeout_sec=0.001)

    def render(self, mode="human"):
        pass