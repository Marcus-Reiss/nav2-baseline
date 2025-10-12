# ppo_training/evaluator.py
import os
import sys
import time
import math
import argparse
import threading
from datetime import datetime
import csv

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ament_index_python.packages import get_package_share_directory

import numpy as np
from stable_baselines3 import PPO

from .env_wrapper import PPOEnvironment  # seu env compatÃ­vel gymnasium


class PPOEvaluator(Node):
    def __init__(self, model_path: str, episodes: int = 10, deterministic: bool = True,
                 step_sleep: float = 0.0, save_results: bool = True):
        super().__init__('ppo_evaluator')

        self.model_path = model_path
        self.episodes = int(episodes)
        self.deterministic = bool(deterministic)
        self.step_sleep = float(step_sleep)
        self.save_results = bool(save_results)

        # Executor + env node (para que callbacks ROS funcionem)
        self._executor = MultiThreadedExecutor()
        self.env_node = PPOEnvironment()
        self._executor.add_node(self.env_node)

        # Start executor thread
        self._exec_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._exec_thread.start()
        self.get_logger().info("Executor iniciado em background (env callbacks ativos).")

        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo nÃ£o encontrado: {self.model_path}")
        self.model = PPO.load(self.model_path)
        self.get_logger().info(f"Modelo carregado: {self.model_path}")

        # Where to save results
        pkg_share = get_package_share_directory('ppo_training')
        self.models_dir = os.path.join(pkg_share, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # storage
        self.results = []

    def run(self):
        successes = 0
        collisions = 0
        total_steps = 0
        total_reward = 0.0

        self.get_logger().info(f"Iniciando avaliaÃ§Ã£o: episodes={self.episodes}, deterministic={self.deterministic}")

        # --- prepare debug CSV ---
        ts_debug = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_csv_path = os.path.join(self.models_dir, f"eval_debug_{ts_debug}.csv")
        debug_f = open(debug_csv_path, 'w', newline='')
        debug_writer = csv.writer(debug_f)
        debug_writer.writerow([
            'episode', 'step',
            'lin_cmd', 'ang_cmd',
            'curr_dist', 'curr_angle_deg',
            'curr_min', 'front_min', 'path_min', 'blocking_factor',
            'r_forward', 'r_block', 'r_front', 'r_prox', 'reward'
        ])
        self.get_logger().info(f"[Eval] Debug CSV: {debug_csv_path}")

        try:
            for ep in range(1, self.episodes + 1):
                obs, info = self.env_node.reset()
                ep_steps = 0
                ep_reward = 0.0
                ep_success = False
                ep_collision = False

                start_time = time.time()
                while True:
                    # Predict action
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    # Step env
                    obs, reward, terminated, truncated, info = self.env_node.step(action)

                    ep_steps += 1
                    ep_reward += float(reward)

                    # --- write debug info ---
                    debug = getattr(self.env_node, 'last_debug', None)
                    if debug is not None:
                        debug_writer.writerow([
                            ep, ep_steps,
                            debug.get('lin_cmd', np.nan),
                            debug.get('ang_cmd', np.nan),
                            debug.get('curr_dist', np.nan),
                            math.degrees(debug.get('curr_angle', np.nan))
                                if debug.get('curr_angle') is not None else np.nan,
                            debug.get('curr_min', np.nan),
                            debug.get('front_min', np.nan),
                            debug.get('path_min', np.nan),
                            debug.get('blocking_factor', np.nan),
                            debug.get('r_forward', np.nan),
                            debug.get('r_block', np.nan),
                            debug.get('r_front', np.nan),
                            debug.get('r_prox', np.nan),
                            float(reward)
                        ])

                    if self.step_sleep > 0.0:
                        time.sleep(self.step_sleep)

                    # Termination check
                    if terminated or truncated:
                        try:
                            dist_to_goal = np.linalg.norm(self.env_node.goal - self.env_node.robot_pose)
                            if dist_to_goal < self.env_node.min_goal_dist:
                                ep_success = True
                                successes += 1
                            elif self.env_node.min_obst_dist < self.env_node.collision_dist:
                                ep_collision = True
                                collisions += 1
                        except Exception:
                            pass

                        total_steps += ep_steps
                        total_reward += ep_reward

                        self.get_logger().info(
                            f"[Eval] Ep {ep}/{self.episodes} | steps={ep_steps}, reward={ep_reward:.2f}, "
                            f"success={ep_success}, collision={ep_collision}"
                        )

                        # Stop robot
                        try:
                            from geometry_msgs.msg import Twist
                            stop = Twist()
                            stop.linear.x = 0.0
                            stop.angular.z = 0.0
                            self.env_node.cmd_pub.publish(stop)
                        except Exception:
                            pass

                        self.results.append({
                            'episode': ep,
                            'steps': ep_steps,
                            'reward': float(ep_reward),
                            'success': bool(ep_success),
                            'collision': bool(ep_collision),
                            'time': time.time() - start_time
                        })
                        break

            # --- summary ---
            success_rate = successes / self.episodes if self.episodes > 0 else 0.0
            avg_steps = total_steps / self.episodes if self.episodes > 0 else 0.0
            avg_reward = total_reward / self.episodes if self.episodes > 0 else 0.0

            summary = {
                'episodes': self.episodes,
                'successes': successes,
                'collisions': collisions,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_reward': avg_reward
            }
            self.get_logger().info(f"[Eval] Summary: {summary}")

            # Save main results
            if self.save_results:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(self.models_dir, f"eval_results_{ts}.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'reward', 'success', 'collision', 'time'])
                    writer.writeheader()
                    for r in self.results:
                        writer.writerow(r)
                self.get_logger().info(f"[Eval] Results saved to {csv_path}")

            return summary

        finally:
            # Close debug CSV
            try:
                debug_f.close()
                self.get_logger().info(f"[Eval] Debug CSV saved: {debug_csv_path}")
            except Exception:
                pass

    def shutdown(self):
        self.get_logger().info("Shutting down evaluator...")
        try:
            self._executor.remove_node(self.env_node)
        except Exception:
            pass
        try:
            self.env_node.destroy_node()
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        self.get_logger().info("Evaluator shutdown complete.")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default='', help='Caminho para ppo_stageX.zip (se vazio, usa ppo_stage3.zip do pacote)')
    parser.add_argument('--episodes', type=int, default=10, help='NÃºmero de episÃ³dios de avaliaÃ§Ã£o')
    parser.add_argument('--deterministic', action='store_true', help='Usar aÃ§Ãµes determinÃ­sticas')
    parser.add_argument('--step_sleep', type=float, default=0.0, help='Sleep entre steps (s) â€” Ãºtil para visualizaÃ§Ã£o em Gazebo')
    parser.add_argument('--no-save', dest='save_results', action='store_false', help='NÃ£o salvar CSV de resultados')
    args, unknown = parser.parse_known_args(argv)

    rclpy.init(args=unknown)

    if args.model:
        model_path = args.model
    else:
        pkg_share = get_package_share_directory('ppo_training')
        model_path = os.path.join(pkg_share, 'models', 'ppo_stage3.zip')

    evaluator = PPOEvaluator(model_path=model_path,
                             episodes=args.episodes,
                             deterministic=args.deterministic,
                             step_sleep=args.step_sleep,
                             save_results=args.save_results)

    try:
        summary = evaluator.run()
    except KeyboardInterrupt:
        evaluator.get_logger().warn("Interrupted by user (KeyboardInterrupt)")
    finally:
        evaluator.shutdown()
        rclpy.shutdown()

    print("EVALUATION SUMMARY:", summary)


if __name__ == '__main__':
    main(sys.argv[1:])