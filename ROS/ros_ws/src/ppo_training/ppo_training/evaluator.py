# ppo_training/evaluator.py
import os
import sys
import time
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

from .env_wrapper import PPOEnvironment  # seu env compatível gymnasium


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
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
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

        self.get_logger().info(f"Iniciando avaliação: episodes={self.episodes}, deterministic={self.deterministic}")

        for ep in range(1, self.episodes + 1):
            # reset returns (obs, info)
            obs, info = self.env_node.reset()
            ep_steps = 0
            ep_reward = 0.0
            ep_success = False
            ep_collision = False

            start_time = time.time()
            while True:
                # model.predict espera um array (obs)
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

                # step publica cmd_vel internamente (env_node.step)
                obs, reward, terminated, truncated, info = self.env_node.step(action)

                ep_steps += 1
                ep_reward += float(reward)

                if self.step_sleep > 0.0:
                    time.sleep(self.step_sleep)  # opcional, para visualização em tempo real

                if terminated or truncated:
                    # decide se foi sucesso ou colisão
                    # sucesso: distância ao goal menor que min_goal_dist
                    try:
                        dist_to_goal = np.linalg.norm(self.env_node.goal - self.env_node.robot_pose)
                        if dist_to_goal < self.env_node.min_goal_dist:
                            ep_success = True
                            successes += 1
                        elif self.env_node.min_obst_dist < self.env_node.collision_dist:
                            ep_collision = True
                            collisions += 1
                        else:
                            # terminado por outra razão (por segurança consideramos truncation)
                            if truncated:
                                # truncation (timeout): nem sucesso nem colisão
                                pass
                    except Exception:
                        # se não for possível determinar, apenas marque truncation
                        pass

                    total_steps += ep_steps
                    total_reward += ep_reward

                    self.get_logger().info(
                        f"[Eval] Episode {ep}/{self.episodes} -> steps={ep_steps}, reward={ep_reward:.3f}, "
                        f"success={ep_success}, collision={ep_collision}"
                    )

                    # stop robot: publicar zero velocity para segurança
                    try:
                        from geometry_msgs.msg import Twist
                        stop = Twist()
                        stop.linear.x = 0.0
                        stop.angular.z = 0.0
                        self.env_node.cmd_pub.publish(stop)
                    except Exception:
                        pass

                    # save episode result
                    self.results.append({
                        'episode': ep,
                        'steps': ep_steps,
                        'reward': float(ep_reward),
                        'success': bool(ep_success),
                        'collision': bool(ep_collision),
                        'time': time.time() - start_time
                    })
                    break

        # summary
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

        # save results
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

    def shutdown(self):
        # stop executor and destroy env node
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
    parser.add_argument('--episodes', type=int, default=10, help='Número de episódios de avaliação')
    parser.add_argument('--deterministic', action='store_true', help='Usar ações determinísticas')
    parser.add_argument('--step_sleep', type=float, default=0.0, help='Sleep entre steps (s) — útil para visualização em Gazebo')
    parser.add_argument('--no-save', dest='save_results', action='store_false', help='Não salvar CSV de resultados')
    args, unknown = parser.parse_known_args(argv)

    rclpy.init(args=unknown)

    # default model path: ppo_stage3.zip no pacote
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

    # print summary to stdout
    print("EVALUATION SUMMARY:", summary)


if __name__ == '__main__':
    main(sys.argv[1:])
    # to run:
    # ros2 run ppo_training evaluate -- --model /home/ROS/ros_ws/install/ppo_training/share/ppo_training/models/ppo_stage3.zip --episodes 10 --deterministic --step_sleep 0.02
