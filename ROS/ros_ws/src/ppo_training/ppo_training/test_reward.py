#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rclpy
from ppo_training.env_wrapper import PPOEnvironment

def run_reward_test():
    env = PPOEnvironment()
    print("\n=== TESTE DE RECOMPENSA PPO ===")

    # Configuração inicial simulada
    env.prev_dist = 2.0      # distância anterior (m)
    env.path_min = 1.0       # caminho livre moderado (m)
    env.front_min = 0.8
    env.curr_min = 0.7
    env.prox_thresh = 0.4
    env.blocking_factor = 0.0

    test_cases = [
        # curr_dist, lin_cmd, ang_cmd, desc
        (1.5, 0.2, 0.0, "Progresso bom para o goal (sem obstáculo)"),
        (2.5, 0.2, 0.0, "Se afastando do goal (recompensa deve ser negativa)"),
        (1.0, 0.2, 0.0, "Progresso grande (recompensa deve ser alta)"),
        (1.0, 0.0, 0.5, "Rotação parada (deve ser quase zero)"),
    ]

    for curr_dist, lin_cmd, ang_cmd, desc in test_cases:
        env.prev_path_min = env.path_min
        env.curr_dist = curr_dist
        env.prev_dist = 2.0  # distância anterior fixa (simula passo anterior)
        env.path_min = 1.0
        env.front_min = 0.8
        env.curr_min = 0.7
        env.blocking_factor = 0.0

        r = env.get_reward(lin_cmd, ang_cmd, curr_dist)
        print(f"\nCaso: {desc}")
        print(f"  prev_dist={env.prev_dist:.2f}, curr_dist={curr_dist:.2f}")
        print(f"  r_forward={env.k_forward * (env.prev_dist - curr_dist):.3f}")
        print(f"  reward_total={r:.3f}")

    # Agora testamos o bloqueio
    print("\n=== TESTE DE BLOQUEIO ===")
    env.prev_dist = 1.5
    env.curr_dist = 1.0
    for path_min in [0.8, 0.3, 0.15]:
        env.path_min = path_min
        if env.path_min < env.prox_thresh:
            env.blocking_factor = max(0.0, (env.prox_thresh - env.path_min) / env.prox_thresh)
        else:
            env.blocking_factor = 0.0
        r = env.get_reward(0.2, 0.0, env.curr_dist)
        print(f"path_min={path_min:.2f} → blocking_factor={env.blocking_factor:.2f}, reward_total={r:.3f}")


def main():
    rclpy.init()
    try:
        run_reward_test()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
