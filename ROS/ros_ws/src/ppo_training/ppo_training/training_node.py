# ppo_training/ppo_training/training_node.py
import os
import sys
import time
import argparse
import threading
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from .env_wrapper import PPOEnvironment  # seu env (Node que implementa step/reset)


class PPOTrainer(Node):
    def __init__(self, config_file: str, stage: int):
        super().__init__('ppo_trainer')

        # pacote share dir (para salvar modelos dentro do pacote)
        # pkg_share = Path(__file__).resolve().parents[1]
        # self.pkg_share = str(pkg_share)
        pkg_share = get_package_share_directory('ppo_training')
        self.pkg_share = pkg_share

        # Carrega arquivo de configuraÃ§Ã£o (yaml)
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f) or {}

        # ParÃ¢metros de training
        # Podemos aceitar timesteps por estÃ¡gio no YAML (lista) ou um Ãºnico valor (int)
        self.timesteps_per_stage = self.config.get('timesteps_per_stage', None)
        if self.timesteps_per_stage is None:
            # fallback: usa total_timesteps_per_stage se exitir, ou defaults
            default = int(self.config.get('default_timesteps_per_stage', 20000))
            self.timesteps_per_stage = [default, default, default]
        elif isinstance(self.timesteps_per_stage, int):
            self.timesteps_per_stage = [self.timesteps_per_stage] * 3
        elif isinstance(self.timesteps_per_stage, list):
            # ensure length 3
            if len(self.timesteps_per_stage) < 3:
                self.timesteps_per_stage = (self.timesteps_per_stage +
                                            [self.timesteps_per_stage[-1]] * (3 - len(self.timesteps_per_stage)))
            else:
                self.timesteps_per_stage = self.timesteps_per_stage[:3]

        self.stage = int(stage)

        # Save dir
        self.models_dir = os.path.join(self.pkg_share, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Executor and env nodes list (para se certificar de limpar depois)
        self._executor = MultiThreadedExecutor()
        self._env_nodes = []

        # Criar ambiente vetorizado (DummyVecEnv com 1 env)
        def make_env():
            # rclpy must be already initialized externally
            env_node = PPOEnvironment()
            # registra o node no executor para que callbacks (scan/odom) sejam processados
            self._env_nodes.append(env_node)
            self._executor.add_node(env_node)
            return env_node

        self.vec_env = DummyVecEnv([make_env])

        # Se o YAML contiver parÃ¢metros SB3 aplicÃ¡veis, aplique-os ao criar o modelo
        sb3_kwargs = {}
        # Filtrar chaves conhecidas (para evitar passar timesteps_per_stage, etc.)
        sb3_allowed = [
            'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
            'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm',
            'tensorboard_log', 'verbose'
        ]
        for k in sb3_allowed:
            if k in self.config:
                sb3_kwargs[k] = self.config[k]

        # Inicialmente, nÃ£o instanciamos o PPO aqui: isso serÃ¡ feito em train_stage()
        self.model = None
        self.sb3_kwargs = sb3_kwargs

        # Start executor background thread to process ROS callbacks for env nodes
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()
        self.get_logger().info("Executor iniciado em thread de background (callbacks ROS ativos).")

    def _model_path_for_stage(self, stage_idx: int):
        return os.path.join(self.models_dir, f"ppo_stage{stage_idx}.zip")

    def _maybe_load_previous(self, stage_idx: int, env):
        """
        If a model for (stage_idx - 1) exists, carrega e retorna o modelo.
        Caso contrário, retorna None.
        """
        if stage_idx <= 1:
            return None

        prev_path = self._model_path_for_stage(stage_idx - 1)
        if os.path.exists(prev_path):
            self.get_logger().info(f"Encontrado modelo do estágio anterior: {prev_path}. Carregando e continuando treino.")
            model = PPO.load(prev_path, env=env)
            return model
        else:
            self.get_logger().info(f"Nenhum modelo do estágio anterior encontrado em {prev_path}. Treinando do zero para stage {stage_idx}.")
            return None

    def train_stage(self, stage_idx: int, timesteps: int, checkpoint_freq: int = 5000):
        """
        Treina para um estágio específico.
        - Se existir modelo do estágio anterior, ele será carregado e continuará o treino.
        - Caso contrário, cria-se um novo modelo.
        """
        env = self.vec_env

        # verifica se hÃ¡ modelo anterior a ser carregado
        loaded_model = self._maybe_load_previous(stage_idx, env)
        if loaded_model is not None:
            self.model = loaded_model
            # Quando continuar treino, nÃ£o resetar contagem de timesteps
            reset_num_timesteps = False
        else:
            # cria novo modelo
            self.model = PPO(policy="MlpPolicy", env=env, verbose=1, **self.sb3_kwargs)
            reset_num_timesteps = True

        # Callbacks: checkpoint a cada checkpoint_freq timesteps (salva dentro de models/)
        checkpoint_dir = os.path.join(self.models_dir, f"stage{stage_idx}_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_cb = CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir,
                                           name_prefix=f"ppo_stage{stage_idx}_ckpt")

        self.get_logger().info(f"Iniciando treinamento do STAGE {stage_idx} por {timesteps} timesteps "
                               f"(checkpoint a cada {checkpoint_freq} steps).")

        try:
            self.model.learn(total_timesteps=timesteps, reset_num_timesteps=reset_num_timesteps, callback=checkpoint_cb)
        except KeyboardInterrupt:
            self.get_logger().warn("Treinamento interrompido via KeyboardInterrupt. Salvando modelo parcial...")
        except Exception as e:
            self.get_logger().error(f"Erro durante learn(): {e}")
            raise

        # Salva o modelo final do stage
        save_path = self._model_path_for_stage(stage_idx)
        self.model.save(save_path)
        self.get_logger().info(f"Modelo final do stage {stage_idx} salvo em: {save_path}")

    def shutdown(self):
        # Para executor e destrÃ³i nÃ³s do ambiente
        self.get_logger().info("Encerrando executor e destruindo nÃ³s de ambiente...")
        try:
            # remover nodes do executor e destruir
            for node in list(self._env_nodes):
                try:
                    self._executor.remove_node(node)
                except Exception:
                    pass
                try:
                    node.destroy_node()
                except Exception:
                    pass
            # solicitar shutdown do executor
            self._executor.shutdown()
        finally:
            # encerra rclpy (fechado externamente no main)
            self.get_logger().info("Shutdown completo.")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage', type=int, default=1, help='Stage atual (1,2,3). Usado para carregar modelo anterior se existir.')
    parser.add_argument('--config', type=str, default='', help='Caminho para ppo_params.yaml (se vazio, usa config/ppo_params.yaml do pacote)')
    parser.add_argument('--checkpoint_freq', type=int, default=5000, help='FrequÃªncia de checkpoint (timesteps)')
    args, unknown = parser.parse_known_args(argv)

    # Inicializa rclpy (necessÃ¡rio antes de criar Nodes)
    rclpy.init(args=unknown)

    # Determina caminho do config file
    if args.config:
        config_file = args.config
    else:
        # caminho relativo ao pacote
        # pkg_share = Path(__file__).resolve().parents[1]
        # config_file = os.path.join(pkg_share, 'config', 'ppo_params.yaml')
        pkg_share = get_package_share_directory('ppo_training')
        config_file = os.path.join(pkg_share, 'config', 'ppo_params.yaml')

    trainer = PPOTrainer(config_file=config_file, stage=args.stage)

    # Determine timesteps para este stage
    # trainer.timesteps_per_stage Ã© uma lista de 3 valores
    stage_idx = args.stage
    if stage_idx < 1 or stage_idx > 3:
        trainer.get_logger().warn("Stage inválido, usando 1.")
        stage_idx = 1

    timesteps = trainer.timesteps_per_stage[stage_idx - 1]

    try:
        trainer.train_stage(stage_idx=stage_idx, timesteps=timesteps, checkpoint_freq=args.checkpoint_freq)
    finally:
        # sempre tentar salvar e encerrar corretamente
        try:
            trainer.shutdown()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])