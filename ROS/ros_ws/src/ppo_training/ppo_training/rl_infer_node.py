#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from nav2_rl_controller.srv import RLInfer

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

class RLInferNode(Node):
    def __init__(self):
        super().__init__('rl_infer_node')

        self.declare_parameter('model_path', 'models/ppo_model.zip')
        self.model_path = self.get_parameter('model_path').value

        # Serviço (agora é a única coisa que o nó faz)
        self.srv = self.create_service(RLInfer, 'rl_infer', self.cb_infer)

        if PPO is None:
            self.get_logger().error("Stable-Baselines3 não foi encontrado. pip install stable-baselines3")
            self.model = None
        else:
            try:
                # Carrega o modelo
                self.model = PPO.load(self.model_path)
                self.get_logger().info(f"Modelo PPO carregado com sucesso de: {self.model_path}")
            except Exception as e:
                self.get_logger().error(f"Falha ao carregar o modelo de '{self.model_path}': {e}")
                self.model = None

    # ====== Callback da Inferência ======
    def cb_infer(self, request, response):
        if self.model is None:
            self.get_logger().warn("Modelo não carregado. Retornando velocidade zero.")
            response.linear_x = 0.0
            response.angular_z = 0.0
            return response
        
        try:
            # 1. Converte o vetor 'obs' do request (que é float64[]) para numpy float32
            # O 'request' agora é a única fonte de observação.
            obs = np.array(request.obs, dtype=np.float32)

            # 2. Executa a predição
            action, _ = self.model.predict(obs, deterministic=True)
            
            # 3. Retorna a ação (velocidades)
            response.linear_x = float(action[0])
            response.angular_z = float(action[1])

        except Exception as e:
            self.get_logger().error(f"Falha na predição do modelo (Model.predict()): {e}")
            self.get_logger().error(f"Formato (shape) da observação recebida: {obs.shape}")
            response.linear_x = 0.0
            response.angular_z = 0.0
            
        return response

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