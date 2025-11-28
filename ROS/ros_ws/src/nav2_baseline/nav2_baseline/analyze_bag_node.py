import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry, Path

import sqlite3
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Configuração para salvar plots sem precisar de janela (headless/docker)
matplotlib.use('Agg') 

class BagAnalyzerNode(Node):
    def __init__(self):
        super().__init__('bag_analyzer_node')
        
        # Declara o parâmetro 'bag_path'
        self.declare_parameter('bag_path', '')
        
        # Pega o valor
        self.bag_dir = self.get_parameter('bag_path').get_parameter_value().string_value

        fid = self.bag_dir.rfind('/')
        self.name_id = self.bag_dir[fid + 1:]
        
        if not self.bag_dir:
            self.get_logger().error("Por favor, forneça o caminho do bag: --ros-args -p bag_path:=/caminho/para/rec2")
            return
        
        # Creating dict for exporting
        self.d = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "tempo": list(),
            "Ltraj": list(),
            "Lplan": list(),
            "ratio": list()
        }

        # Loop over directory
        for k in range(1, 11):
            self.bag_path = f'{self.bag_dir}/rec{k}.db3'
            self.analyze()

        # Exporting csv
        self.export_csv()

    def calculate_distance(self, points):
        dist = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            dist += math.sqrt(dx**2 + dy**2)
        return dist

    def export_csv(self):
        df = pd.DataFrame(self.d)
        df.to_csv(f'{os.getcwd()}/metrics/metrics_{self.name_id}.csv', index=False)

    def analyze(self):
        self.get_logger().info(f"Analisando bag: {self.bag_path}")

        # Lógica de conexão SQLite (idêntica à anterior, adaptada para o Node logger)
        db_path = self.bag_path
        if os.path.isdir(self.bag_path):
             # Tenta achar o .db3 dentro da pasta
             bag_name = os.path.basename(os.path.normpath(self.bag_path))
             potential_db = os.path.join(self.bag_path, f"{bag_name}_0.db3")
             if os.path.exists(potential_db):
                 db_path = potential_db
        
        # Caso o usuário aponte direto para o .db3
        if not db_path.endswith('.db3') and not os.path.exists(db_path):
             # Fallback para tentar achar qualquer db3 na pasta
             files = os.listdir(self.bag_path)
             db_files = [f for f in files if f.endswith('.db3')]
             if db_files:
                 db_path = os.path.join(self.bag_path, db_files[0])

        self.get_logger().info(f"Lendo banco de dados: {db_path}")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
        except Exception as e:
            self.get_logger().error(f"Erro ao abrir banco de dados: {e}")
            return

        # Mapear tópicos
        topics = {}
        try:
            for row in cursor.execute("SELECT id, name, type FROM topics"):
                topics[row[1]] = {'id': row[0], 'type': row[2]}
        except Exception as e:
            self.get_logger().error(f"Erro ao ler tabela de tópicos (formato do bag inválido?): {e}")
            return

        if '/odom' not in topics:
            self.get_logger().error("Tópico /odom não encontrado no bag.")
            return

        odom_x = []
        odom_y = []
        odom_times = []
        plan_x = []
        plan_y = []
        found_plan = False

        # Ler Odom
        odom_query = f"SELECT timestamp, data FROM messages WHERE topic_id = {topics['/odom']['id']} ORDER BY timestamp"
        for timestamp, data in cursor.execute(odom_query):
            msg = deserialize_message(data, Odometry)
            odom_x.append(msg.pose.pose.position.x)
            odom_y.append(msg.pose.pose.position.y)
            odom_times.append(timestamp)

        # Ler Plan (Apenas o primeiro)
        if '/plan' in topics:
            plan_query = f"SELECT data FROM messages WHERE topic_id = {topics['/plan']['id']} LIMIT 1"
            for (data,) in cursor.execute(plan_query):
                msg = deserialize_message(data, Path)
                if len(msg.poses) > 0:
                    for pose_stamped in msg.poses:
                        plan_x.append(pose_stamped.pose.position.x)
                        plan_y.append(pose_stamped.pose.position.y)
                    found_plan = True
                    self.get_logger().info("Plano global encontrado.")
        
        conn.close()

        if not odom_times:
            self.get_logger().warn("Sem dados de odometria.")
            return

        # --- Métricas ---
        start_time = odom_times[0]
        end_time = odom_times[-1]
        t_travel = (end_time - start_time) / 1e9

        executed_path = list(zip(odom_x, odom_y))
        l_traj = self.calculate_distance(executed_path)

        l_plan = 0.0
        if found_plan:
            planned_path = list(zip(plan_x, plan_y))
            l_plan = self.calculate_distance(planned_path)
        
        ratio = l_traj / l_plan if l_plan > 0 else 0

        # --- Output ---
        print("\n" + "="*40)
        print(f"RESULTADOS: {os.path.basename(self.bag_path)}")
        print("="*40)
        print(f"Tempo (T_travel):   {t_travel:.2f} s")
        print(f"Dist. Real (L_traj): {l_traj:.2f} m")
        print(f"Dist. Plan (L_plan): {l_plan:.2f} m")
        if l_plan > 0:
            print(f"Razão (L/P):        {ratio:.2f}")
        print("="*40 + "\n")

        # Filling dictionary ===============================================

        self.d["tempo"].append(f'{t_travel:.2f}')
        self.d["Ltraj"].append(f'{l_traj:.2f}')
        self.d["Lplan"].append(f'{l_plan:.2f}')
        self.d["ratio"].append(f'{ratio:.2f}')

        # ==================================================================

        # --- Plotagem ---
        # Salva na pasta atual onde o comando foi rodado
        output_filename = f"grafs_traj/{self.name_id}_traj_{os.path.basename(os.path.normpath(self.bag_path))}.png"
        
        plt.figure(figsize=(10, 6))
        if found_plan:
            plt.plot(plan_x, plan_y, 'r--', label=f'Planejado ({l_plan:.2f}m)', linewidth=2, alpha=0.7)
        
        plt.plot(odom_x, odom_y, 'b-', label=f'Executado ({l_traj:.2f}m)', linewidth=2)
        plt.plot(odom_x[0], odom_y[0], 'go', label='Início')
        plt.plot(odom_x[-1], odom_y[-1], 'rx', label='Fim')

        plt.title(f"Trajetória: {os.path.basename(self.bag_path)}\nTempo: {t_travel:.2f}s")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        self.get_logger().info(f"Salvando gráfico em: {os.getcwd()}/grafs_traj/{output_filename}")
        plt.savefig(output_filename)
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = BagAnalyzerNode()
    # Não usamos rclpy.spin(node) porque é um script de processamento único
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()