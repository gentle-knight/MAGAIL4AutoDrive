import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_dir = os.path.join(project_root, "Env")
sys.path.insert(0, project_root)
sys.path.insert(0, env_dir)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scenario_env import MultiAgentScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
import pickle
import os

class DummyPolicy:
    """占位策略"""
    def act(self, *args, **kwargs):
        return np.array([0.0, 0.0])

class ExpertDataAnalyzer:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.env = MultiAgentScenarioEnv(
            config={
                "data_directory": data_directory,
                "is_multi_agent": True,
                "num_controlled_agents": 3,
                "use_render": False,
                "sequential_seed": True,
            },
            agent2policy=DummyPolicy()  # 添加必需参数
        )
        
        self.statistics = {
            "num_scenarios": 0,
            "num_trajectories": 0,
            "trajectory_lengths": [],
            "velocities": [],
            "speeds": [],  # 速度大小
            "accelerations": [],
            "heading_changes": [],
            "inter_vehicle_distances": [],
            "num_vehicles_per_scenario": [],
            "static_vehicles": 0,  # 统计静止车辆
        }
    
    def analyze_all_scenarios(self, num_scenarios=None):
        """遍历所有场景并收集统计信息"""
        scenario_count = 0
        
        while True:
            try:
                obs = self.env.reset()
                
                if not hasattr(self.env, 'expert_trajectories'):
                    print("⚠️ 环境缺少expert_trajectories属性")
                    break
                
                expert_trajs = self.env.expert_trajectories
                
                if len(expert_trajs) == 0:
                    continue
                
                scenario_count += 1
                self.statistics["num_scenarios"] += 1
                self.statistics["num_vehicles_per_scenario"].append(len(expert_trajs))
                
                # 分析每条轨迹
                for obj_id, traj in expert_trajs.items():
                    self.analyze_single_trajectory(traj)
                
                # 分析车辆间交互
                self.analyze_vehicle_interactions(expert_trajs)
                
                print(f"已分析场景 {scenario_count}/{num_scenarios}, 车辆数: {len(expert_trajs)}")
                
                if num_scenarios and scenario_count >= num_scenarios:
                    break
                    
            except Exception as e:
                print(f"场景 {scenario_count} 处理失败: {e}")
                break
        
        self.env.close()
    
    def analyze_single_trajectory(self, traj):
        """分析单条轨迹"""
        self.statistics["num_trajectories"] += 1
        
        length = traj["length"]
        self.statistics["trajectory_lengths"].append(length)
        
        # 速度分析
        velocities = traj["velocities"]
        speeds = np.linalg.norm(velocities, axis=1)
        self.statistics["velocities"].extend(velocities.tolist())
        self.statistics["speeds"].extend(speeds.tolist())
        
        # 检查是否为静止车辆
        if np.max(speeds) < 0.5:  # 最大速度小于0.5m/s视为静止
            self.statistics["static_vehicles"] += 1
        
        # 加速度分析
        if length > 1:
            accelerations = np.diff(speeds) * 10  # 10Hz数据
            self.statistics["accelerations"].extend(accelerations.tolist())
        
        # 航向角变化
        headings = traj["headings"]
        if length > 1:
            heading_changes = np.diff(headings)
            heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))
            self.statistics["heading_changes"].extend(heading_changes.tolist())
    
    def analyze_vehicle_interactions(self, expert_trajs):
        """分析车辆间的距离"""
        if len(expert_trajs) < 2:
            return
        
        traj_list = list(expert_trajs.values())
        
        for i in range(len(traj_list)):
            for j in range(i+1, len(traj_list)):
                traj_i = traj_list[i]
                traj_j = traj_list[j]
                
                start_time = max(traj_i["start_timestep"], traj_j["start_timestep"])
                end_time = min(traj_i["end_timestep"], traj_j["end_timestep"])
                
                if start_time >= end_time:
                    continue
                
                idx_i_start = start_time - traj_i["start_timestep"]
                idx_i_end = end_time - traj_i["start_timestep"]
                idx_j_start = start_time - traj_j["start_timestep"]
                idx_j_end = end_time - traj_j["start_timestep"]
                
                pos_i = traj_i["positions"][idx_i_start:idx_i_end, :2]
                pos_j = traj_j["positions"][idx_j_start:idx_j_end, :2]
                
                distances = np.linalg.norm(pos_i - pos_j, axis=1)
                self.statistics["inter_vehicle_distances"].extend(distances.tolist())
    
    def generate_report(self, save_dir="./analysis_results"):
        """生成统计报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        stats = self.statistics
        
        print("\n" + "="*60)
        print("专家数据集统计报告")
        print("="*60)
        print(f"总场景数: {stats['num_scenarios']}")
        print(f"总轨迹数: {stats['num_trajectories']}")
        print(f"静止车辆数: {stats['static_vehicles']} ({stats['static_vehicles']/stats['num_trajectories']*100:.1f}%)")
        print(f"平均每场景车辆数: {np.mean(stats['num_vehicles_per_scenario']):.2f} ± {np.std(stats['num_vehicles_per_scenario']):.2f}")
        
        print(f"\n轨迹长度统计 (帧数 @ 10Hz):")
        print(f"  平均: {np.mean(stats['trajectory_lengths']):.2f} 帧 ({np.mean(stats['trajectory_lengths'])*0.1:.2f}秒)")
        print(f"  中位数: {np.median(stats['trajectory_lengths']):.2f} 帧")
        print(f"  最小/最大: {np.min(stats['trajectory_lengths'])} / {np.max(stats['trajectory_lengths'])} 帧")
        
        print(f"\n速度统计 (m/s):")
        speeds = np.array(stats['speeds'])
        print(f"  平均: {np.mean(speeds):.2f} ± {np.std(speeds):.2f}")
        print(f"  中位数: {np.median(speeds):.2f}")
        print(f"  最小/最大: {np.min(speeds):.2f} / {np.max(speeds):.2f}")
        print(f"  静止帧(<0.5m/s): {np.sum(speeds < 0.5)} ({np.sum(speeds < 0.5)/len(speeds)*100:.1f}%)")
        
        print(f"\n加速度统计 (m/s²):")
        accs = np.array(stats['accelerations'])
        print(f"  平均: {np.mean(accs):.4f} ± {np.std(accs):.2f}")
        print(f"  最小/最大: {np.min(accs):.2f} / {np.max(accs):.2f}")
        
        if len(stats['inter_vehicle_distances']) > 0:
            dists = np.array(stats['inter_vehicle_distances'])
            print(f"\n车辆间距离统计 (m):")
            print(f"  平均: {np.mean(dists):.2f} ± {np.std(dists):.2f}")
            print(f"  最小: {np.min(dists):.2f}")
            print(f"  近距离交互(<5m): {np.sum(dists < 5.0)} ({np.sum(dists < 5.0)/len(dists)*100:.2f}%)")
        
        # 保存数据
        with open(os.path.join(save_dir, "statistics.pkl"), "wb") as f:
            pickle.dump(stats, f)
        
        # 绘制可视化
        self.plot_distributions(save_dir)
        
        print(f"\n✓ 报告已保存到: {save_dir}")
    
    def plot_distributions(self, save_dir):
        """绘制分布图"""
        stats = self.statistics
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 轨迹长度分布
        axes[0, 0].hist(stats['trajectory_lengths'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Trajectory Length (frames @ 10Hz)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Trajectory Length Distribution')
        axes[0, 0].axvline(np.mean(stats['trajectory_lengths']), color='red', 
                           linestyle='--', label=f'Mean: {np.mean(stats["trajectory_lengths"]):.1f}')
        axes[0, 0].legend()
        
        # 2. 速度分布
        axes[0, 1].hist(stats['speeds'], bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Speed (m/s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Speed Distribution')
        axes[0, 1].axvline(np.mean(stats['speeds']), color='red', 
                           linestyle='--', label=f'Mean: {np.mean(stats["speeds"]):.2f}')
        axes[0, 1].legend()
        
        # 3. 加速度分布
        axes[0, 2].hist(stats['accelerations'], bins=50, edgecolor='black')
        axes[0, 2].set_xlabel('Acceleration (m/s²)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Acceleration Distribution')
        
        # 4. 每场景车辆数
        axes[1, 0].hist(stats['num_vehicles_per_scenario'], bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Vehicles per Scenario')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Vehicles per Scenario')
        
        # 5. 航向角变化
        axes[1, 1].hist(stats['heading_changes'], bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Heading Change (rad)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Heading Change Distribution')
        
        # 6. 车辆间距离
        if len(stats['inter_vehicle_distances']) > 0:
            axes[1, 2].hist(stats['inter_vehicle_distances'], bins=50, 
                           range=(0, 50), edgecolor='black')
            axes[1, 2].set_xlabel('Inter-vehicle Distance (m)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Distance Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distributions.png"), dpi=300)
        print(f"  ✓ 分布图已保存")

if __name__ == "__main__":
    WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/data"
    data_dir = AssetLoader.file_path(WAYMO_DATA_DIR, "exp_filtered", unix_style=False)
    
    print("开始分析专家数据...")
    analyzer = ExpertDataAnalyzer(data_dir)
    analyzer.analyze_all_scenarios(num_scenarios=100)  # 分析100个场景
    analyzer.generate_report()
