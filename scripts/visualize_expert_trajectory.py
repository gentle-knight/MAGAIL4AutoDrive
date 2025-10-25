import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_dir = os.path.join(project_root, "Env")
sys.path.insert(0, project_root)
sys.path.insert(0, env_dir)

# 现在可以导入了
from scenario_env import MultiAgentScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DummyPolicy:
    """
    占位策略,用于数据检查时初始化环境
    不需要实际执行动作,只是为了满足环境初始化要求
    """
    def act(self, *args, **kwargs):
        # 返回零动作 [throttle, steering]
        return np.array([0.0, 0.0])

def visualize_expert_trajectory(env, scenario_idx=0):
    """
    可视化专家轨迹的俯视图动画
    """
    env.reset()
    expert_trajs = env.expert_trajectories
    
    if len(expert_trajs) == 0:
        print("当前场景无专家轨迹")
        return
    
    # 设置绘图
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 获取所有轨迹的最大时间长度
    max_timestep = max(traj["end_timestep"] for traj in expert_trajs.values())
    min_timestep = min(traj["start_timestep"] for traj in expert_trajs.values())
    
    # 绘制完整轨迹(淡色)
    colors = plt.cm.tab10(np.linspace(0, 1, len(expert_trajs)))
    for idx, (obj_id, traj) in enumerate(expert_trajs.items()):
        positions = traj["positions"][:, :2]
        ax.plot(positions[:, 0], positions[:, 1], 
                color=colors[idx], alpha=0.3, linewidth=1,
                label=f'Vehicle {obj_id[:6]}')
    
    # 初始化当前位置标记
    scatter = ax.scatter([], [], s=200, c='red', marker='o', edgecolors='black', linewidths=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Expert Trajectory Visualization - Scenario {scenario_idx}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    def update(frame):
        current_time = min_timestep + frame
        
        # 收集当前时间所有车辆的位置
        current_positions = []
        for traj in expert_trajs.values():
            if traj["start_timestep"] <= current_time <= traj["end_timestep"]:
                idx = current_time - traj["start_timestep"]
                pos = traj["positions"][idx, :2]
                current_positions.append(pos)
        
        if len(current_positions) > 0:
            current_positions = np.array(current_positions)
            scatter.set_offsets(current_positions)
        
        time_text.set_text(f'Time: {frame * 0.1:.1f}s (Frame {frame})')
        return scatter, time_text
    
    anim = FuncAnimation(fig, update, frames=max_timestep-min_timestep+1,
                        interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    WAYMO_DATA_DIR = r"/home/huangfukk/mdsn"
    data_dir = AssetLoader.file_path(WAYMO_DATA_DIR, "exp_filtered", unix_style=False)
    
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": data_dir,
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "use_render": False,
        },
        agent2policy=DummyPolicy()
    )
    
    # 可视化第一个场景
    anim = visualize_expert_trajectory(env, scenario_idx=0)
