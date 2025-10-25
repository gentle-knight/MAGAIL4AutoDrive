import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_dir = os.path.join(project_root, "Env")

sys.path.insert(0, project_root)
sys.path.insert(0, env_dir)

from scenario_env import MultiAgentScenarioEnv
from metadrive.engine.asset_loader import AssetLoader
import numpy as np

class DummyPolicy:
    """
    占位策略,用于数据检查时初始化环境
    不需要实际执行动作,只是为了满足环境初始化要求
    """
    def act(self, *args, **kwargs):
        # 返回零动作 [throttle, steering]
        return np.array([0.0, 0.0])

def check_available_fields():
    """
    检查Waymo转MetaDrive数据中实际可用的字段
    """
    WAYMO_DATA_DIR = r"/home/huangfukk/mdsn"
    data_dir = AssetLoader.file_path(WAYMO_DATA_DIR, "exp_filtered", unix_style=False)
    
    # 创建占位策略
    dummy_policy = DummyPolicy()
    
    # 初始化环境,传入必需的agent2policy参数
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": data_dir,
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "use_render": False,
            "sequential_seed": True,
        },
        agent2policy=dummy_policy  # 添加这个必需参数
    )
    
    print("✓ 环境初始化成功")
    
    # 重置环境以加载数据
    print("正在加载场景数据...")
    env.reset()
    
    # 检查是否有expert_trajectories属性
    if hasattr(env, 'expert_trajectories'):
        print(f"✓ expert_trajectories属性存在,包含 {len(env.expert_trajectories)} 条轨迹")
    else:
        print("⚠️ expert_trajectories属性不存在,请先修改scenario_env.py添加轨迹存储功能")
    
    # 获取一个track样本
    sample_track = None
    for scenario_id, track in env.engine.traffic_manager.current_traffic_data.items():
        if track["type"] == "VEHICLE":
            sample_track = track
            print(f"\n找到样本车辆: scenario_id = {scenario_id}")
            break
    
    if sample_track is None:
        print("未找到车辆轨迹数据")
        env.close()
        return
    
    print("="*60)
    print("Track数据结构分析")
    print("="*60)
    
    # 1. 顶层字段
    print("\n1. Track顶层字段:")
    for key in sample_track.keys():
        print(f"   - {key}: {type(sample_track[key])}")
    
    # 2. metadata字段
    print("\n2. track['metadata']字段:")
    if "metadata" in sample_track:
        for key, value in sample_track["metadata"].items():
            if isinstance(value, (str, int, float, bool)):
                print(f"   - {key}: {type(value).__name__} = {value}")
            else:
                print(f"   - {key}: {type(value).__name__}")
    
    # 3. state字段
    print("\n3. track['state']字段:")
    if "state" in sample_track:
        for key, value in sample_track["state"].items():
            if isinstance(value, np.ndarray):
                print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
                # 打印第一个有效值
                if "valid" in sample_track["state"]:
                    valid_idx = np.argmax(sample_track["state"]["valid"])
                    if valid_idx >= 0 and valid_idx < len(value):
                        print(f"     示例值 (index {valid_idx}): {value[valid_idx]}")
            else:
                print(f"   - {key}: {type(value)} = {value}")
    
    print("\n" + "="*60)
    print("建议存储的字段:")
    print("="*60)
    
    # 检查必需字段
    required_fields = ["position", "heading", "velocity", "valid"]
    print("\n必需字段:")
    all_required_exist = True
    for field in required_fields:
        if "state" in sample_track and field in sample_track["state"]:
            print(f"  ✓ {field} (存在)")
        else:
            print(f"  ✗ {field} (缺失)")
            all_required_exist = False
    
    # 检查可选字段
    optional_fields = ["length", "width", "height", "bbox"]
    print("\n可选字段:")
    available_optional = []
    for field in optional_fields:
        if "state" in sample_track and field in sample_track["state"]:
            print(f"  + {field} (在state中)")
            available_optional.append(field)
        elif "metadata" in sample_track and field in sample_track["metadata"]:
            print(f"  + {field} (在metadata中)")
            available_optional.append(field)
        else:
            print(f"  - {field} (不存在)")
    
    print("\n" + "="*60)
    print("推荐的trajectory_data结构:")
    print("="*60)
    
    if all_required_exist:
        print("""
trajectory_data = {
    "object_id": object_id,
    "scenario_id": scenario_id,
    "valid_mask": valid[first_show:last_show+1].copy(),
    "positions": track["state"]["position"][first_show:last_show+1].copy(),
    "headings": track["state"]["heading"][first_show:last_show+1].copy(),
    "velocities": track["state"]["velocity"][first_show:last_show+1].copy(),
    "timesteps": np.arange(first_show, last_show+1),
    "start_timestep": first_show,
    "end_timestep": last_show,
    "length": last_show - first_show + 1
}
""")
        
        if available_optional:
            print("如果需要车辆尺寸,可选添加:")
            for field in available_optional:
                if field in ["length", "width", "height"]:
                    print(f'  trajectory_data["vehicle_{field}"] = track["state" or "metadata"]["{field}"][first_show]')
    else:
        print("⚠️ 缺少必需字段,请检查数据转换流程")
    
    # 如果有expert_trajectories,展示一个样本
    if hasattr(env, 'expert_trajectories') and len(env.expert_trajectories) > 0:
        print("\n" + "="*60)
        print("expert_trajectories样本:")
        print("="*60)
        sample_traj = list(env.expert_trajectories.values())[0]
        for key, value in sample_traj.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    
    env.close()
    print("\n✓ 分析完成")

if __name__ == "__main__":
    check_available_fields()
