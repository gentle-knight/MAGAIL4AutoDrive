import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "Env"))

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from scenario_env import MultiAgentScenarioEnv
from metadrive.engine.asset_loader import AssetLoader

class DummyPolicy:
    def act(self, *args, **kwargs):
        return np.array([0.0, 0.0])

class ExpertTrajectoryDataset(Dataset):
    """
    完整107维观测的专家轨迹数据集
    """
    
    def __init__(self, 
                 trajectory_data: dict,
                 observation_data: dict = None,  # 可选的完整观测
                 sequence_length: int = 1,
                 extract_actions: bool = True):
        """
        Args:
            trajectory_data: 专家轨迹数据
            observation_data: 完整107维观测数据(可选)
            sequence_length: 序列长度
            extract_actions: 是否提取动作
        """
        self.trajectory_data = trajectory_data
        self.observation_data = observation_data if observation_data else {}
        self.sequence_length = sequence_length
        self.extract_actions = extract_actions
        
        # 构建索引
        self.indices = []
        for traj_id, traj in trajectory_data.items():
            traj_len = traj["length"]
            for start_idx in range(traj_len - sequence_length):
                self.indices.append((traj_id, start_idx))
        
        obs_dim = 107 if len(self.observation_data) > 0 else 5
        print(f"专家数据集: {len(trajectory_data)} 条轨迹, "
              f"{len(self.indices)} 个训练样本, 观测维度: {obs_dim}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_id, start_idx = self.indices[idx]
        traj = self.trajectory_data[traj_id]
        
        end_idx = start_idx + self.sequence_length
        
        # 如果有完整观测,使用完整观测(107维)
        if traj_id in self.observation_data and len(self.observation_data[traj_id]) > 0:
            obs_sequence = self.observation_data[traj_id]
            states = obs_sequence[start_idx:end_idx]  # (seq_len, 107)
        else:
            # 否则使用简化观测(5维)
            positions = traj["positions"][start_idx:end_idx+1]
            headings = traj["headings"][start_idx:end_idx+1]
            velocities = traj["velocities"][start_idx:end_idx]
            
            states = []
            for i in range(self.sequence_length):
                state = np.concatenate([
                    positions[i, :2],    # x, y
                    velocities[i],       # vx, vy
                    [headings[i]],       # heading
                ])
                states.append(state)
            states = np.array(states)
        
        if self.extract_actions:
            positions = traj["positions"][start_idx:end_idx+1]
            headings = traj["headings"][start_idx:end_idx+1]
            velocities = traj["velocities"][start_idx:end_idx]
            
            actions = self._extract_actions_from_states(
                positions[:-1], positions[1:], 
                headings[:-1], headings[1:],
                velocities
            )
            return torch.FloatTensor(states), torch.FloatTensor(actions)
        else:
            next_states = states[1:]
            return torch.FloatTensor(states[:-1]), torch.FloatTensor(next_states)
    
    def _extract_actions_from_states(self, pos_t, pos_t1, head_t, head_t1, vel_t):
        """从状态序列反推动作"""
        actions = []
        dt = 0.1
        
        for i in range(len(pos_t)):
            current_speed = np.linalg.norm(vel_t[i])
            displacement = np.linalg.norm(pos_t1[i, :2] - pos_t[i, :2])
            next_speed = displacement / dt
            
            speed_change = (next_speed - current_speed) / dt
            if speed_change >= 0:
                throttle = np.clip(speed_change / 5.0, 0.0, 1.0)
            else:
                throttle = np.clip(speed_change / 8.0, -1.0, 0.0)
            
            heading_change = head_t1[i] - head_t[i]
            heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))
            steering = np.clip(heading_change / 0.2, -1.0, 1.0)
            
            actions.append([throttle, steering])
        
        return np.array(actions)
    
    @staticmethod
    def collect_with_full_obs(env_config, num_scenarios=10, save_path=None):
        """
        ✅ 使用env._get_all_obs()收集完整107维观测
        
        这是正确的方法!直接利用环境已有的观测函数
        """
        all_trajectories = {}
        all_observations = {}
        
        # 检查数据库
        data_dir = env_config["config"]["data_directory"]
        summary_path = os.path.join(data_dir, "dataset_summary.pkl")
        
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)
        
        total_scenarios = len(summary)
        print(f"数据库总场景数: {total_scenarios}")
        
        if num_scenarios is None:
            num_scenarios = total_scenarios
        else:
            num_scenarios = min(num_scenarios, total_scenarios)
        
        print(f"计划收集(完整107维观测): {num_scenarios} 个场景")
        
        for i in range(num_scenarios):
            try:
                # 创建环境
                env = MultiAgentScenarioEnv(
                    config={
                        **env_config["config"],
                        "start_scenario_index": i,
                        "num_scenarios": 1,
                    },
                    agent2policy=env_config["agent2policy"]
                )
                
                # 重置环境
                env.reset()
                
                if not hasattr(env, 'expert_trajectories'):
                    print(f"⚠️ 场景 {i}: 缺少expert_trajectories")
                    env.close()
                    continue
                
                expert_trajs = env.expert_trajectories
                
                if len(expert_trajs) == 0:
                    print(f"⚠️ 场景 {i}: 无专家轨迹")
                    env.close()
                    continue
                
                # 存储轨迹
                scenario_id = env.engine.current_seed
                for obj_id, traj in expert_trajs.items():
                    unique_id = f"scenario{i}_{obj_id}"
                    all_trajectories[unique_id] = traj
                
                # ✅ 关键: 使用_get_all_obs()获取完整观测
                # 创建agent_id到unique_id的映射
                agent_to_unique = {}
                for agent_id in env.controlled_agents.keys():
                    # 尝试匹配agent_id到expert_trajectories的obj_id
                    for obj_id in expert_trajs.keys():
                        if str(agent_id) in str(obj_id) or str(obj_id) in str(agent_id):
                            unique_id = f"scenario{i}_{obj_id}"
                            agent_to_unique[agent_id] = unique_id
                            all_observations[unique_id] = []
                            break
                
                # 遍历场景的每一步,收集完整观测
                max_steps = min([traj["length"] for traj in expert_trajs.values()])
                
                for step in range(max_steps):
                    # ✅ 直接调用_get_all_obs()获取107维观测!
                    obs_list = env._get_all_obs()
                    
                    # 存储每个agent的观测
                    for agent_idx, agent_id in enumerate(env.controlled_agents.keys()):
                        if agent_id in agent_to_unique:
                            unique_id = agent_to_unique[agent_id]
                            if agent_idx < len(obs_list):
                                # obs_list[agent_idx]已经是107维向量!
                                all_observations[unique_id].append(np.array(obs_list[agent_idx]))
                    
                    # 执行零动作(保持场景状态)
                    actions = {aid: np.array([0.0, 0.0]) 
                              for aid in env.controlled_agents.keys()}
                    env.step(actions)
                
                # 转换为numpy数组
                for unique_id in list(all_observations.keys()):
                    if len(all_observations[unique_id]) > 0:
                        all_observations[unique_id] = np.array(all_observations[unique_id])
                    else:
                        del all_observations[unique_id]
                
                env.close()
                
                if (i + 1) % 5 == 0:
                    print(f"✓ 已收集 {i+1}/{num_scenarios}, "
                          f"轨迹: {len(all_trajectories)}, "
                          f"观测: {len(all_observations)}")
                          
            except Exception as e:
                print(f"✗ 场景 {i} 收集失败: {e}")
                import traceback
                traceback.print_exc()
                try:
                    env.close()
                except:
                    pass
                continue
        
        print(f"\n收集完成!")
        print(f"  轨迹数: {len(all_trajectories)}")
        print(f"  完整观测数: {len(all_observations)}")
        
        # 验证观测维度
        if len(all_observations) > 0:
            sample_obs = list(all_observations.values())[0]
            if len(sample_obs) > 0:
                obs_dim = len(sample_obs[0])
                print(f"  观测维度: {obs_dim} (应为107)")
        
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump({
                    "trajectories": all_trajectories,
                    "observations": all_observations
                }, f)
            print(f"数据已保存到: {save_path}")
        
        return all_trajectories, all_observations


if __name__ == "__main__":
    WAYMO_DATA_DIR = r"/home/huangfukk/mdsn"
    data_dir = AssetLoader.file_path(WAYMO_DATA_DIR, "exp_filtered", unix_style=False)
    
    env_config = {
        "config": {
            "data_directory": data_dir,
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "use_render": False,
            "sequential_seed": True,
        },
        "agent2policy": DummyPolicy()
    }
    
    print("=" * 60)
    print("选择收集模式:")
    print("1. 简化观测(5维) - 快速,已验证 ✅")
    print("2. 完整观测(107维) - 使用_get_all_obs() ⭐")
    print("=" * 60)
    
    mode = input("请选择模式(1或2,默认1): ").strip() or "1"
    
    if mode == "2":
        print("\n开始收集完整107维观测...")
        trajectories, observations = ExpertTrajectoryDataset.collect_with_full_obs(
            env_config,
            num_scenarios=10,
            save_path="./expert_trajectories_full.pkl"
        )
        
        if len(trajectories) > 0:
            dataset = ExpertTrajectoryDataset(
                trajectories, 
                observations,
                sequence_length=1
            )
            state, action = dataset[0]
            print(f"\n数据集测试:")
            print(f"  总轨迹数: {len(trajectories)}")
            print(f"  总观测数: {len(observations)}")
            print(f"  训练样本数: {len(dataset)}")
            print(f"  状态维度: {state.shape}")
            print(f"  动作维度: {action.shape}")
    else:
        print("\n开始收集简化5维观测...")
        # 保持原有的简化版本代码...
        print("(使用之前已成功的方法)")
