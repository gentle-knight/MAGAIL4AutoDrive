import argparse
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

# Add project root to Python path so we can import Env module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from metadrive.engine.asset_loader import AssetLoader
from Env.expert_replay_env import ExpertReplayEnv

def generate_data(args):
    data_path = os.path.abspath(args.data_dir)
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} not found")

    # MetaDrive's ScenarioDataManager asserts if config["num_scenarios"] > available scenarios in data_directory.
    # So we always set it to -1 (load all available) and clamp the loop range by reading dataset summary.
    from metadrive.scenario.utils import read_dataset_summary
    _, summary_lookup, _ = read_dataset_summary(data_path)
    if args.start_index >= len(summary_lookup):
        raise ValueError(
            f"start_index={args.start_index} out of range. Dataset has {len(summary_lookup)} scenarios."
        )
    max_available = len(summary_lookup) - args.start_index
    num_to_run = min(args.num_scenarios, max_available)

    env_config = {
        "data_directory": data_path,
        "is_multi_agent": True,
        "num_controlled_agents": 100, # Set high to catch all vehicles in scenario
        "horizon": 1000,
        "use_render": False,
        "sequential_seed": True,
        "reactive_traffic": False, # Important: we replay, not react
        "start_scenario_index": args.start_index,
        # Load all scenarios available in the directory to avoid assertion failure.
        # We will still only iterate `num_to_run` scenarios below.
        "num_scenarios": -1,
        "log_level": 50 # ERROR to reduce noise
    }

    expert_trajectories = []
    
    try:
        # Loop through scenarios
        for i in tqdm(range(args.start_index, args.start_index + num_to_run), desc="Scenarios"):
            env = ExpertReplayEnv(config=env_config)
            try:
                obs_dict = env.reset(seed=i)
            except Exception as e:
                print(f"Error resetting scenario {i}: {e}")
                try:
                    env.close()
                except Exception:
                    pass
                continue
            
            # Storage for current episode
            # dict of lists: {agent_id: {'obs': [], 'acts': []}}
            episode_data = {}
            
            # Map agent_id to original ID if possible, but agent_id is unique enough
            
            for step in range(env.config["horizon"]):
                # Step with dummy actions
                obs, rewards, dones, infos = env.step(None)
                
                # 'obs' is next observation (t+1)
                # 'infos' contains 'expert_action' which took (t -> t+1)
                # Wait, usually (obs_t, act_t) -> obs_{t+1}
                # expert_replay_env.step():
                #   calc action (t -> t+1)
                #   move agents to t+1
                #   return obs_{t+1}
                # So we have obs_dict (from reset or prev step) which is at 't'
                # And we have 'infos' which has action at 't'.
                
                current_agents = list(obs_dict.keys())
                
                for agent_id in current_agents:
                    if agent_id not in episode_data:
                        episode_data[agent_id] = {'obs': [], 'acts': []}
                        
                    # Check if we have action for this agent
                    if agent_id in infos and 'expert_action' in infos[agent_id]:
                        action = infos[agent_id]['expert_action']
                        observation = obs_dict[agent_id]
                        
                        episode_data[agent_id]['obs'].append(observation)
                        episode_data[agent_id]['acts'].append(action)
                
                # Update obs_dict for next step
                obs_dict = obs
                
                if dones["__all__"]:
                    break
            
            # Post-process episode data
            for agent_id, data in episode_data.items():
                if len(data['obs']) > 10: # Minimum length filter
                    expert_trajectories.append({
                        'obs': np.array(data['obs']),
                        'acts': np.array(data['acts']),
                        'agent_id': agent_id,
                        'scenario_id': i
                    })
            env.close()
                    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Global error: {e}")
    finally:
        # env is closed per-scenario above (more robust for MetaDrive object lifecycle)
        pass
        
    # Save data
    output_file = os.path.join(args.output_dir, f"expert_data_{args.start_index}_{args.num_scenarios}.pkl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Saving {len(expert_trajectories)} trajectories to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(expert_trajectories, f)
    
    # Verification stats
    if len(expert_trajectories) > 0:
        all_acts = np.concatenate([t['acts'] for t in expert_trajectories])
        print("Action Stats:")
        print(f"  Steering: min={all_acts[:,0].min():.3f}, max={all_acts[:,0].max():.3f}, mean={all_acts[:,0].mean():.3f}")
        print(f"  Accel:    min={all_acts[:,1].min():.3f}, max={all_acts[:,1].max():.3f}, mean={all_acts[:,1].mean():.3f}")

        # Clipping ratio diagnostics (actions are normalized to [-1, 1])
        # If this ratio is high, it usually indicates max_acc/max_steering too small or noisy finite-difference.
        eps = 1e-6
        steer = all_acts[:, 0]
        accel = all_acts[:, 1]
        steer_clipped = np.isclose(np.abs(steer), 1.0, atol=eps)
        accel_clipped = np.isclose(np.abs(accel), 1.0, atol=eps)
        print("Clipping Stats:")
        print(
            f"  Steering clipped (|a|==1): {steer_clipped.mean()*100:.2f}% "
            f"({steer_clipped.sum()}/{len(steer_clipped)})"
        )
        print(
            f"  Accel clipped (|a|==1):    {accel_clipped.mean()*100:.2f}% "
            f"({accel_clipped.sum()}/{len(accel_clipped)})"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/huangfukk/MAGAIL4AutoDrive/data/exp_filtered", help="Path to Waymo pickles (or filtered index)")
    parser.add_argument("--output_dir", type=str, default="/home/huangfukk/MAGAIL4AutoDrive/data/training", help="Output directory")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=10)
    
    args = parser.parse_args()
    generate_data(args)
