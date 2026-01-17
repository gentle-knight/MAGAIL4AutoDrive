import argparse
import os
import sys
import time

# Add project root to Python path so we can import Env module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Env.expert_replay_env import ExpertReplayEnv

def visualize_replay(args):
    data_path = os.path.abspath(args.data_dir)
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} not found")

    # Same as data generation: avoid MetaDrive assertion when requested num_scenarios > available.
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
        "num_controlled_agents": 100,
        "horizon": args.horizon,
        "use_render": True,  # Enable rendering
        "sequential_seed": True,
        "reactive_traffic": False,
        "start_scenario_index": args.start_index,
        "num_scenarios": -1,
        "log_level": 40, # ERROR
        # "pstats": True, # For performance debugging
    }

    print(f"Initializing ExpertReplayEnv with data from {data_path}...")
    env = ExpertReplayEnv(config=env_config)

    try:
        for i in range(args.start_index, args.start_index + num_to_run):
            print(f"\n--- Playing Scenario {i} ---")
            try:
                obs = env.reset(seed=i)
            except Exception as e:
                print(f"Error resetting scenario {i}: {e}")
                continue
            
            print(f"Scenario loaded. Controlled agents: {len(env.controlled_agents)}")
            
            for step in range(args.horizon):
                # Step
                obs, rewards, dones, infos = env.step(None)
                
                # Render
                env.render(mode="top_down", 
                           text={
                               "Step": step, 
                               "Agents": len(env.controlled_agents),
                               "Scenario": i
                           })
                
                # Sleep to control playback speed
                time.sleep(0.05)
                
                if dones["__all__"]:
                    print(f"Scenario {i} finished at step {step}")
                    break
                    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Global error: {e}")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/huangfukk/MAGAIL4AutoDrive/data/exp_filtered", help="Path to Waymo data")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=500)
    
    args = parser.parse_args()
    visualize_replay(args)
