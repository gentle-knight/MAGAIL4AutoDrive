import argparse
import os
import sys
import torch
import numpy as np
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_magail import Actor, MAGAILScenarioEnv
from metadrive.engine.engine_utils import close_engine

def visualize_model(args):
    # 1. Load Environment
    data_path = os.path.abspath(args.data_dir)
    env_config = {
        "data_directory": data_path,
        "is_multi_agent": True,
        "num_controlled_agents": 3,
        "horizon": args.horizon,
        "use_render": True,  # Visualisation enabled
        "sequential_seed": True,
        "start_scenario_index": args.start_index,
        "num_scenarios": args.num_scenarios,
        "log_level": 40,
    }
    
    print("Initializing MAGAILScenarioEnv...")
    try:
        env = MAGAILScenarioEnv(config=env_config, agent2policy={})
    except Exception as e:
        print(f"Error init env: {e}. Trying to close lingering engine...")
        try:
            close_engine()
        except:
            pass
        env = MAGAILScenarioEnv(config=env_config, agent2policy={})

    # 2. Load Model
    state_dim = 45
    action_dim = 2
    
    actor = Actor(state_dim, action_dim).cuda()
    
    model_path = args.model_path
    if not os.path.exists(model_path):
        # Try to find it in runs/
        potential_path = os.path.join("runs", "magail_production", model_path)
        if os.path.exists(potential_path):
            model_path = potential_path
        else:
            # Try appending _actor.pth
            potential_path = model_path + "_actor.pth"
            if os.path.exists(potential_path):
                model_path = potential_path
            else:
                raise ValueError(f"Model path {args.model_path} not found.")
                
    print(f"Loading model from {model_path}...")
    actor.load_state_dict(torch.load(model_path))
    actor.eval()
    
    # 3. Run Loop
    try:
        for i in range(args.start_index, args.start_index + args.num_scenarios):
            print(f"\n--- Playing Scenario {i} ---")
            
            # Reset
            try:
                # Use sequential seed logic or specific seed?
                # ExpertReplayEnv/ScenarioEnv logic: seed matches scenario index if configured right
                obs_dict = env.reset(seed=i)
            except Exception as e:
                print(f"Error resetting {i}: {e}. Skipping.")
                # Try soft reset
                try:
                    close_engine()
                    env = MAGAILScenarioEnv(config=env_config, agent2policy={})
                except:
                    pass
                continue
                
            print(f"Scenario loaded. Controlled agents: {len(obs_dict)}")
            
            step_count = 0
            while True:
                actions = {}
                # Inference
                for agent_id, obs in obs_dict.items():
                    # Preprocess obs: (45,) -> (1, 45) tensor
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).cuda()
                    with torch.no_grad():
                        dist = actor(obs_tensor)
                        # Deterministic action for viz? Or sample?
                        # Usually deterministic (mean) is better for checking performance
                        # But training uses sample.
                        if args.deterministic:
                            action = torch.tanh(dist.mean) # Use mean of Gaussian
                        else:
                            pre_tanh = dist.sample()
                            action = torch.tanh(pre_tanh)
                            
                    actions[agent_id] = action.cpu().numpy().flatten()
                
                # Step
                obs_dict, rewards, dones, infos = env.step(actions)
                
                # Render
                env.render(
                    mode="top_down",
                    text={
                        "Scenario": i,
                        "Step": step_count,
                        "Agents": len(obs_dict)
                    }
                )
                
                step_count += 1
                # time.sleep(0.02) # Slow down if needed
                
                if dones["__all__"] or step_count >= args.horizon:
                    print(f"Scenario finished at step {step_count}")
                    break
                    
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to actor model pth (e.g. runs/magail_production/model_50_actor.pth)")
    parser.add_argument("--data_dir", type=str, default="data/exp_filtered")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--deterministic", action="store_true", help="Use mean action instead of sampling")
    
    args = parser.parse_args()
    visualize_model(args)
