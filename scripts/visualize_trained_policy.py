"""
Unified visualization for BC and MAGAIL trained policies.
Use --policy_type bc or magail (or auto-detect from --model_path: .pt -> bc, else magail).
"""
import argparse
import os
import sys
import torch
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Env.bc_env import BCScenarioEnv
from metadrive.engine.engine_utils import close_engine


def _resolve_data_dir(args):
    """Resolve data directory: explicit or auto-detect under project data/."""
    if args.data_dir:
        data_dir = args.data_dir
    else:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(current_dir, "data", "exp_filtered")
        if not os.path.exists(data_dir):
            data_dir = os.path.join(current_dir, "data", "exp_converted")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}. Please specify --data_dir.")
    return data_dir


def _resolve_model_path(model_path, policy_type):
    """Resolve model path: if not found, try models/bc or models/magail."""
    if os.path.exists(model_path):
        return model_path
    if policy_type == "bc":
        candidate = os.path.join("models", "bc", model_path)
    else:
        candidate = os.path.join("models", "magail", model_path)
    if os.path.exists(candidate):
        return candidate
    if policy_type == "magail" and not model_path.endswith("_actor.pth"):
        candidate = model_path + "_actor.pth"
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Model path {model_path} not found (tried {candidate}).")


def visualize_model(args):
    policy_type = (args.policy_type or "auto").lower()
    if policy_type == "auto":
        policy_type = "bc" if args.model_path.endswith(".pt") else "magail"

    data_dir = _resolve_data_dir(args)
    data_path = os.path.abspath(data_dir)
    env_config = {
        "data_directory": data_path,
        "is_multi_agent": True,
        "num_controlled_agents": 3,
        "horizon": args.horizon,
        "use_render": True,
        "sequential_seed": True,
        "start_scenario_index": args.start_index,
        "num_scenarios": args.num_scenarios,
        "log_level": 40,
    }

    print(f"Initializing BCScenarioEnv (policy_type={policy_type})...")
    try:
        env = BCScenarioEnv(env_config, agent2policy={})
    except Exception as e:
        print(f"Error init env: {e}. Trying to close lingering engine...")
        try:
            close_engine()
        except Exception:
            pass
        env = BCScenarioEnv(env_config, agent2policy={})

    state_dim = 45
    action_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = _resolve_model_path(args.model_path, policy_type)
    print(f"Loading model from {model_path}...")

    if policy_type == "bc":
        from Algorithm.policy import StateIndependentPolicy
        policy = StateIndependentPolicy(
            state_shape=(state_dim,),
            action_shape=(action_dim,),
            hidden_units=(256, 256),
            hidden_activation=torch.nn.Tanh(),
        ).to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
    else:
        from train_magail import Actor
        actor = Actor(state_dim, action_dim).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()

    try:
        for i in range(args.start_index, args.start_index + args.num_scenarios):
            print(f"\n--- Playing Scenario {i} ---")
            try:
                obs_dict = env.reset(seed=i)
            except Exception as e:
                print(f"Error resetting {i}: {e}. Skipping.")
                try:
                    close_engine()
                    env = BCScenarioEnv(env_config, agent2policy={})
                except Exception:
                    pass
                continue

            print(f"Scenario loaded. Controlled agents: {len(obs_dict)}")
            step_count = 0
            episode_reward = 0.0

            while True:
                actions = {}
                agent_ids = list(obs_dict.keys())
                obs_list = [obs_dict[aid] for aid in agent_ids]
                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)

                with torch.no_grad():
                    if policy_type == "bc":
                        actions_np = policy(obs_tensor).cpu().numpy()
                    else:
                        dist = actor(obs_tensor)
                        if args.deterministic:
                            actions_np = torch.tanh(dist.mean).cpu().numpy()
                        else:
                            actions_np = torch.tanh(dist.sample()).cpu().numpy()

                for idx, aid in enumerate(agent_ids):
                    actions[aid] = actions_np[idx].flatten()

                obs_dict, rewards, dones, infos = env.step(actions)
                episode_reward += sum(rewards.values())

                env.render(
                    mode="top_down",
                    text={
                        "Scenario": i,
                        "Step": step_count,
                        "Agents": len(obs_dict),
                        "Total Reward": f"{episode_reward:.2f}",
                    },
                )
                step_count += 1

                if dones["__all__"] or step_count >= args.horizon:
                    print(f"Scenario finished at step {step_count}, reward {episode_reward:.2f}")
                    break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BC or MAGAIL trained policy in 45-dim scenario env."
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="auto",
        choices=["auto", "bc", "magail"],
        help="Policy type: bc (StateIndependentPolicy .pt) or magail (Actor _actor.pth). auto = infer from model_path.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/bc/policy_best.pt",
        help="Path to model: BC .pt (e.g. models/bc/policy_best.pt) or MAGAIL _actor.pth (e.g. models/magail/model_50_actor.pth)",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Waymo data directory (default: data/exp_filtered)")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--deterministic", action="store_true", help="For MAGAIL: use mean action instead of sampling")

    args = parser.parse_args()
    visualize_model(args)
