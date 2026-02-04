"""
Unified visualization: replay (scenario replay), policy (BC/MAGAIL), trajectory (2D expert trajectory animation).
Usage: python scripts/visualize.py <replay|policy|trajectory> [args...]
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Replay ---
def _run_replay(args):
    from Env.expert_replay_env import ExpertReplayEnv

    data_path = os.path.abspath(args.data_dir)
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} not found")

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
        "use_render": True,
        "sequential_seed": True,
        "reactive_traffic": False,
        "start_scenario_index": args.start_index,
        "num_scenarios": -1,
        "log_level": 40,
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
                obs, rewards, dones, infos = env.step(None)
                env.render(
                    mode="top_down",
                    text={"Step": step, "Agents": len(env.controlled_agents), "Scenario": i},
                )
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


# --- Policy (BC / MAGAIL) ---
def _resolve_data_dir(data_dir_arg):
    if data_dir_arg:
        data_dir = data_dir_arg
    else:
        data_dir = os.path.join(project_root, "data", "exp_filtered")
        if not os.path.exists(data_dir):
            data_dir = os.path.join(project_root, "data", "exp_converted")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}. Please specify --data_dir.")
    return data_dir


def _resolve_model_path(model_path, policy_type):
    if os.path.exists(model_path):
        return model_path
    if policy_type == "bc":
        candidate = os.path.join(project_root, "models", "bc", os.path.basename(model_path))
    else:
        candidate = os.path.join(project_root, "models", "magail", os.path.basename(model_path))
    if os.path.exists(candidate):
        return candidate
    if policy_type == "magail" and not model_path.endswith("_actor.pth"):
        candidate = os.path.join(project_root, "models", "magail", os.path.basename(model_path) + "_actor.pth")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Model path {model_path} not found.")


def _run_policy(args):
    from Env.bc_env import BCScenarioEnv
    from metadrive.engine.engine_utils import close_engine

    policy_type = (args.policy_type or "auto").lower()
    if policy_type == "auto":
        policy_type = "bc" if args.model_path.endswith(".pt") else "magail"

    data_dir = _resolve_data_dir(args.data_dir)
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
            if len(obs_dict) == 0:
                print(f"Scenario {i} has no controlled agents (all filtered out). Skipping.")
                continue
            step_count = 0
            episode_reward = 0.0

            while True:
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

                actions = {aid: actions_np[idx].flatten() for idx, aid in enumerate(agent_ids)}
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


# --- Trajectory (matplotlib 2D animation) ---
def _build_expert_trajectories_from_env(env):
    """Build expert_trajectories dict from env (ExpertReplayEnv has traffic_manager.current_traffic_data)."""
    if hasattr(env, "expert_trajectories") and env.expert_trajectories:
        return env.expert_trajectories
    if not hasattr(env, "engine") or not hasattr(env.engine, "traffic_manager"):
        return {}
    from metadrive.type import MetaDriveType
    data = getattr(env.engine.traffic_manager, "current_traffic_data", None)
    if not data:
        return {}
    expert_trajs = {}
    for scenario_id, track in data.items():
        if track.get("type") != MetaDriveType.VEHICLE or "state" not in track:
            continue
        state = track["state"]
        positions = state.get("position")
        if positions is None:
            continue
        valid = state.get("valid", np.ones(len(positions), dtype=bool))
        valid = np.asarray(valid).flatten()
        if valid.size != len(positions):
            valid = np.ones(len(positions), dtype=bool)
        first_show = int(np.argmax(valid)) if valid.any() else 0
        last_show = len(valid) - 1 - int(np.argmax(valid[::-1])) if valid.any() else len(positions) - 1
        obj_id = track.get("metadata", {}).get("object_id", str(scenario_id))
        expert_trajs[obj_id] = {
            "positions": np.asarray(positions),
            "start_timestep": first_show,
            "end_timestep": last_show,
        }
    return expert_trajs


def _run_trajectory_animation(expert_trajs, scenario_idx):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(expert_trajs) == 0:
        print("No expert trajectories to visualize.")
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    max_timestep = max(t["end_timestep"] for t in expert_trajs.values())
    min_timestep = min(t["start_timestep"] for t in expert_trajs.values())

    colors = plt.cm.tab10(np.linspace(0, 1, len(expert_trajs)))
    for idx, (obj_id, traj) in enumerate(expert_trajs.items()):
        positions = np.asarray(traj["positions"])
        if positions.ndim >= 2:
            positions = positions[:, :2]
        else:
            continue
        ax.plot(
            positions[:, 0], positions[:, 1],
            color=colors[idx], alpha=0.3, linewidth=1,
            label=f"Vehicle {str(obj_id)[:6]}",
        )

    scatter = ax.scatter([], [], s=200, c="red", marker="o", edgecolors="black", linewidths=2)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Expert Trajectory Visualization - Scenario {scenario_idx}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    def update(frame):
        current_time = min_timestep + frame
        current_positions = []
        for traj in expert_trajs.values():
            st, et = traj["start_timestep"], traj["end_timestep"]
            if st <= current_time <= et:
                pos = np.asarray(traj["positions"])
                if pos.ndim >= 2:
                    pos = pos[current_time - st, :2]
                else:
                    continue
                current_positions.append(pos)
        if current_positions:
            scatter.set_offsets(np.array(current_positions))
        time_text.set_text(f"Time: {frame * 0.1:.1f}s (Frame {frame})")
        return scatter, time_text

    anim = FuncAnimation(
        fig, update, frames=max_timestep - min_timestep + 1,
        interval=100, blit=True, repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return anim


def _run_trajectory(args):
    from Env.expert_replay_env import ExpertReplayEnv

    data_dir = _resolve_data_dir(args.data_dir)
    data_path = os.path.abspath(data_dir)
    env_config = {
        "data_directory": data_path,
        "is_multi_agent": True,
        "num_controlled_agents": 100,
        "horizon": 500,
        "use_render": False,
        "sequential_seed": True,
        "reactive_traffic": False,
        "start_scenario_index": args.scenario_idx,
        "num_scenarios": 1,
        "log_level": 40,
    }

    env = ExpertReplayEnv(config=env_config)
    try:
        env.reset(seed=args.scenario_idx)
        expert_trajs = _build_expert_trajectories_from_env(env)
        _run_trajectory_animation(expert_trajs, args.scenario_idx)
    finally:
        env.close()


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Unified visualization: replay, policy (BC/MAGAIL), trajectory.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="replay | policy | trajectory")

    # Common args for data_dir (used by all)
    def add_common_data_args(p):
        p.add_argument("--data_dir", type=str, default="data/exp_filtered", help="Waymo scenario directory")
        p.add_argument("--start_index", type=int, default=0)
        p.add_argument("--num_scenarios", type=int, default=1)
        p.add_argument("--horizon", type=int, default=200)

    # replay
    pr = subparsers.add_parser("replay", help="Replay scenario with ExpertReplayEnv (no policy)")
    add_common_data_args(pr)
    pr.set_defaults(horizon=500)

    # policy
    pp = subparsers.add_parser("policy", help="Visualize BC or MAGAIL trained policy")
    add_common_data_args(pp)
    pp.add_argument("--policy_type", type=str, default="auto", choices=["auto", "bc", "magail"])
    pp.add_argument("--model_path", type=str, default="models/bc/policy_best.pt")
    pp.add_argument("--deterministic", action="store_true", help="MAGAIL: use mean action")

    # trajectory
    pt = subparsers.add_parser("trajectory", help="2D matplotlib animation of expert trajectories")
    pt.add_argument("--data_dir", type=str, default="data/exp_filtered")
    pt.add_argument("--scenario_idx", type=int, default=0)

    args = parser.parse_args()

    # Resolve data_dir relative to project root when default
    if args.mode != "trajectory":
        if args.data_dir in ("data/exp_filtered", "data/exp_converted"):
            args.data_dir = os.path.join(project_root, args.data_dir)
    else:
        if args.data_dir in ("data/exp_filtered", "data/exp_converted"):
            args.data_dir = os.path.join(project_root, args.data_dir)

    if args.mode == "replay":
        _run_replay(args)
    elif args.mode == "policy":
        _run_policy(args)
    elif args.mode == "trajectory":
        _run_trajectory(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
