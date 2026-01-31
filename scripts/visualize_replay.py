import argparse
import os
import sys
import time
import pickle

# Add project root to Python path so we can import Env module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Env.expert_replay_env import ExpertReplayEnv

def _scan_for_large_acc(env, horizon, min_abs_acc):
    max_abs_acc = 0.0
    for step in range(horizon):
        _, _, dones, infos = env.step(None)
        for info in infos.values():
            raw = info.get("expert_action_raw")
            if raw is None:
                continue
            raw_acc = abs(raw.get("raw_acc", 0.0))
            if raw_acc > max_abs_acc:
                max_abs_acc = raw_acc
            if raw_acc >= min_abs_acc:
                return True, max_abs_acc, step
        if dones["__all__"]:
            break
    return False, max_abs_acc, None

def visualize_replay(args):
    data_path = os.path.abspath(args.data_dir)
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory {data_path} not found")

    # Load dataset summary for scenario id display
    summary_path = os.path.join(data_path, "dataset_summary.pkl")
    scenario_ids = None
    if os.path.exists(summary_path):
        with open(summary_path, "rb") as f:
            summary = pickle.load(f)
        scenario_ids = list(summary.keys()) if isinstance(summary, dict) else list(summary)

    # Same as data generation: avoid MetaDrive assertion when requested num_scenarios > available.
    from metadrive.scenario.utils import read_dataset_summary
    _, summary_lookup, _ = read_dataset_summary(data_path)
    if args.start_index >= len(summary_lookup):
        raise ValueError(
            f"start_index={args.start_index} out of range. Dataset has {len(summary_lookup)} scenarios."
        )
    max_available = len(summary_lookup) - args.start_index
    num_to_run = min(args.num_scenarios, max_available)

    base_env_config = {
        "data_directory": data_path,
        "is_multi_agent": True,
        "num_controlled_agents": 100,
        "horizon": args.horizon,
        "use_render": True,  # Enable rendering
        "sequential_seed": True,
        "reactive_traffic": False,
        "num_scenarios": -1,
        "log_level": 40, # ERROR
        # "pstats": True, # For performance debugging
    }

    try:
        for i in range(args.start_index, args.start_index + num_to_run):
            print(f"\n--- Playing Scenario {i} ---")
            env_config = dict(base_env_config)
            env_config["start_scenario_index"] = args.start_index
            print(f"Initializing ExpertReplayEnv with data from {data_path} (start_index={i})...")
            env = ExpertReplayEnv(config=env_config)
            try:
                obs = env.reset(seed=i)
            except Exception as e:
                print(f"Error resetting scenario {i}: {e}")
                try:
                    env.close()
                except Exception:
                    pass
                continue
            
            print(
                "Scenario loaded. "
                f"Controlled agents: {len(env.controlled_agents)} | "
                f"scenario_id: {getattr(env, 'current_scenario_id', None)} | "
                f"scenario_index: {getattr(env, 'current_scenario_index', None)}"
            )
            if scenario_ids is not None and 0 <= i < len(scenario_ids):
                print(f"Dataset scenario file: {scenario_ids[i]}")

            if args.min_abs_acc is not None:
                hit, max_abs_acc, hit_step = _scan_for_large_acc(env, args.horizon, args.min_abs_acc)
                # Close the env used for scanning to avoid reset issues across scenarios.
                try:
                    env.close()
                except Exception:
                    pass
                if not hit:
                    print(
                        f"Skip scenario {i}: max |raw_acc|={max_abs_acc:.3f} < {args.min_abs_acc:.3f}"
                    )
                    continue

                print(
                    f"Scenario {i} hit |raw_acc|>={args.min_abs_acc:.3f} "
                    f"(max={max_abs_acc:.3f}, first_hit_step={hit_step})"
                )

                # Recreate env and play with rendering
                env_config = dict(base_env_config)
                env_config["start_scenario_index"] = args.start_index
                print(f"Initializing ExpertReplayEnv with data from {data_path} (start_index={i})...")
                env = ExpertReplayEnv(config=env_config)
                try:
                    obs = env.reset(seed=i)
                except Exception as e:
                    print(f"Error resetting scenario {i}: {e}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    continue
            
            for step in range(args.horizon):
                # Step
                obs, rewards, dones, infos = env.step(None)
                
                # Render
                env.render(
                    mode="top_down",
                    text={
                        "Step": step,
                        "Agents": len(env.controlled_agents),
                        "Scenario": i,
                        "ScenarioID": getattr(env, "current_scenario_id", None),
                        "ScenarioIdx": getattr(env, "current_scenario_index", None),
                    },
                )
                
                # Sleep to control playback speed
                time.sleep(0.05)
                
                if dones["__all__"]:
                    print(f"Scenario {i} finished at step {step}")
                    break
            try:
                env.close()
            except Exception:
                pass
                    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Global error: {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("Environment closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/huangfukk/MAGAIL4AutoDrive/data/exp_filtered", help="Path to Waymo data")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--min_abs_acc", type=float, default=None, help="Only play scenarios with |raw_acc| >= this value")
    
    args = parser.parse_args()
    visualize_replay(args)
