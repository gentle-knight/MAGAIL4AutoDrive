"""
BC 训练脚本：负责数据加载、环境评估、日志与保存；BC 算法由 Algorithm.bc 提供。
使用方式不变：python train_bc.py [--expert_data_path data/training_data] [--save_dir models/bc] ...
"""
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Algorithm.policy import StateIndependentPolicy
from Algorithm.bc import train_bc_epoch, eval_bc_epoch
from Env.bc_env import BCScenarioEnv
from Env.bc_ego_replay_env import BCEgoReplayEnv
from dataset.loader import load_expert_pkl, get_expert_scenario_ids


def evaluate_policy(policy, args, device):
    """在 BCScenarioEnv（多智能体）或 BCEgoReplayEnv（单智能体）中评估策略。
    仅使用专家数据中出现过的 scenario_id。单智能体模式下仅 ego 受策略控制，其他车专家回放。"""
    waymo_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    data_dir = os.path.join(waymo_data_dir, "exp_filtered")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(waymo_data_dir, "exp_converted")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Could not find scenario data in {waymo_data_dir}. Evaluation skipped.")
        return 0.0, 0.0, 0.0

    scenario_ids = get_expert_scenario_ids(args.expert_data_path, max_ids=5)
    if not scenario_ids:
        print("[WARN] No scenario_id in expert pkl, falling back to scenarios [0,1,2]. Eval may have 0 controlled agents.")
        scenario_ids = [0, 1, 2]

    total_rewards = []
    total_steps = []
    collision_episodes = 0
    horizon = 200
    single_agent = getattr(args, "single_agent", False)

    for idx, scenario_id in enumerate(scenario_ids):
        env_config = {
            "data_directory": data_dir,
            "is_multi_agent": True,
            "num_controlled_agents": 100,
            "use_render": False,
            "sequential_seed": True,
            "horizon": horizon,
            "start_scenario_index": scenario_id,
            "num_scenarios": 1,
            "log_level": 50,
        }
        if single_agent:
            env = BCEgoReplayEnv(config=env_config)
        else:
            env = BCScenarioEnv(env_config, agent2policy=None)
        try:
            obs_dict = env.reset(seed=scenario_id)
        except Exception as e:
            print(f"  Eval Episode {idx} (scenario {scenario_id}): reset failed: {e}")
            env.close()
            continue

        n_controlled = len(env.controlled_agents)
        n_total_in_scenario = getattr(env, "num_controlled_in_scenario", n_controlled) if not single_agent else 1
        if n_controlled == 0:
            print(
                f"  Eval Episode {idx} (scenario {scenario_id}): 0 controlled agents (total in scenario: {n_total_in_scenario}), skip."
            )
            env.close()
            continue

        episode_reward = 0.0
        step_count = 0
        had_near_collision = False
        dones = {"__all__": False}
        while not dones["__all__"] and step_count < horizon:
            step_count += 1
            if not obs_dict:
                obs_dict, _, dones, _ = env.step({})
                continue
            agent_ids = list(obs_dict.keys())
            obs_list = [obs_dict[aid] for aid in agent_ids]
            obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)
            with torch.no_grad():
                actions, _ = policy.sample(obs_tensor)
                actions = actions.cpu().numpy()
            action_dict = {aid: act for aid, act in zip(agent_ids, actions)}
            obs_dict, rewards, dones, infos = env.step(action_dict)
            episode_reward += sum(rewards.values())
            if infos:
                for _aid, info in infos.items():
                    if isinstance(info, dict) and info.get("near_collision", False):
                        had_near_collision = True
                        break

        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        if had_near_collision:
            collision_episodes += 1
        mode_str = "single-agent (ego)" if single_agent else f"agents (current): {n_controlled}, total in scenario: {n_total_in_scenario}"
        print(
            f"  Eval Episode {idx} (scenario {scenario_id}): Total Reward {episode_reward:.2f}, steps {step_count}, {mode_str}"
        )
        env.close()

    if not total_rewards:
        print("  No valid eval episodes (all skipped or failed).")
        return 0.0, 0.0, 0.0
    avg_reward = float(np.mean(total_rewards))
    avg_steps = float(np.mean(total_steps)) if total_steps else 0.0
    collision_rate = float(collision_episodes / max(1, len(total_rewards)))
    print(
        f"  Average Evaluation Reward: {avg_reward:.2f} | Mean Episode Length: {avg_steps:.1f} | "
        f"Collision Rate (near): {collision_rate:.3f}"
    )
    return avg_reward, collision_rate, avg_steps


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("logs/bc", exist_ok=True)
    log_dir = os.path.join("logs", "bc", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    agent_id_filter = "default_agent" if getattr(args, "single_agent", False) else None
    obs_data, act_data = load_expert_pkl(
        args.expert_data_path,
        filter_terminal_last_step=args.filter_terminal_last_step,
        agent_id_filter=agent_id_filter,
    )
    obs_tensor = torch.FloatTensor(obs_data)
    act_tensor = torch.FloatTensor(act_data)
    dataset = TensorDataset(obs_tensor, act_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    state_dim = obs_data.shape[1]
    action_dim = act_data.shape[1]
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

    policy = StateIndependentPolicy(
        state_shape=(state_dim,),
        action_shape=(action_dim,),
        hidden_units=(256, 256),
        hidden_activation=torch.nn.Tanh(),
    ).to(device)
    optimizer = Adam(policy.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        avg_train_loss = train_bc_epoch(policy, train_loader, optimizer, device)
        scheduler.step()
        avg_val_loss = eval_bc_epoch(policy, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(policy.state_dict(), os.path.join(args.save_dir, "policy_best.pt"))

        # Periodic checkpointing (II-style)
        if args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0:
            torch.save(policy.state_dict(), os.path.join(args.save_dir, f"policy_epoch{epoch+1}.pt"))

        if (epoch + 1) % args.eval_freq == 0:
            eval_reward, eval_collision_rate, eval_mean_steps = evaluate_policy(policy, args, device)
            writer.add_scalar("Reward/eval", eval_reward, epoch)
            writer.add_scalar("Eval/collision_rate_near", eval_collision_rate, epoch)
            writer.add_scalar("Eval/mean_episode_length", eval_mean_steps, epoch)

    torch.save(policy.state_dict(), os.path.join(args.save_dir, "policy_final.pt"))
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data_path", type=str, default="data/training_data", help="Path to expert data pickle or directory")
    parser.add_argument("--save_dir", type=str, default="models/bc", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--checkpoint_freq", type=int, default=50, help="Save policy_epochN.pt every N epochs. Set <=0 to disable.")
    parser.add_argument(
        "--filter_terminal_last_step",
        action="store_true",
        help="Drop the last (obs, act) pair of each trajectory to approximate training on non-terminal steps (II-style).",
    )
    parser.add_argument(
        "--single_agent",
        action="store_true",
        help="Use single-agent (ego) expert data and evaluation; load only default_agent trajectories and evaluate with BCEgoReplayEnv.",
    )
    args = parser.parse_args()
    main(args)
