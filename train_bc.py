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
from dataset.loader import load_expert_pkl, get_expert_scenario_ids


def evaluate_policy(policy, args, device):
    """在 BCScenarioEnv 中评估策略：仅使用专家数据中出现过的 scenario_id，保证 eval 有受控车。
    输出与 replay 对齐：agents (current)=reset 时受控车数，total in scenario=该场景受控轨迹总数（car_birth_info_list 长度）。"""
    waymo_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    data_dir = os.path.join(waymo_data_dir, "exp_filtered")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(waymo_data_dir, "exp_converted")
    if not os.path.exists(data_dir):
        print(f"[ERROR] Could not find scenario data in {waymo_data_dir}. Evaluation skipped.")
        return 0.0

    scenario_ids = get_expert_scenario_ids(args.expert_data_path, max_ids=5)
    if not scenario_ids:
        print("[WARN] No scenario_id in expert pkl, falling back to scenarios [0,1,2]. Eval may have 0 controlled agents.")
        scenario_ids = [0, 1, 2]

    total_rewards = []
    horizon = 200

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
        }
        env = BCScenarioEnv(env_config, agent2policy=None)
        try:
            obs_dict = env.reset(seed=scenario_id)
        except Exception as e:
            print(f"  Eval Episode {idx} (scenario {scenario_id}): reset failed: {e}")
            env.close()
            continue

        n_controlled = len(env.controlled_agents)
        n_total_in_scenario = getattr(env, "num_controlled_in_scenario", n_controlled)
        if n_controlled == 0:
            print(
                f"  Eval Episode {idx} (scenario {scenario_id}): 0 controlled agents (total in scenario: {n_total_in_scenario}), skip."
            )
            env.close()
            continue

        episode_reward = 0.0
        step_count = 0
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
            obs_dict, rewards, dones, _ = env.step(action_dict)
            episode_reward += sum(rewards.values())

        total_rewards.append(episode_reward)
        print(
            f"  Eval Episode {idx} (scenario {scenario_id}): Total Reward {episode_reward:.2f}, steps {step_count}, "
            f"agents (current): {n_controlled}, total in scenario: {n_total_in_scenario}"
        )
        env.close()

    if not total_rewards:
        print("  No valid eval episodes (all skipped or failed).")
        return 0.0
    avg_reward = float(np.mean(total_rewards))
    print(f"  Average Evaluation Reward: {avg_reward:.2f}")
    return avg_reward


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("logs/bc", exist_ok=True)
    log_dir = os.path.join("logs", "bc", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    obs_data, act_data = load_expert_pkl(args.expert_data_path)
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

        if (epoch + 1) % args.eval_freq == 0:
            eval_reward = evaluate_policy(policy, args, device)
            writer.add_scalar("Reward/eval", eval_reward, epoch)

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
    args = parser.parse_args()
    main(args)
