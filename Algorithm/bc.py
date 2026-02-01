"""
Behavior Cloning (BC) 算法：仅包含损失与单 epoch 训练/评估逻辑。
数据加载、环境评估、日志与保存由训练脚本 (train_bc.py) 负责。
"""
import torch


def bc_loss(policy, states, actions):
    """
    BC 损失：负对数似然 -E[log pi(a|s)]。
    states: (B, state_dim), actions: (B, action_dim), 均在 policy 所在 device 上。
    """
    log_pi = policy.evaluate_log_pi(states, actions)
    return -log_pi.mean()


def train_bc_epoch(policy, train_loader, optimizer, device):
    """
    训练一个 epoch，返回平均 train loss。
    policy 与 optimizer 由调用方管理，本函数只做前向、损失、反向与 step。
    """
    policy.train()
    total_loss = 0.0
    n_batches = 0
    for states, actions in train_loader:
        states = states.to(device)
        actions = actions.to(device)
        loss = bc_loss(policy, states, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches else 0.0


def eval_bc_epoch(policy, val_loader, device):
    """
    在验证集上评估一个 epoch，返回平均 val loss（无梯度）。
    """
    policy.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(device)
            actions = actions.to(device)
            log_pi = policy.evaluate_log_pi(states, actions)
            loss = -log_pi.mean().item()
            total_loss += loss
            n_batches += 1
    return total_loss / n_batches if n_batches else 0.0
