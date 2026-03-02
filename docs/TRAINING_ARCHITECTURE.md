# MAGAIL 训练方案架构文档

## 目录
1. [训练数据结构](#1-训练数据结构)
2. [多智能体训练机制](#2-多智能体训练机制)
3. [完整训练流程](#3-完整训练流程)
4. [当前项目问题](#4-当前项目问题)
5. [TensorBoard 日志问题](#5-tensorboard-日志问题)

---

## 1. 训练数据结构

### 1.1 数据维度

**观测空间 (Observation Space)**
- **维度**: 45维
- **组成**:
  - **Ego状态** (5维): `[position_x, position_y, velocity_x, velocity_y, heading_theta]`
  - **邻居信息** (40维): 最多10个邻居，每个邻居4维特征
    - 每个邻居: `[relative_x, relative_y, velocity_x, velocity_y]`
    - 如果邻居数量 < 10，用零填充

**动作空间 (Action Space)**
- **维度**: 2维
- **组成**: `[steering, accel]`
- **范围**: 归一化到 `[-1, 1]`

### 1.2 数据格式

**专家数据文件结构** (`.pkl` 文件):
```python
# 每个 .pkl 文件包含一个列表，每个元素是一条车辆轨迹
trajectories = [
    {
        'obs': np.array,      # Shape: (T, 45) - T为轨迹长度（可变）
        'acts': np.array,     # Shape: (T, 2) - 对应的动作序列
        'agent_id': str,      # 车辆ID
        'scenario_id': int    # 场景ID
    },
    ...
]
```

**数据特点**:
- 轨迹长度 `T` 是**可变的**，取决于车辆在场景中的存活时间
- 最小轨迹长度过滤: 只保留长度 > 10 的轨迹
- 数据已通过静态车辆过滤（移动距离 < 5m 且最大速度 < 1m/s 的车辆被过滤）

### 1.3 数据生成流程

**脚本**: `scripts/generate_expert_data.py`

**流程**:
1. 从 Waymo 数据 (`data/exp_filtered`) 加载场景
2. 使用 `ExpertReplayEnv` 回放专家轨迹
3. 通过逆动力学 (`Env/inverse_dynamics.py`) 计算动作
4. 构建45维观测（Ego + 10个最近邻居）
5. 过滤无效轨迹（长度 < 10）
6. 保存为 `.pkl` 文件到 `data/training_data/`

**关键代码位置**:
- 观测构建: `Env/expert_replay_env.py` 的 `_get_all_obs()` 方法
- 动作计算: `Env/inverse_dynamics.py` 的 `compute_action()` 方法

---

## 2. 多智能体训练机制

### 2.1 可变长度处理

**问题**: 不同场景中智能体数量不同，每个智能体的轨迹长度也不同。

**解决方案**:

1. **数据层面** (`dataset/magail_dataset.py`):
   - 将轨迹**展平**为独立的 `(state, action)` 对
   - 每个样本是独立的，不保留序列信息
   - 这样所有轨迹可以统一处理，不受长度限制

```python
# MAGAILExpertDataset 的处理方式
for traj in self.trajectories:
    obs = traj['obs']  # (T, 45)
    acts = traj['acts']  # (T, 2)
    # 展平为独立样本
    for i in range(len(obs)):
        self.flat_data.append((obs[i], acts[i]))  # 每个样本: (45,), (2,)
```

2. **训练环境层面** (`train_magail.py`):
   - 每个 episode 动态处理不同数量的智能体
   - 在 rollout 循环中，为每个活跃智能体独立收集数据
   - 所有智能体的数据合并到一个 `memory` 中

```python
# Rollout 循环
for agent_id, obs in obs_dict.items():
    act, logprob = ppo_agent.select_action(obs)
    actions[agent_id] = act
    # 所有智能体的数据都存入同一个 memory
    memory['states'].append(obs)
    memory['actions'].append(actions[agent_id])
    ...
```

3. **观测维度固定**:
   - 通过 `MAGAILScenarioEnv` 确保观测维度始终为45维
   - 邻居数量不足时用零填充，保证维度一致

### 2.2 多智能体交互

**环境设置**:
- 使用 `MAGAILScenarioEnv` (继承自 `MultiAgentScenarioEnv`)
- 自定义 `_get_all_obs()` 方法，确保观测格式与专家数据一致
- 每个智能体独立选择动作，环境统一执行

**关键点**:
- 所有智能体共享同一个策略网络（参数共享）
- 每个智能体独立计算动作和奖励
- 数据收集时将所有智能体的经验合并

---

## 3. 完整训练流程

### 3.1 数据准备阶段

**步骤 1: 生成专家数据**
```bash
python scripts/generate_expert_data.py \
    --data_dir data/exp_filtered \
    --output_dir data/training_data \
    --num_scenarios 100 \
    --start_index 0
```

**输出**: `data/training_data/expert_data_*.pkl`

### 3.2 模型初始化

**网络架构**:

1. **Actor (策略网络)**:
   - 输入: 45维状态
   - 输出: 2维动作（连续）
   - 结构: MLP (45 → 256 → 256 → 2)
   - 输出分布: 高斯分布（均值 + 可学习标准差）

2. **Critic (价值网络)**:
   - 输入: 45维状态
   - 输出: 标量价值
   - 结构: MLP (45 → 256 → 256 → 1)

3. **Discriminator (鉴别器)**:
   - 输入: 45维状态 + 2维动作 = 47维
   - 输出: 标量（0-1之间，表示专家概率）
   - 结构: MLP (47 → 256 → 256 → 1) + Sigmoid

### 3.3 训练循环

**主循环** (`train_magail.py` 的 `train()` 函数):

```
For each episode:
    1. 收集 Rollout
       - 重置环境（随机选择场景）
       - 运行策略收集轨迹
       - 存储 (state, action, logprob, next_state, done)
    
    2. 训练 Discriminator
       - 采样专家批次
       - 采样策略批次
       - 更新鉴别器:
         - Expert loss: BCE(D(s_e, a_e), 1)
         - Policy loss: BCE(D(s_p, a_p), 0)
         - Total: L_d = L_expert + L_policy
    
    3. 计算 GAIL 奖励
       - 对所有策略状态-动作对:
         reward = -log(1 - D(s, a) + ε)
       - 替换环境奖励
    
    4. 更新策略 (PPO)
       - 计算 GAE (Generalized Advantage Estimation)
       - PPO 更新 (K epochs):
         - 计算优势函数
         - 计算策略损失（带clip）
         - 计算价值损失
         - 更新 Actor 和 Critic
```

### 3.4 训练目标

**Discriminator 目标**:
```
L_D = E_{(s,a)~π_E}[-log(D(s,a))] + E_{(s,a)~π_θ}[-log(1-D(s,a))]
```
- 最大化区分专家数据和策略数据的能力

**Policy (Generator) 目标**:
```
L_π = E_{(s,a)~π_θ}[-log(D(s,a))] - λ_H(π_θ)
```
- 通过 PPO 优化，使用 GAIL 奖励作为信号
- 最大化鉴别器给出的"专家概率"
- 同时保持策略熵（探索）

**PPO 更新**:
```python
# 优势函数 (GAE)
advantages = compute_gae(rewards, values, next_values, dones, gamma, lambda)

# 策略损失
ratios = exp(log_probs - old_log_probs)
surr1 = ratios * advantages
surr2 = clip(ratios, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2) + 0.01 * entropy

# 价值损失
value_loss = MSE(critic(states), returns)

# 总损失
total_loss = policy_loss + 0.5 * value_loss
```

### 3.5 关键代码位置

- **训练主循环**: `train_magail.py:278-505`
- **PPO 更新**: `train_magail.py:90-146`
- **Discriminator 更新**: `train_magail.py:429-462`
- **GAIL 奖励计算**: `train_magail.py:472-477`

---

## 4. 当前项目问题

### 4.1 环境重置问题

**问题描述**:
- MetaDrive 环境在快速重置时可能出现对象清理不完整的问题
- 错误信息: "You should clear all generated objects..."

**当前处理**:
- 代码中已有异常处理机制（`train_magail.py:288-342`）
- 重置失败时会尝试关闭并重新创建环境
- 但可能导致训练不稳定

**建议修复**:
- 在每次重置前显式清理所有对象
- 增加重置间隔，避免过于频繁的重置
- 考虑使用环境池（Environment Pool）复用环境实例

### 4.2 观测维度对齐

**问题描述**:
- 原始 `MultiAgentScenarioEnv` 返回108维观测（包含Lidar）
- 专家数据使用45维观测
- 维度不匹配会导致训练失败

**当前解决方案**:
- 通过 `MAGAILScenarioEnv` 重写 `_get_all_obs()` 方法
- 确保训练环境与专家数据使用相同的观测格式

**代码位置**: `train_magail.py:223-262`

### 4.3 数据收集效率

**问题描述**:
- 每个 episode 都需要完整运行环境收集数据
- 可变长度轨迹导致 batch 大小不一致
- 可能影响训练稳定性

**当前处理**:
- 使用展平的数据集，每个样本独立
- 在 rollout 时收集所有智能体的数据，合并处理

**潜在改进**:
- 考虑使用经验回放缓冲区
- 实现轨迹级别的采样（保留序列信息）

### 4.4 内存管理

**问题描述**:
- 长时间训练可能导致内存泄漏
- 环境对象可能没有完全释放

**当前处理**:
- 代码中有显式的 `gc.collect()` 和 `torch.cuda.empty_cache()`
- 但可能不够彻底

**建议**:
- 定期检查内存使用
- 考虑限制 rollout 长度
- 使用更激进的清理策略

### 4.5 训练稳定性

**问题描述**:
- Discriminator 可能过早收敛，导致策略无法学习
- GAIL 奖励可能不稳定

**当前处理**:
- 使用标准的 GAIL 奖励公式: `-log(1 - D(s,a) + ε)`
- PPO 的 clip 机制提供稳定性

**潜在改进**:
- 考虑使用 WGAN-GP 或 LSGAN 损失
- 实现 Discriminator 的预训练
- 添加奖励归一化

---

## 5. TensorBoard 日志问题

### 5.1 问题分析

**现象**:
- `runs/magail_0112/` 目录下只有模型文件（`.pth`），没有 TensorBoard 事件文件（`events.out.tfevents.*`）
- 其他目录（`magail_full`, `magail_production`）有事件文件

**可能原因**:

1. **TensorBoard 未安装**:
   - 代码中有 try-except 处理（`train_magail.py:269-274`）
   - 如果 TensorBoard 未安装，`writer` 会被设置为 `None`
   - 训练会继续，但不会写入日志

2. **日志写入失败**:
   - 即使 `SummaryWriter` 创建成功，如果写入时出错，可能不会生成文件
   - 需要检查是否有异常被静默捕获

3. **训练中断**:
   - 如果训练在写入第一个日志前中断，可能没有事件文件
   - 但模型文件已保存，说明训练至少运行了一段时间

### 5.2 检查方法

**步骤 1: 检查 TensorBoard 安装**
```bash
python -c "import tensorboard; print(tensorboard.__version__)"
```

**步骤 2: 检查训练脚本中的日志写入**
查看 `train_magail.py:493-496`:
```python
if writer:
    writer.add_scalar('Loss/Discriminator', disc_loss.item(), i_episode)
    writer.add_scalar('Loss/Policy', ppo_loss, i_episode)
    writer.add_scalar('Reward/Mean_GAIL', np.mean(all_gail_rewards), i_episode)
```

**步骤 3: 检查日志目录权限**
```bash
ls -la runs/magail_0112/
```

### 5.3 解决方案

**方案 1: 确保 TensorBoard 已安装**
```bash
pip install tensorboard
```

**方案 2: 添加显式刷新**
在训练循环结束后，显式调用 `writer.flush()`:
```python
if writer:
    writer.flush()  # 确保数据写入磁盘
```

**方案 3: 添加日志验证**
在训练开始时检查日志目录:
```python
if writer:
    # 测试写入
    writer.add_scalar('Test/Initialization', 0.0, 0)
    writer.flush()
    print(f"TensorBoard logging enabled. Log dir: {args.log_dir}")
else:
    print("WARNING: TensorBoard not available. Logging disabled.")
```

**方案 4: 使用文件日志作为备份**
即使 TensorBoard 不可用，也可以写入文本日志:
```python
import logging
logging.basicConfig(
    filename=os.path.join(args.log_dir, 'training.log'),
    level=logging.INFO
)
```

### 5.4 代码修复建议

**在 `train_magail.py` 中添加以下改进**:

1. **确保 disc_loss 在 CPU 上**:
```python
# 第425行附近
disc_loss = torch.tensor(0.0).cuda()  # 改为 .cuda() 或保持 CPU
# 或者在使用时转换
if writer:
    disc_loss_value = disc_loss.item() if isinstance(disc_loss, torch.Tensor) else disc_loss
    writer.add_scalar('Loss/Discriminator', disc_loss_value, i_episode)
```

2. **添加显式刷新**:
```python
# 第496行后添加
if writer:
    writer.flush()  # 确保数据写入磁盘
```

3. **添加初始化验证**:
```python
# 第271行后添加
if writer:
    # 测试写入
    writer.add_scalar('Test/Initialization', 0.0, 0)
    writer.flush()
    print(f"✓ TensorBoard logging enabled. Log dir: {args.log_dir}")
    # 检查文件是否创建
    import glob
    event_files = glob.glob(os.path.join(args.log_dir, "events.out.tfevents.*"))
    if event_files:
        print(f"✓ TensorBoard event file created: {event_files[0]}")
else:
    print("⚠ WARNING: TensorBoard not available. Logging disabled.")
```

4. **在训练结束时确保关闭**:
```python
# 第505行后添加
if writer:
    writer.flush()  # 最后一次刷新
    writer.close()
    print(f"TensorBoard logs saved to {args.log_dir}")
```

### 5.5 验证修复

**重新训练测试**:
```bash
python train_magail.py \
    --expert_data_dir data/training_data \
    --data_dir data/exp_filtered \
    --batch_size 1024 \
    --max_episodes 10 \
    --log_dir runs/test_tensorboard
```

**检查输出**:
```bash
# 应该看到事件文件
ls runs/test_tensorboard/events.out.tfevents.*

# 启动 TensorBoard
tensorboard --logdir runs/test_tensorboard
```

**对于 magail_0112 训练**:
由于该训练已经完成且没有日志文件，建议：
1. 检查训练时的控制台输出，确认是否有 "TensorBoard not installed" 消息
2. 如果确实没有 TensorBoard，可以重新运行少量 episode 来验证修复
3. 或者查看是否有其他日志文件（如 `training.log`）

---

## 附录: 关键文件清单

### 核心训练文件
- `train_magail.py`: 主训练脚本
- `dataset/magail_dataset.py`: 专家数据集加载
- `Env/expert_replay_env.py`: 专家回放环境
- `Env/scenario_env.py`: 多智能体场景环境
- `Env/inverse_dynamics.py`: 逆动力学计算

### 数据生成文件
- `scripts/generate_expert_data.py`: 专家数据生成
- `scripts/visualize_replay.py`: 数据可视化
- `scripts/analyze_expert_data.py`: 数据分析

### 配置文件
- `README.md`: 项目说明
- `TRAINING_ARCHITECTURE.md`: 本文档

---

## 总结

本项目的 MAGAIL 训练方案通过以下方式处理多智能体可变长度问题：

1. **数据层面**: 将轨迹展平为独立样本，统一处理
2. **环境层面**: 动态处理不同数量的智能体，合并经验
3. **网络层面**: 固定输入维度（45维），通过零填充处理邻居不足的情况

训练流程遵循标准的 GAIL 框架，使用 PPO 作为策略优化算法。当前主要问题集中在环境稳定性和日志记录方面，需要进一步优化。
