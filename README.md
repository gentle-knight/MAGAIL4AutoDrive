# MAGAIL4AutoDrive

基于 **MetaDrive** 仿真器和 **Waymo Open Motion Dataset** 的自动驾驶多智能体模仿学习（MAGAIL）与行为克隆（BC）训练系统。

本项目旨在从真实的 Waymo 驾驶数据中提取专家轨迹，并通过模仿学习（Imitation Learning）训练能够适应复杂交互场景的自动驾驶策略。

## 目录结构

```text
MAGAIL4AutoDrive/
├── Algorithm/                 # 强化学习与模仿学习算法实现
│   ├── policy.py              # 基础策略网络 (MLP 等)
│   ├── ppo.py                 # PPO 算法实现
│   ├── magail.py              # MAGAIL 算法核心逻辑
│   ├── disc.py                # 判别器 (Discriminator) 网络
│   └── ...
├── Env/                       # 仿真环境封装 (MetaDrive Wrapper)
│   ├── bc_env.py              # BCScenarioEnv，45 维观测（BC/MAGAIL 共用）
│   ├── scenario_env.py       # 多智能体基础场景环境
│   ├── expert_replay_env.py  # 专家轨迹回放环境（数据生成与回放）
│   ├── inverse_dynamics.py   # 逆动力学模块 (轨迹 -> 动作)
│   ├── simple_idm_policy.py  # ConstantVelocityPolicy 占位策略
│   └── ...
├── dataset/                   # 数据集加载器
│   ├── loader.py              # 主流水线：load_expert_pkl、MAGAILExpertDataset
│   └── expert_dataset.py      # 可选 107 维/5 维管线
├── scripts/                   # 工具脚本（数据、回放、可视化、分析）
│   ├── generate_expert_data.py    # 从 Waymo 生成专家 (obs, act) pkl
│   ├── visualize.py              # 可视化统一入口（replay / policy / trajectory）
│   ├── analyze_expert_data.py    # 数据分布分析
│   ├── launch_tensorboard.py     # 启动 TensorBoard
│   ├── README.md                  # 脚本用法说明
│   └── ...
├── data/                      # 数据目录（相对路径）
│   ├── exp_filtered/          # Waymo 场景数据
│   ├── training_data/        # 专家 pkl 输出（generate_expert_data）
│   └── trajectories/         # 其他轨迹 pkl（如 expert_dataset 输出）
├── models/                    # 模型保存目录（相对路径）
│   ├── bc/                    # BC 模型 (.pt)
│   └── magail/                # MAGAIL 模型 (*_actor.pth, *_critic.pth)
├── logs/                      # 训练日志 (TensorBoard)
│   ├── bc/
│   └── magail/
├── train_bc.py                # [根目录] BC 训练
├── train_magail.py            # [根目录] MAGAIL 训练
└── README.md
```

## 路径约定（相对项目根）

- **数据**：Waymo 场景 `data/exp_filtered`；专家 pkl `data/training_data`；其他轨迹 `data/trajectories`
- **模型**：BC `models/bc/`，MAGAIL `models/magail/`
- **日志**：TensorBoard 写入 `logs/bc/`、`logs/magail/`

所有默认路径均为相对项目根，便于在不同设备上复用。

## 数据处理流程

从 Waymo Motion 原始数据到本项目训练用专家 pkl，依次为：

**1) 下载 Waymo Motion（TFRecord）**  
安装 `gsutil` 并登录 Google 账号后，例如只下载 training_20s：

```bash
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s" ./waymo/
```

**2) ScenarioNet Convert（TFRecord → ScenarioNet 场景库）**  
需安装 ScenarioNet、MetaDrive 及 TensorFlow 2.11、protobuf 3.20；转换时不用 GPU。

```bash
python -m scenarionet.convert_waymo -d data/exp_converted --raw_data_path ./waymo/training_20s --num_workers 64
```

**3) ScenarioNet Filter（按需筛选场景）**  
从 convert 得到的场景库中筛掉含红绿灯、天桥等场景，输出到如 `data/exp_filtered`。具体命令以 ScenarioNet 文档为准（Operations → Filter）。

**4) 本项目：生成专家 pkl**  
使用筛选后的场景目录，生成训练用 pkl 到 `data/training_data`：

```bash
python scripts/generate_expert_data.py --data_dir data/exp_filtered --output_dir data/training_data --num_scenarios 100 --start_index 0
```

## 核心工作流

### 1. 数据准备
使用 `scripts/generate_expert_data.py` 将 Waymo 数据转换为训练用 `.pkl`，输出到 `data/training_data/`。

```bash
python scripts/generate_expert_data.py --data_dir data/exp_filtered --output_dir data/training_data --num_scenarios 100
```

### 2. 行为克隆 (BC)
- **训练**：`python train_bc.py`（模型保存到 `models/bc/`，日志到 `logs/bc/`）
- **可视化**：`python scripts/visualize.py policy --policy_type bc --model_path models/bc/policy_best.pt`

### 3. 多智能体对抗模仿学习 (MAGAIL)
- **训练**：`python train_magail.py`（模型保存到 `models/magail/`，日志到 `logs/magail/`）
- **可视化**：`python scripts/visualize.py policy --policy_type magail --model_path models/magail/model_50_actor.pth`

### 4. 可视化统一入口
可视化统一使用 `scripts/visualize.py`，子命令：`replay`（场景回放）、`policy`（BC/MAGAIL 策略）、`trajectory`（专家轨迹 2D 动画）。详见 [scripts/README.md](scripts/README.md)。

## 文件与模块职责

### 根目录脚本
- **train_bc.py**：BC 训练，从 `dataset.loader` 加载专家 pkl，模型与日志写入 `models/bc/`、`logs/bc/`
- **train_magail.py**：MAGAIL 训练，环境使用 `BCScenarioEnv`（45 维），从 `dataset.loader` 加载专家数据，模型与日志写入 `models/magail/`、`logs/magail/`

### Env 模块
- **Env/bc_env.py**：`BCScenarioEnv`，45 维观测（Ego 5 维 + 10 邻居×4 维），BC 与 MAGAIL 训练/评估共用
- **Env/scenario_env.py**：`MultiAgentScenarioEnv` 基类，Waymo 场景加载与步进
- **Env/expert_replay_env.py**：专家轨迹回放与逆动力学动作，供 `generate_expert_data.py` 与回放可视化
- **Env/inverse_dynamics.py**：轨迹 → 油门/转向动作

### Algorithm 模块
- **Algorithm/policy.py**：`StateIndependentPolicy`，BC 使用的 MLP 策略

### scripts 目录
工具脚本用途与用法见 [scripts/README.md](scripts/README.md)。
