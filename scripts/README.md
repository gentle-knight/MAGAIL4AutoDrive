# scripts 工具脚本说明

本目录包含数据生成、回放、可视化与分析等工具脚本。训练脚本（`train_bc.py`、`train_magail.py`）位于项目根目录。

## 路径约定（相对项目根）

- **数据**：`data/exp_filtered`（Waymo 场景）、`data/training_data`（专家 pkl 输出）
- **模型**：`models/bc/`（BC）、`models/magail/`（MAGAIL）
- **日志**：`logs/bc/`、`logs/magail/`（TensorBoard）

---

## 脚本列表与用法

### 数据生成

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| [generate_expert_data.py](generate_expert_data.py) | 从 Waymo 数据生成专家 (obs, act) 的 pkl | `python scripts/generate_expert_data.py --data_dir data/exp_filtered --output_dir data/training_data --num_scenarios 100` |

**常用参数**：`--data_dir`（默认 `data/exp_filtered`）、`--output_dir`（默认 `data/training_data`）、`--start_index`、`--num_scenarios`。

---

### 回放与可视化

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| [visualize_replay.py](visualize_replay.py) | 原始专家轨迹回放（ExpertReplayEnv） | `python scripts/visualize_replay.py --data_dir data/exp_filtered --num_scenarios 1 --horizon 200` |
| [visualize_trained_policy.py](visualize_trained_policy.py) | **BC/MAGAIL 共用**：加载训练好的策略在 45 维场景中可视化 | 见下方「训练策略可视化」小节 |

#### 训练策略可视化（visualize_trained_policy.py）

使用训练好的 **BC** 或 **MAGAIL** 模型在 45 维场景环境中运行，并实时渲染俯瞰图（top-down view）。统一入口：`scripts/visualize_trained_policy.py`。

**BC 模型**：
```bash
python scripts/visualize_trained_policy.py --policy_type bc --model_path models/bc/policy_best.pt --data_dir data/exp_filtered --num_scenarios 1
```

**MAGAIL 模型**：
```bash
python scripts/visualize_trained_policy.py --policy_type magail --model_path models/magail/model_50_actor.pth --data_dir data/exp_filtered --num_scenarios 1 --deterministic
```

**自动推断类型**（根据 `--model_path` 扩展名：`.pt` → BC，否则 → MAGAIL）：
```bash
python scripts/visualize_trained_policy.py --model_path models/bc/policy_best.pt
python scripts/visualize_trained_policy.py --model_path models/magail/model_50_actor.pth
```

**根目录 BC 薄包装**：`python visualize_bc.py --model_path models/bc/policy_best.pt`

**参数**：`--policy_type`（`auto`|`bc`|`magail`）、`--model_path`（默认 `models/bc/policy_best.pt`）、`--data_dir`、`--start_index`、`--num_scenarios`、`--horizon`、`--deterministic`（仅 MAGAIL）。环境统一为 45 维 `BCScenarioEnv`，渲染为 MetaDrive top_down。数据目录未指定时默认 `data/exp_filtered`（不存在则 `data/exp_converted`）。

---

### 数据分析与检查

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| [analyze_expert_data.py](analyze_expert_data.py) | 分析专家数据分布与统计 | 见脚本内 `__main__`（依赖 env 与数据目录配置） |
| [check_track_fields.py](check_track_fields.py) | 检查 Waymo 轨迹字段 | 见脚本内 `__main__` |
| [check_database_info.py](check_database_info.py) | 检查数据库/场景信息 | 见脚本内 `__main__`（含硬编码路径，可按需改为 `data/exp_filtered`） |
| [visualize_expert_trajectory.py](visualize_expert_trajectory.py) | 用 matplotlib 画专家轨迹动画 | 依赖 `env.expert_trajectories`，与当前 env 接口可能不一致，可选使用 |

---

### 其他

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| [launch_tensorboard.py](launch_tensorboard.py) | 启动 TensorBoard | `python scripts/launch_tensorboard.py --logdir logs`（或 `logs/bc` / `logs/magail`） |

---

## 与训练流程的对应关系

1. **数据准备**：`generate_expert_data.py` → 输出到 `data/training_data/*.pkl`
2. **BC 训练**：根目录 `train_bc.py` → 模型保存到 `models/bc/`，日志到 `logs/bc/`
3. **MAGAIL 训练**：根目录 `train_magail.py` → 模型保存到 `models/magail/`，日志到 `logs/magail/`
4. **可视化**：`visualize_trained_policy.py`（或根目录 `visualize_bc.py` 仅 BC）→ 从 `models/bc` 或 `models/magail` 加载模型，数据目录默认 `data/exp_filtered`
