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

### 可视化（统一入口）

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| [visualize.py](visualize.py) | **replay**：场景回放（ExpertReplayEnv）；**policy**：BC/MAGAIL 策略；**trajectory**：专家轨迹 2D 动画 | 见下方 |

**子命令**：

- **replay**（原始专家轨迹回放）：
```bash
python scripts/visualize.py replay --data_dir data/exp_filtered --num_scenarios 1 --horizon 500
```

- **policy**（BC 或 MAGAIL 训练策略）：与专家数据生成/回放一致——同一套车道+静态筛选、且会生成背景车（bg_*），使观测分布与训练集一致，便于在训练集上公平演示。
```bash
python scripts/visualize.py policy --policy_type bc --model_path models/bc/policy_best.pt --data_dir data/exp_filtered --num_scenarios 1
python scripts/visualize.py policy --policy_type magail --model_path models/magail/model_50_actor.pth --num_scenarios 1 --deterministic
```

- **trajectory**（专家轨迹 matplotlib 俯视图动画）：
```bash
python scripts/visualize.py trajectory --data_dir data/exp_filtered --scenario_idx 0
```

**公共参数**：`--data_dir`（默认 `data/exp_filtered`）、`--start_index`、`--num_scenarios`、`--horizon`。policy 模式另有 `--policy_type`（auto/bc/magail）、`--model_path`、`--deterministic`（仅 MAGAIL）。

---

### 数据分析与检查

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| [analyze_expert_data.py](analyze_expert_data.py) | 分析专家数据分布与统计 | 见脚本内 `__main__`（依赖 env 与数据目录配置） |
| [check_track_fields.py](check_track_fields.py) | 检查 Waymo 轨迹字段 | 见脚本内 `__main__` |
| [check_database_info.py](check_database_info.py) | 检查数据库/场景信息 | 见脚本内 `__main__`（含硬编码路径，可按需改为 `data/exp_filtered`） |

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
4. **可视化**：`scripts/visualize.py`（子命令 replay / policy / trajectory）→ 数据目录默认 `data/exp_filtered`
