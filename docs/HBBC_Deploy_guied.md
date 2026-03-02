# HBBC 策略部署指南

本文档说明如何将 `weights/hbbc.pt` 部署到 MetaDrive 项目中的**背景车辆**上，作为车辆控制策略使用。

---

## 0. 本仓库适配说明（MAGAIL4AutoDrive）

本仓库已落地一套可直接使用的 HBBC 背景车接入实现，核心代码：

- `Env/hbbc_actor_critic.py`：HBBC 所需 `ActorCritic` 最小推理网络
- `Env/hbbc_background_policy.py`：模型加载、18 维观测构建、latent 管理（含 JSON 覆盖）
- `Env/bc_env.py`：`BCScenarioEnv` 动态背景车 HBBC 接入（静态背景车保持不变）
- `Env/bc_ego_replay_env.py`：`BCEgoReplayEnv` 动态背景车 HBBC 接入（ego-only 评估兼容）

与原文档示例不同点：

1. 当前仓库 `BaseVehicle` 没有 `pos_buffer/rot_buffer/action_buffer`，因此 8 维 `base_state` 使用当前可得车辆状态重建；
2. 仅动态背景车使用 HBBC，静态背景车仍作为占位/邻居车辆；
3. 支持通过 JSON 手动指定场景中某些车辆的 latent（`object_id` / `agent_id` 双 key）。

---

## 1. 概述

### 1.1 HBBC 是什么

**HBBC**（Hierarchical Behavior-Based Controller）是一个低层驾驶策略网络，输入车辆状态和行为条件，输出连续控制动作 `[steering, acceleration]`，可直接用于 MetaDrive 的车辆控制。

### 1.2 依赖

- **PyTorch**
- **NumPy**
- **MetaDrive**（需包含 `BaseVehicle`、`BasePolicy` 等基础组件）

---

## 2. 模型加载

### 2.1 模型架构

HBBC 对应 `ActorCritic` 网络，需按以下参数实例化：

```python
import torch
from algorithms.modules import ActorCritic  # 或复制 actor_critic.py 到目标项目

hbbc = ActorCritic(
    num_actor_obs=18,
    num_critic_obs=18,
    num_actions=2,
    latent_c_dim=4,      # 行为模式数
    latent_eps_dim=6,    # 风格向量维度
    use_style_latent=True,
).to(device)

# 加载权重
checkpoint = torch.load("path/to/hbbc.pt", map_location=device, weights_only=False)
hbbc.load_state_dict(checkpoint['actor_critic'])
hbbc.eval()
```

### 2.2 推理接口

```python
with torch.no_grad():
    actions = hbbc.act_inference(obs_tensor)  # obs_tensor: (batch, 18), 输出: (batch, 2)
```

---

## 3. 输入规格（18 维）

HBBC 的输入为 `hbbc_obs`，维度 18，由三部分拼接：

```
hbbc_obs = [base_state(8) | latent_eps(6) | latent_c(4)]
```

### 3.1 base_state（8 维）

从车辆对象构建，需按**精确顺序**拼接。实现如下（需配合 `relative_pos_local`、`rot_matrix_inv`、`clip` 等工具函数）：

```python
import numpy as np

def build_hbbc_base_state(vehicle):
    """
    从 MetaDrive 车辆对象构建 HBBC 的 8 维 base_state。
    要求 vehicle 具有: position, pos_buffer, rot_buffer, heading_buffer,
    speed_km_h, max_speed_km_h, eps_step, acceleration, yaw_rate, action_buffer
    """
    from metadrive.utils.math import clip  # 或 np.clip
    
    veh_pos = list(vehicle.position) + [0]
    init_veh_rot = np.array([vehicle.rot_buffer[0][0], vehicle.rot_buffer[0][1], vehicle.rot_buffer[0][2]])
    init_veh_pos = list(vehicle.pos_buffer[0]) + [0]
    init_veh_heading = vehicle.heading_buffer[0]

    # 局部位置（本实现中置 0）
    veh_pos_local = relative_pos_local(init_veh_pos, veh_pos, init_veh_rot)[:2]
    veh_pos_local[0] /= 10
    veh_pos_local[1] /= 2

    # 局部航向（本实现中置 0）
    veh_heading = vehicle.heading
    cross = np.cross(init_veh_heading, veh_heading)
    dot = np.dot(init_veh_heading, veh_heading)
    veh_heading_local = np.arctan2(cross, dot)

    veh_vel = clip((vehicle.speed_km_h + 1) / (vehicle.max_speed_km_h + 1), 0.0, 1.0)
    veh_acc = vehicle.acceleration / 5 if vehicle.eps_step > 1 else 0
    yaw_rate = vehicle.yaw_rate
    last_action_0 = vehicle.action_buffer[-1][0]
    last_action_1 = vehicle.action_buffer[-1][1]

    # 8 维，顺序固定
    obs = np.concatenate((
        veh_pos_local * 0,           # 2 维，置 0
        [veh_heading_local * 0],     # 1 维，置 0
        [veh_vel],                   # 1 维
        [veh_acc * 0],               # 1 维，置 0
        [yaw_rate * 0.5],            # 1 维
        [last_action_0], [last_action_1]  # 2 维
    )).astype(np.float32)
    return obs
```

### 3.2 latent_eps（6 维）

风格向量，需 **L2 归一化** 且在 `[-1, 1]` 内：

```python
# 随机采样（每个 episode 或每辆车可固定/随机）
latent_eps = np.random.randn(6).astype(np.float32)
latent_eps = latent_eps / (np.linalg.norm(latent_eps) + 1e-8)
latent_eps = np.clip(latent_eps, -1.0, 1.0)
```

### 3.3 latent_c（4 维）

行为模式 one-hot，4 选 1：

```python
# 随机选一个模式 (0~3)
mode = np.random.randint(0, 4)
latent_c = np.zeros(4, dtype=np.float32)
latent_c[mode] = 1.0
```

### 3.4 完整观测拼接

```python
def build_hbbc_obs(vehicle, latent_eps, latent_c):
    base = build_hbbc_base_state(vehicle)
    return np.concatenate([base, latent_eps, latent_c], axis=-1)  # shape: (18,)
```

---

## 4. 必需工具函数

若目标项目无以下函数，需自行实现或从 styledrive 的 `envs/utils.py` 拷贝：

```python
def rot_matrix(t):
    """t: [roll, pitch, yaw], 返回 3x3 旋转矩阵"""
    roll, pitch, yaw = t[0], t[1], t[2]
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    r_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    r_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    r_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return np.dot(np.dot(r_yaw, r_pitch), r_roll)

def rot_matrix_inv(t):
    return rot_matrix(t).T

def relative_pos_local(coord, coord_t, veh_rot):
    """将 coord_t 从世界坐标变换到以 coord 为原点、veh_rot 为姿态的局部坐标"""
    r_pos_global = np.array(coord_t) - np.array(coord)
    rot_mat_inv = rot_matrix_inv(veh_rot)
    return rot_mat_inv @ r_pos_global
```

`clip` 可用 `np.clip` 或 `metadrive.utils.math.clip`。

---

## 5. 车辆属性要求

使用 HBBC 的车辆需继承或兼容 MetaDrive 的 `BaseVehicle`，并具备：

| 属性 | 说明 |
|------|------|
| `position` | 当前位置 (x, y) 或 (x, y, z) |
| `heading` | 航向单位向量 |
| `heading_theta` | 航向角（弧度） |
| `pos_buffer` | `deque`，至少 1 个元素，`pos_buffer[0]` 为 episode 起始位姿 |
| `rot_buffer` | `deque`，`(roll, pitch, yaw)`，`rot_buffer[0]` 为起始姿态 |
| `heading_buffer` | `deque`，`heading_buffer[0]` 为起始航向 |
| `action_buffer` | `deque`，`action_buffer[-1]` 为上一时刻动作 `(steering, acc)` |
| `speed_km_h` | 当前速度 km/h |
| `max_speed_km_h` | 最大速度 km/h |
| `acceleration` | 当前加速度 |
| `yaw_rate` | 偏航角速度 (rad/s) |
| `eps_step` | 本 episode 的步数 |
| `last_heading_theta` | 上一帧航向角（用于 yaw_rate） |

`BaseVehicle` 在 `before_step` 中会更新 `pos_buffer`、`rot_buffer`、`heading_buffer`、`action_buffer`，只要在配置中设置 `veh_obs_len >= 1`（建议 3–10）即可。

---

## 6. 输出动作格式

HBBC 输出 2 维连续动作，与 MetaDrive 动作空间一致：

```python
# actions: (2,) 或 (batch, 2)
# actions[0]: steering  ∈ [-1, 1]
# actions[1]: acceleration ∈ [-1, 1]，正=油门，负=刹车
```

环境会在 `_preprocess_actions` 中做限幅与平滑，无需在策略内再次裁剪。

---

## 7. 部署为 MetaDrive 策略（背景车）

### 7.1 自定义 Policy

实现一个继承 `BasePolicy` 的策略，在 `act` 中调用 HBBC：

```python
from metadrive.policy.base_policy import BasePolicy
import torch
import numpy as np

class HBBCPolicy(BasePolicy):
    def __init__(self, control_object, random_seed=None, hbbc_path="weights/hbbc.pt", device="cpu"):
        super().__init__(control_object, random_seed)
        self.device = torch.device(device)
        self.hbbc = self._load_hbbc(hbbc_path)
        self.latent_eps = None
        self.latent_c = None
        self._resample_latent()

    def _load_hbbc(self, path):
        from algorithms.modules import ActorCritic  # 根据实际路径调整
        model = ActorCritic(
            num_actor_obs=18, num_critic_obs=18, num_actions=2,
            latent_c_dim=4, latent_eps_dim=6, use_style_latent=True
        ).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt['actor_critic'])
        model.eval()
        return model

    def _resample_latent(self):
        self.latent_eps = np.random.randn(6).astype(np.float32)
        self.latent_eps = self.latent_eps / (np.linalg.norm(self.latent_eps) + 1e-8)
        self.latent_eps = np.clip(self.latent_eps, -1.0, 1.0)
        mode = np.random.randint(0, 4)
        self.latent_c = np.zeros(4, dtype=np.float32)
        self.latent_c[mode] = 1.0

    def act(self, agent_id=None):
        vehicle = self.control_object
        base_state = build_hbbc_base_state(vehicle)
        obs = np.concatenate([base_state, self.latent_eps, self.latent_c], axis=-1)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            actions = self.hbbc.act_inference(obs_t).cpu().numpy().squeeze()
        self.action_info["action"] = actions.tolist()
        return [float(actions[0]), float(actions[1])]

    def reset(self):
        super().reset()
        self._resample_latent()
```

### 7.2 配置背景车使用 HBBC

在环境配置中为背景车辆指定 `HBBCPolicy`：

```python
config = {
    # ...
    "agent_configs": {
        "agent0": {
            "policy": HBBCPolicy,
            "policy_kwargs": {"hbbc_path": "path/to/hbbc.pt", "device": "cuda:0"},
        }
    },
    # 若使用 traffic 的 policy 配置方式，则需在 traffic 管理逻辑中
    # 将部分或全部背景车的 policy 替换为 HBBCPolicy
}
```

若背景车由 TrafficManager 等模块统一管理，需在该模块的 policy 选择逻辑中加入对 `HBBCPolicy` 的分配。

### 7.3 与 TrafficManager 集成

若背景车由 `PGTrafficManager` 等生成，需在添加策略时改为使用 `HBBCPolicy`：

```python
# 原代码通常为:
# self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())

# 改为:
from your_policy_module import HBBCPolicy
self.add_policy(random_v.id, HBBCPolicy, random_v, self.generate_seed(),
                hbbc_path="path/to/hbbc.pt", device="cuda:0")
```

`add_policy` 的额外参数会传给 Policy 的 `__init__`。若接口不支持传参，可修改 `HBBCPolicy` 从全局配置读取路径，或使用自定义 TrafficManager 子类。

**注意**：HBBC 在 styledrive 中基于 scenario 轨迹训练，不包含路由逻辑。背景车若需要沿车道/路线行驶，可能需：
- 在项目中为 HBBC 车辆配置 `navigation`，或
- 仅对部分背景车使用 HBBC（如混合 IDM + HBBC），或
- 在目标项目中验证 HBBC 在开放道路上的表现后决定是否全量使用。

### 7.4 注意事项

1. **latent 生命周期**：可为每辆车在 spawn 时采样一次，或在每个 episode reset 时重采样。
2. **首帧 action_buffer**：首步 `action_buffer[-1]` 通常为 `(0, 0)`，由 `BaseVehicle` 初始化保证。
3. **同步更新 buffer**：车辆必须在每步调用 `before_step` 之类接口，更新 `pos_buffer`、`action_buffer` 等，否则观测会错位。
4. **veh_obs_len**：车辆配置中设置 `veh_obs_len >= 3`（建议 10），确保 buffer 长度足够。

---

## 8. ActorCritic 网络定义（可移植）

若目标项目无法导入 styledrive 的 `algorithms`，可把以下简化版 `ActorCritic` 放到本项目中单独使用：

```python
import torch
import torch.nn as nn

def get_activation(name):
    return getattr(nn, name)()

class ActorCritic(nn.Module):
    def __init__(self, num_actor_obs=18, num_critic_obs=18, num_actions=2,
                 latent_c_dim=4, latent_eps_dim=6, use_style_latent=True,
                 actor_hidden_dims=[512, 256, 128], activation='elu'):
        super().__init__()
        act_fn = getattr(nn, activation.upper())()
        self.latent_c_dim = latent_c_dim
        self.latent_eps_dim = latent_eps_dim
        self.use_style_latent = use_style_latent

        layers = []
        layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        layers.append(act_fn)
        for i in range(len(actor_hidden_dims) - 1):
            layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            layers.append(act_fn)
        self.actor_trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(actor_hidden_dims[-1], num_actions)

        if use_style_latent:
            style_layers = [nn.Linear(latent_eps_dim, 512), act_fn,
                           nn.Linear(512, 256), act_fn, nn.Linear(256, 128), act_fn]
            self.style_trunk = nn.Sequential(*style_layers)
            self.style_head = nn.Linear(128, latent_eps_dim)
            self.style_activation = torch.tanh

    def act_inference(self, observations):
        if self.use_style_latent:
            obs = observations[..., :-(self.latent_c_dim + self.latent_eps_dim)]
            eps = observations[..., -self.latent_c_dim - self.latent_eps_dim:-self.latent_c_dim]
            c = observations[..., -self.latent_c_dim:]
            eps = self.style_activation(self.style_head(self.style_trunk(eps)))
            observations = torch.cat([obs, eps, c], dim=-1)
        embedding = self.actor_trunk(observations)
        return self.actor_head(embedding)
```

加载与调用方式与前面一致。

---

## 9. 简要检查清单

- [ ] 正确加载 `hbbc.pt` 的 `actor_critic` 权重
- [ ] `build_hbbc_base_state` 输出 8 维，顺序与文档一致
- [ ] `latent_eps` 6 维、L2 归一化
- [ ] `latent_c` 4 维 one-hot
- [ ] 车辆具备 `pos_buffer`、`rot_buffer`、`heading_buffer`、`action_buffer` 等属性
- [ ] 策略返回 `[steering, acceleration]`，范围 [-1, 1]
- [ ] 每步更新上述 buffer，保证观测连续

---

## 10. 本仓库配置项与 JSON 示例

可通过环境配置控制 HBBC 背景车行为：

- `enable_hbbc_background`：是否启用动态背景车 HBBC（`True/False`）
- `hbbc_model_path`：模型路径（默认 `models/hbbc/hbbc.pt`）
- `hbbc_inference_device`：推理设备（如 `cpu` / `cuda:0`）
- `hbbc_latent_mode`：`per_vehicle_fixed` 或 `per_episode_reset`
- `hbbc_latent_json_path`：可选，手动 latent JSON 路径

`hbbc_latent_json_path` 内容格式（优先按 `object_id` 匹配，失败回退 `agent_id`）：

```json
{
  "global": {
    "latent_eps": [0.35, -0.12, 0.28, 0.46, -0.22, 0.18],
    "latent_c": [0, 0, 1, 0]
  },
  "object_id": {
    "12345": {
      "latent_eps": [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
      "latent_c": [0, 1, 0, 0]
    }
  },
  "agent_id": {
    "controlled_abcde": {
      "latent_eps": [0.5, 0.1, -0.1, 0.2, -0.3, 0.4],
      "latent_c": [1, 0, 0, 0]
    }
  }
}
```

匹配优先级为：`object_id` > `agent_id` > `global` > 随机采样。  
`latent_eps` 会做 L2 归一化，`latent_c` 会强制 one-hot；非法输入会告警并回退随机采样。

---

## 11. 参考来源

- 策略与观测：`envs/ad_hbbc_gym.py` 中的 `ADObservation.vehicle_state`
- 模型：`algorithms/modules/actor_critic.py` 中 `ActorCritic`
- 工具：`envs/utils.py` 中的 `relative_pos_local`、`rot_matrix`、`rot_matrix_inv`
