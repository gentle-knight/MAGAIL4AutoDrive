# MAGAIL4AutoDrive

> 基于多智能体生成对抗模仿学习(MAGAIL)的自动驾驶训练系统 | MetaDrive + Waymo Open Motion Dataset

[![MetaDrive](https://img.shields [![Python](https://img.shields.io/io/badge/Dataset目实现了适配多智能体场景的GAIL算法训练系统,核心创新在于**改进判别器架构支持动态车辆数量**,利用Transformer处理1-100+辆车的交互场景。

**核心特性:**
- ✅ 完整的Waymo数据处理pipeline(12,201个场景)
- ✅ 车道过滤和红绿灯检测优化
- ✅ 支持5维简化/107维完整观测空间
- ✅ 专家轨迹数据集(52K+训练样本)
- 🚧 MAGAIL算法实现(判别器+策略网络)

***

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone <repository_url>
cd MAGAIL4AutoDrive

# 安装依赖
pip install metadrive-simulator==0.4.3 torch numpy matplotlib scenarionet

# 创建必需目录
mkdir -p analysis_results
touch scripts/__init__.py dataset/__init__.py Algorithm/__init__.py
```

### 数据准备

```bash
# 1. 转换Waymo数据
python -m scenarionet.convert_waymo -d ~/mdsn/exp_converted --raw_data_path /path/to/waymo --num_files=150

# 2. 筛选场景(无红绿灯)
python -m scenarionet.filter --database_path ~/mdsn/exp_filtered --from ~/mdsn/exp_converted --no_traffic_light

# 3. 验证数据集
python scripts/check_database_info.py
```

### 运行环境

```bash
# 测试多智能体环境
python Env/run_multiagent_env.py

# 收集专家数据(10个场景测试)
python dataset/expert_dataset.py
```

***

## 📁 项目结构

```
MAGAIL4AutoDrive/
├── Env/                      # 仿真环境模块
│   ├── scenario_env.py       # 多智能体场景环境(含轨迹存储)
│   ├── run_multiagent_env.py# 环境运行脚本
│   └── simple_idm_policy.py  # 测试策略
│
├── dataset/                  # 数据集模块
│   └── expert_dataset.py     # PyTorch Dataset(5维观测)
│
├── scripts/                  # 工具脚本
│   ├── check_track_fields.py      # 数据字段验证
│   ├── check_database_info.py     # 数据库信息检查
│   ├── analyze_expert_data.py     # 统计分析
│   └── visualize_expert_trajectory.py # 轨迹可视化
│
├── Algorithm/                # MAGAIL算法(待完善)
│   ├── bert.py              # Transformer判别器
│   ├── disc.py              # 判别器网络
│   ├── policy.py            # 策略网络
│   ├── ppo.py               # PPO优化器
│   └── magail.py            # MAGAIL训练循环
│
└── analysis_results/         # 分析输出
    ├── statistics.pkl        # 数据统计
    └── distributions.png     # 可视化图表
```

***

## 🎯 核心功能

### 1. 环境与数据处理

**scenario_env.py** - 多智能体场景环境
- 专家轨迹完整存储(位置、速度、航向角、车辆尺寸)
- 车道区域过滤(自动移除非车道车辆)
- 红绿灯状态检测(双重保障机制)
- 107维完整观测空间(激光雷达+车道线)

**expert_dataset.py** - 专家数据集
- 状态-动作对提取(逆动力学)
- 批量采样和序列化
- 支持PyTorch DataLoader

### 2. 数据分析工具

| 脚本 | 功能 | 输出 |
|------|------|------|
| `check_database_info.py` | 验证数据库完整性 | 场景总数、映射关系 |
| `check_track_fields.py` | 检查可用字段 | 必需/可选字段列表 |
| `analyze_expert_data.py` | 统计分析 | 轨迹长度、速度、交互频率 |
| `visualize_expert_trajectory.py` | 轨迹可视化 | 动画展示车辆运动 |

### 3. MAGAIL算法

**判别器** (Algorithm/bert.py + disc.py)
- Transformer编码器处理动态车辆数量
- CLS标记或均值池化聚合特征
- 支持集中式/去中心化/零和模式

**策略网络** (Algorithm/policy.py + ppo.py)
- Actor-Critic架构
- 参数共享机制(所有车辆共享模型)
- PPO/TRPO优化器

***

## ⚙️ 配置说明

```python
# 环境配置
config = {
    # 数据路径
    "data_directory": "~/mdsn/exp_filtered",
    
    # 多智能体设置
    "num_controlled_agents": 3,     # 初始车辆数
    "max_controlled_vehicles": 10,  # 最大车辆数限制
    
    # 车道过滤
    "filter_offroad_vehicles": True, # 启用车道过滤
    "lane_tolerance": 3.0,           # 容差(米)
    
    # 场景加载
    "sequential_seed": True,         # 顺序加载场景
    "horizon": 1000,                 # 最大步数
}
```

***

## 📊 数据集统计

**当前数据规模**(基于exp_filtered):
- 场景总数: **12,201**
- 已收集场景: 10个测试场景
- 轨迹数: 900条
- 训练样本: **52,065**个(s,a)对
- 观测维度: 5维(简化) / 107维(完整)
- 动作维度: 2维(油门/刹车, 转向)

**数据质量**:
- 静止车辆占比: 54.8%(正常,包含停车场和路边停车)
- 平均轨迹长度: 67帧(6.7秒 @ 10Hz)
- 平均速度: 1.46 m/s
- 近距离交互(<5m): 1.92%

***

## 🛠️ 使用示例

### 收集专家数据

```python
# dataset/expert_dataset.py
from expert_dataset import ExpertTrajectoryDataset

# 收集1000个场景
trajectories = ExpertTrajectoryDataset.collect_from_env(
    env_config,
    num_scenarios=1000,
    save_path="./expert_trajectories.pkl"
)

# 创建数据集
dataset = ExpertTrajectoryDataset(trajectories, sequence_length=1)
```

### 环境测试

```python
from scenario_env import MultiAgentScenarioEnv

env = MultiAgentScenarioEnv(
    config=config,
    agent2policy=your_policy
)

obs = env.reset()
for step in range(1000):
    actions = {aid: policy(obs[aid]) for aid in env.controlled_agents}
    obs, rewards, dones, infos = env.step(actions)
```

***

## ❓ 常见问题

### Q1: KeyError: 'bbox'
**原因**: Waymo转换数据不含bbox字段  
**解决**: 使用length/width/height,代码已添加条件检查

### Q2: ModuleNotFoundError: scenario_env
**原因**: Python路径问题  
**解决**: 脚本开头添加:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Env"))
```

### Q3: 多次reset失败(clear_objects错误)
**原因**: MetaDrive对象管理bug  
**解决**: 每次收集数据都重新创建环境(已实现)

### Q4: 静止车辆占比过高
**原因**: Waymo真实场景包含停车场等静止车辆  
**解决**: 可在数据收集时过滤平均速度<2m/s的轨迹

***

## 📈 开发路线图

### ✅ 已完成(Phase 1)
- [x] 数据转换与筛选
- [x] 完整轨迹存储
- [x] 数据质量分析
- [x] PyTorch Dataset构建

### 🚧 进行中(Phase 2)
- [ ] 107维完整观测空间
- [ ] 数据质量过滤
- [ ] 轨迹可视化工具

### 📅 计划中(Phase 3-4)
- [ ] 判别器网络实现
- [ ] Actor-Critic策略网络
- [ ] MAGAIL训练循环
- [ ] TensorBoard监控
- [ ] 实验与评估

***

## 📚 参考资料

- [MetaDrive Documentation](https://metadrive-simulator.readthedocs.io/)
- [Waymo Open Dataset](https://waymo.com/open/)
- [MAGAIL Paper](https://arxiv.org/abs/1807.09936)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)

## 📄 License

MIT License

***

**💡 提示**: 项目处于活跃开发中,欢迎提Issue或PR贡献代码!

[1](https://blog.csdn.net/BxuqBlockchain/article/details/133606934)
[2](https://blog.csdn.net/sinat_28461591/article/details/148351123)
[3](https://www.reddit.com/r/Python/comments/13kpoti/readmeai_autogenerate_readmemd_files/)
[4](https://www.reddit.com/r/learnprogramming/comments/1298ix8/what_does_a_good_readme_look_like_for_personal/)
[5](https://juejin.cn/post/7195763127883169853)
[6](https://jimmysong.io/trans/spec-driven-development-using-markdown/)
[7](https://www.showapi.com/news/article/66b602964ddd79f11a001e3c)
[8](https://learn.microsoft.com/zh-cn/nuget/nuget-org/package-readme-on-nuget-org)