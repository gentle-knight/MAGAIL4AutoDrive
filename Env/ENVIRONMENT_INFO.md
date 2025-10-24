# 多智能体场景环境详细说明

## 1. 观测信息详解

### 观测向量结构（总维度：107维）

每个智能体的观测向量包含以下信息：

```python
观测向量 = [
    # 1. 车辆状态信息 (5维)
    position_x,           # 车辆X坐标
    position_y,           # 车辆Y坐标
    velocity_x,           # X方向速度
    velocity_y,           # Y方向速度
    heading_theta,        # 朝向角度
    
    # 2. 前向激光雷达 (80维)
    lidar_1,              # 第1个激光束的距离
    lidar_2,              # 第2个激光束的距离
    ...
    lidar_80,             # 第80个激光束的距离
    # 范围：30米，用于前方障碍物检测
    
    # 3. 侧向激光雷达 (10维)
    side_lidar_1,         # 第1个侧向激光束的距离
    ...
    side_lidar_10,        # 第10个侧向激光束的距离
    # 范围：8米，用于侧方障碍物检测
    
    # 4. 车道线检测 (10维)
    lane_line_1,          # 第1个车道线检测距离
    ...
    lane_line_10,         # 第10个车道线检测距离
    # 范围：3米，用于车道线识别
    
    # 5. 导航信息 (2维)
    destination_x,        # 目标点X坐标
    destination_y,        # 目标点Y坐标
]
```

### 观测信息说明

1. **车辆状态 (5维)**
   - 位置：全局坐标系下的(x, y)
   - 速度：车辆在全局坐标系下的速度分量
   - 朝向：车辆的航向角（弧度）

2. **激光雷达 (100维)**
   - 前向80束：覆盖前方视野，检测动态和静态障碍物
   - 侧向10束：检测侧方物体，用于变道等操作
   - 车道线10束：专门检测车道线位置

3. **导航信息 (2维)**
   - 目标位置：从专家数据中提取的车辆最终位置

**总维度：5 + 80 + 10 + 10 + 2 = 107维**

---

## 2. 多场景加载逻辑

### 2.1 场景加载机制

MetaDrive的ScenarioEnv通过以下配置管理多场景：

```python
config = {
    "data_directory": "path/to/dataset",  # 包含dataset_mapping.pkl的目录
    "num_scenarios": 3,                   # 场景总数（从mapping文件读取）
    "start_scenario_index": 0,            # 起始场景索引
    "sequential_seed": True,              # 是否顺序切换场景
}
```

### 2.2 场景切换逻辑

#### 场景索引管理
- **初始化**：环境启动时从`start_scenario_index`开始
- **顺序模式**（`sequential_seed=True`）：
  - 每次`env.reset()`后自动切换到下一个场景
  - 循环顺序：场景0 → 场景1 → 场景2 → 场景0 ...
  
#### 示例流程
```python
env = MultiAgentScenarioEnv(config={
    "data_directory": "path/to/dataset",  # 假设有3个场景
    "start_scenario_index": 0,
    "sequential_seed": True,
})

obs = env.reset(0)   # 使用场景0
# ... 运行场景0 ...

obs = env.reset()    # 自动切换到场景1
# ... 运行场景1 ...

obs = env.reset()    # 自动切换到场景2
# ... 运行场景2 ...

obs = env.reset()    # 循环回到场景0
```

### 2.3 当前场景信息获取

可以通过以下方式查看当前场景信息：

```python
# 获取当前场景索引
current_scenario = env.engine.current_seed

# 获取场景总数
total_scenarios = env.config["num_scenarios"]

# 查看场景ID
scenario_id = env.engine.traffic_manager.current_scenario_id
```

### 2.4 手动指定场景

如果需要固定使用某个场景：

```python
# 方法1：在reset时指定
obs = env.reset(seed=1)  # 使用场景1

# 方法2：配置固定场景
config = {
    "start_scenario_index": 2,    # 从场景2开始
    "sequential_seed": False,     # 禁用自动切换
}
```

---

## 3. 车辆观测获取机制

### 3.1 观测获取流程

```
step() 被调用
    ↓
更新所有车辆物理状态
    ↓
_spawn_controlled_agents()  # 生成新车辆（按时间步）
    ↓
_get_all_obs()  # 获取所有车辆观测
    ↓
    遍历 controlled_agents:
        ├─ 获取车辆状态 (position, velocity, heading)
        ├─ 调用激光雷达传感器
        ├─ 组装观测向量
        └─ 添加到 obs_list
    ↓
返回 obs_list（包含所有车辆的观测）
```

### 3.2 观测返回格式

```python
obs = env.reset()
# obs 是一个列表，每个元素是一个车辆的观测向量

obs = [
    [obs_vehicle_0],  # 第1辆车的107维观测
    [obs_vehicle_1],  # 第2辆车的107维观测
    ...
]

# 在step中
obs, rewards, dones, infos = env.step(actions)
# obs格式相同，但只包含当前存活的车辆
```

### 3.3 观测一致性保证

1. **物理状态同步**
   - 所有车辆在同一物理时间步获取观测
   - 保证观测的时间一致性

2. **传感器独立性**
   - 每个车辆有独立的传感器
   - 激光雷达从各自位置发射

3. **动态车辆管理**
   - 新车辆在生成时立即获取观测
   - 观测列表动态更新

---

## 4. 常见问题解答

### Q1: 为什么观测维度是107而不是其他？
**A**: 
- 车辆状态: 5维 (x, y, vx, vy, heading)
- 前向激光雷达: 80维
- 侧向激光雷达: 10维
- 车道线检测: 10维
- 目标位置: 2维
- **总计: 5 + 80 + 10 + 10 + 2 = 107维**

### Q2: 如何确保多场景都被使用？
**A**: 设置`sequential_seed=True`，环境会自动循环遍历所有场景。

### Q3: 车辆在不同时间步生成，如何获取观测？
**A**: 每次调用`step()`时：
1. 先检查是否有新车辆需要生成（`_spawn_controlled_agents`）
2. 为所有现存车辆（包括新生成的）获取观测
3. 返回的obs_list包含所有当前存活车辆的观测

### Q4: 如果场景中车辆数量不同怎么办？
**A**: 
- 观测列表长度动态调整
- 使用`max_controlled_vehicles`可限制最大车辆数
- 使用`filter_offroad_vehicles`可过滤无效车辆

### Q5: 观测数据的坐标系是什么？
**A**:
- 位置/速度/目标：**全局坐标系**（世界坐标）
- 激光雷达：**车辆局部坐标系**（以车辆为中心）

---

## 5. 配置建议

### 5.1 训练配置
```python
config = {
    "data_directory": "path/to/dataset",
    "sequential_seed": True,           # 循环使用所有场景
    "filter_offroad_vehicles": True,   # 过滤无效车辆
    "max_controlled_vehicles": 20,     # 限制车辆数防止过载
    "inherit_expert_velocity": False,  # 训练时不继承速度
    "horizon": 300,                    # 每场景运行300步
}
```

### 5.2 评估配置
```python
config = {
    "data_directory": "path/to/dataset",
    "sequential_seed": False,          # 固定场景
    "start_scenario_index": 0,         # 指定场景
    "filter_offroad_vehicles": True,
    "max_controlled_vehicles": None,   # 不限制车辆数
    "inherit_expert_velocity": True,   # 评估时继承速度
    "verbose_reset": True,             # 输出详细信息
}
```

---

## 6. 调试技巧

### 6.1 查看当前场景信息
```python
# 在reset后查看
print(f"当前场景: {env.engine.current_seed}")
print(f"总场景数: {env.config['num_scenarios']}")
print(f"可控车辆数: {len(env.controlled_agents)}")
```

### 6.2 查看观测维度
```python
obs = env.reset()
print(f"车辆数量: {len(obs)}")
if len(obs) > 0:
    print(f"单车辆观测维度: {len(obs[0])}")
```

### 6.3 启用详细日志
```python
config = {
    "verbose_reset": True,       # 重置时详细统计
    "debug_lane_filter": True,   # 车道过滤调试
}
```

