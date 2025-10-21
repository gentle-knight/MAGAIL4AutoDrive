# 更新日志

## 2025-01-20 问题修复与优化

### ✅ 已解决的问题

#### 1. 车辆生成位置偏差问题
**问题描述：** 部分车辆生成于草坪、停车场等非车道区域

**解决方案：**
- 实现 `_is_position_on_lane()` 方法：检测位置是否在有效车道上
- 实现 `_filter_valid_spawn_positions()` 方法：自动过滤非车道区域车辆
- 支持容差参数（默认3米）处理边界情况
- 在 `reset()` 时自动执行过滤，并输出统计信息

**配置参数：**
```python
"filter_offroad_vehicles": True,  # 启用/禁用过滤
"lane_tolerance": 3.0,           # 容差范围（米）
"max_controlled_vehicles": 10,   # 最大车辆数限制
```

#### 2. 红绿灯信息采集问题
**问题描述：**
- 部分红绿灯状态为 None
- 车道分段时部分车辆无法获取红绿灯状态

**解决方案：**
- 实现 `_get_traffic_light_state()` 方法，采用双重检测策略
- 方法1（优先）：从导航模块获取当前车道，直接查询（高效）
- 方法2（兜底）：遍历所有车道匹配位置（处理特殊情况）
- 完善异常处理，None 状态返回 0（无红绿灯）
- 返回值：0=无/未知, 1=绿灯, 2=黄灯, 3=红灯

#### 3. 性能优化问题
**问题描述：** FPS只有15帧，CPU利用率不到20%

**解决方案：**
- 创建 `run_multiagent_env_fast.py`：激光雷达优化版（30-60 FPS）
- 创建 `run_multiagent_env_parallel.py`：多进程并行版（300-600 steps/s）
- 提供详细的性能优化文档

### 📝 修改的文件

1. **Env/scenario_env.py**
   - 新增 `_is_position_on_lane()` 方法
   - 新增 `_filter_valid_spawn_positions()` 方法
   - 新增 `_get_traffic_light_state()` 方法
   - 更新 `default_config()` 添加配置参数
   - 更新 `reset()` 调用过滤逻辑
   - 更新 `_get_all_obs()` 使用新的红绿灯检测方法

2. **Env/run_multiagent_env.py**
   - 添加车道过滤配置参数

3. **Env/run_multiagent_env_fast.py**
   - 添加车道过滤配置
   - 性能优化配置

4. **Env/run_multiagent_env_parallel.py**
   - 添加车道过滤配置
   - 多进程并行实现

5. **README.md**
   - 更新问题说明，添加解决方案
   - 添加配置示例和测试方法
   - 添加问题解决总结

6. **新增文件**
   - `Env/test_lane_filter.py`：功能测试脚本

### 🧪 测试方法

```bash
# 测试车道过滤和红绿灯检测功能
python Env/test_lane_filter.py

# 运行标准版本（带过滤和可视化）
python Env/run_multiagent_env.py

# 运行高性能版本（适合训练）
python Env/run_multiagent_env_fast.py

# 运行多进程并行版本（最高吞吐量）
python Env/run_multiagent_env_parallel.py
```

### 💡 使用建议

1. **调试阶段**：使用 `run_multiagent_env.py`，启用渲染和车道过滤
2. **训练阶段**：使用 `run_multiagent_env_fast.py`，关闭渲染，启用所有优化
3. **大规模训练**：使用 `run_multiagent_env_parallel.py`，充分利用多核CPU

### ⚙️ 配置说明

所有配置参数都可以在创建环境时通过 `config` 字典传递：

```python
env = MultiAgentScenarioEnv(
    config={
        # 基础配置
        "data_directory": "...",
        "is_multi_agent": True,
        "horizon": 300,
        
        # 车道过滤（新增）
        "filter_offroad_vehicles": True,  # 启用车道过滤
        "lane_tolerance": 3.0,           # 容差3米
        "max_controlled_vehicles": 10,   # 最多10辆车
        
        # 性能优化
        "use_render": False,
        "decision_repeat": 5,
        ...
    },
    agent2policy=your_policy
)
```

### 🔍 技术细节

**车道检测逻辑：**
1. 使用 `lane.lane.point_on_lane()` 精确检测
2. 使用 `lane.local_coordinates()` 计算横向距离
3. 支持容差参数处理边界情况

**红绿灯检测逻辑：**
1. 优先从 `vehicle.navigation.current_lane` 获取
2. 失败时遍历所有车道查找
3. 所有异常均有保护，确保稳定性

**性能优化原理：**
- 减少激光束数量降低计算量
- 多进程绕过Python GIL限制
- 充分利用多核CPU

