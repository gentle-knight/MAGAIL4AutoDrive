# 调试功能使用指南

## 📋 概述

已为车道过滤和红绿灯检测功能添加了详细的调试输出，帮助您诊断和理解代码行为。

---

## 🎛️ 调试开关

### 1. 配置参数

在创建环境时，可以通过 `config` 参数启用调试模式：

```python
env = MultiAgentScenarioEnv(
    config={
        # ... 其他配置 ...
        
        # 🔥 调试开关
        "debug_lane_filter": True,     # 启用车道过滤调试
        "debug_traffic_light": True,   # 启用红绿灯检测调试
    },
    agent2policy=your_policy
)
```

### 2. 默认值

两个调试开关默认都是 `False`（关闭），避免正常运行时产生大量日志。

---

## 📊 车道过滤调试 (`debug_lane_filter=True`)

### 输出内容

```
📍 场景信息统计:
  - 总车道数: 123

🔍 开始车道过滤: 共 51 辆车待检测

车辆 1/51: ID=128
  🔍 检测位置 (-4.11, 46.76), 容差=3.0m
    ✅ 在车道上 (车道184, 检查了32条)
  ✅ 保留

车辆 7/51: ID=134
  🔍 检测位置 (-51.34, -3.77), 容差=3.0m
    ❌ 不在任何车道上 (检查了123条车道)
  ❌ 过滤 (原因: 不在车道上)

... (所有车辆)

📊 过滤结果: 保留 45 辆, 过滤 6 辆
```

### 调试信息说明

| 信息 | 含义 |
|------|------|
| 📍 场景信息统计 | 场景的基本信息（车道数、红绿灯数） |
| 🔍 开始车道过滤 | 开始过滤，显示待检测车辆总数 |
| 🔍 检测位置 | 车辆的坐标和使用的容差值 |
| ✅ 在车道上 | 找到了车辆所在的车道，显示车道ID和检查次数 |
| ❌ 不在任何车道上 | 所有车道都检查完了，未找到匹配的车道 |
| 📊 过滤结果 | 最终统计：保留多少辆，过滤多少辆 |

### 典型输出案例

**情况1：车辆在正常车道上**
```
车辆 1/51: ID=128
  🔍 检测位置 (-4.11, 46.76), 容差=3.0m
    ✅ 在车道上 (车道184, 检查了32条)
  ✅ 保留
```
→ 检查了32条车道后找到匹配的车道184

**情况2：车辆在草坪/停车场**
```
车辆 7/51: ID=134
  🔍 检测位置 (-51.34, -3.77), 容差=3.0m
    ❌ 不在任何车道上 (检查了123条车道)
  ❌ 过滤 (原因: 不在车道上)
```
→ 检查了所有123条车道都不匹配，该车辆被过滤

---

## 🚦 红绿灯检测调试 (`debug_traffic_light=True`)

### 输出内容

```
📍 场景信息统计:
  - 总车道数: 123
  - 有红绿灯的车道数: 0
    ⚠️ 场景中没有红绿灯！

🚦 检测车辆红绿灯 - 位置: (-4.1, 46.8)
  方法1-导航模块:
    current_lane = <metadrive.component.lane.straight_lane.StraightLane object>
    lane_index = 184
    has_traffic_light = False
    该车道没有红绿灯
  方法2-遍历车道: 开始遍历 123 条车道
    ✓ 找到车辆所在车道: 184 (检查了32条)
    has_traffic_light = False
    该车道没有红绿灯
  结果: 返回 0 (无红绿灯/未知)
```

### 调试信息说明

| 信息 | 含义 |
|------|------|
| 有红绿灯的车道数 | 统计场景中有多少个红绿灯 |
| ⚠️ 场景中没有红绿灯 | 如果数量为0，会特别提示 |
| 方法1-导航模块 | 尝试从导航系统获取 |
| current_lane | 导航系统返回的当前车道对象 |
| lane_index | 车道的唯一标识符 |
| has_traffic_light | 该车道是否有红绿灯 |
| status | 红绿灯的状态（GREEN/YELLOW/RED/None） |
| 方法2-遍历车道 | 兜底方案，遍历所有车道查找 |
| ✓ 找到车辆所在车道 | 遍历找到了匹配的车道 |

### 典型输出案例

**情况1：场景没有红绿灯**
```
📍 场景信息统计:
  - 有红绿灯的车道数: 0
    ⚠️ 场景中没有红绿灯！

🚦 检测车辆红绿灯 - 位置: (-4.1, 46.8)
  方法1-导航模块:
    ...
    has_traffic_light = False
    该车道没有红绿灯
  结果: 返回 0 (无红绿灯/未知)
```
→ 所有车辆都会返回0，这是正常的

**情况2：有红绿灯且状态正常**
```
🚦 检测车辆红绿灯 - 位置: (10.5, 20.3)
  方法1-导航模块:
    current_lane = <...>
    lane_index = 205
    has_traffic_light = True
    status = TRAFFIC_LIGHT_GREEN
  ✅ 方法1成功: 绿灯
```
→ 方法1直接成功，返回1（绿灯）

**情况3：红绿灯状态为None**
```
🚦 检测车辆红绿灯 - 位置: (10.5, 20.3)
  方法1-导航模块:
    current_lane = <...>
    lane_index = 205
    has_traffic_light = True
    status = None
  ⚠️ 方法1: 红绿灯状态为None
```
→ 有红绿灯，但状态异常，返回0

**情况4：导航失败，方法2兜底**
```
🚦 检测车辆红绿灯 - 位置: (15.2, 30.5)
  方法1-导航模块: 不可用 (hasattr=True, not_none=False)
  方法2-遍历车道: 开始遍历 123 条车道
    ✓ 找到车辆所在车道: 210 (检查了45条)
    has_traffic_light = True
    status = TRAFFIC_LIGHT_RED
  ✅ 方法2成功: 红灯
```
→ 方法1失败，方法2兜底成功，返回3（红灯）

---

## 🧪 测试方法

### 方式1：使用测试脚本

```bash
# 标准测试（无详细调试）
python Env/test_lane_filter.py

# 调试模式（详细输出）
python Env/test_lane_filter.py --debug
```

### 方式2：在代码中直接启用

```python
from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy

env = MultiAgentScenarioEnv(
    config={
        "data_directory": "...",
        "use_render": False,
        
        # 启用调试
        "debug_lane_filter": True,
        "debug_traffic_light": True,
    },
    agent2policy=ConstantVelocityPolicy(target_speed=50)
)

obs = env.reset(0)
# 调试信息会自动输出
```

---

## 📝 调试输出控制

### 场景1：只想看车道过滤

```python
config = {
    "debug_lane_filter": True,
    "debug_traffic_light": False,  # 关闭红绿灯调试
}
```

### 场景2：只想看红绿灯检测

```python
config = {
    "debug_lane_filter": False,
    "debug_traffic_light": True,  # 只看红绿灯
}
```

### 场景3：生产环境（关闭所有调试）

```python
config = {
    "debug_lane_filter": False,
    "debug_traffic_light": False,
}
# 或者直接不设置这两个参数，默认就是False
```

---

## 💡 常见问题诊断

### 问题1：所有红绿灯状态都是0

**检查调试输出：**
```
📍 场景信息统计:
  - 有红绿灯的车道数: 0
    ⚠️ 场景中没有红绿灯！
```

**结论：** 场景本身没有红绿灯，返回0是正常的

---

### 问题2：车辆被过滤但不应该过滤

**检查调试输出：**
```
车辆 X: ID=XXX
  🔍 检测位置 (x, y), 容差=3.0m
    ❌ 不在任何车道上 (检查了123条车道)
  ❌ 过滤 (原因: 不在车道上)
```

**可能原因：**
1. 车辆确实在非车道区域（草坪/停车场）
2. 容差值太小，可以尝试增大 `lane_tolerance`
3. 车道数据有问题

**解决方案：**
```python
config = {
    "lane_tolerance": 5.0,  # 增大容差到5米
}
```

---

### 问题3：性能下降

启用调试模式会有大量输出，影响性能：

**解决方案：**
- 只在开发/调试时启用
- 生产环境关闭所有调试开关
- 或者只测试少量车辆：
  ```python
  config = {
      "max_controlled_vehicles": 5,  # 只测试5辆车
      "debug_traffic_light": True,
  }
  ```

---

## 📌 最佳实践

1. **开发阶段**：启用调试，理解代码行为
2. **调试问题**：根据需要选择性启用调试
3. **性能测试**：关闭所有调试
4. **生产运行**：永久关闭调试

---

## 🔧 调试输出示例

完整的调试运行示例：

```bash
cd /home/huangfukk/MAGAIL4AutoDrive
python Env/test_lane_filter.py --debug
```

输出会包含：
- 场景统计信息
- 每辆车的详细检测过程
- 最终的过滤/检测结果
- 性能统计

---

## 📖 相关文档

- `README.md` - 项目总览和问题解决
- `CHANGELOG.md` - 更新日志
- `PERFORMANCE_OPTIMIZATION.md` - 性能优化指南

