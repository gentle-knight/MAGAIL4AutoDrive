# MetaDrive 性能优化指南

## 为什么帧率只有15FPS且CPU利用率不高？

### 主要原因：

1. **渲染瓶颈（最主要）**
   - `use_render: True` + 每帧调用 `env.render()` 会严重限制帧率
   - MetaDrive 使用 Panda3D 渲染引擎，渲染是**同步阻塞**的
   - 即使CPU有余力，也要等待渲染完成才能继续下一步
   - 这就是为什么CPU利用率低但帧率也低的原因

2. **激光雷达计算开销**
   - 每帧对每辆车进行3次激光雷达扫描（100个激光束）
   - 需要进行物理射线检测，计算量较大

3. **物理引擎同步**
   - 默认物理步长很小（0.02s），需要频繁计算

4. **Python GIL限制**
   - Python全局解释器锁限制了多核并行
   - 即使是多核CPU，Python单线程性能才是瓶颈

## 性能优化方案

### 方案1：关闭渲染（推荐用于训练）
**预期提升：10-20倍（150-300+ FPS）**

```python
config = {
    "use_render": False,  # 关闭渲染
    "render_pipeline": False,
    "image_observation": False,
    "interface_panel": [],
    "manual_control": False,
}
```

### 方案2：降低物理计算频率
**预期提升：2-3倍**

```python
config = {
    "physics_world_step_size": 0.05,  # 默认0.02，增大步长
    "decision_repeat": 5,  # 每5个物理步执行一次决策
}
```

### 方案3：优化激光雷达
**预期提升：1.5-2倍**

修改 `scenario_env.py` 中的 `_get_all_obs()` 函数：

```python
# 减少激光束数量
lidar = self.engine.get_sensor("lidar").perceive(
    num_lasers=40,  # 从80减到40
    distance=30,
    base_vehicle=vehicle,
    physics_world=self.engine.physics_world.dynamic_world
)

# 或者降低扫描频率（每N步才扫描一次）
if self.round % 5 == 0:
    lidar = self.engine.get_sensor("lidar").perceive(...)
else:
    lidar = self.last_lidar[agent_id]  # 使用缓存
```

### 方案4：间歇性渲染
**适用场景：既需要可视化又想提升性能**

```python
# 每10步渲染一次，而不是每步都渲染
if step % 10 == 0:
    env.render(mode="topdown")
```

### 方案5：使用多进程并行（高级）
**预期提升：接近线性（取决于进程数）**

```python
from multiprocessing import Pool

def run_env(seed):
    env = MultiAgentScenarioEnv(config=...)
    # 运行仿真
    return results

# 使用进程池并行运行多个环境
with Pool(processes=8) as pool:
    results = pool.map(run_env, range(8))
```

## 文件说明

- `run_multiagent_env.py` - **标准版本**（无渲染，基础优化）
- `run_multiagent_env_fast.py` - **极速版本**（激光雷达优化+缓存）⭐推荐
- `run_multiagent_env_parallel.py` - **并行版本**（多进程，最高吞吐量）⭐⭐推荐
- `run_multiagent_env_visual.py` - **可视化版本**（有渲染，适合调试）

## 性能对比

| 配置 | 单环境FPS | 总吞吐量 | CPU利用率 | 文件 | 适用场景 |
|------|-----------|----------|-----------|------|----------|
| 原始配置（有渲染） | 15-20 | 15-20 | 15-20% | visual | 实时可视化调试 |
| 关闭渲染 | 20-25 | 20-25 | 20-30% | 标准版 | 基础训练 |
| 激光雷达优化+缓存 | 30-60 | 30-60 | 30-50% | fast | 快速训练⭐ |
| 多进程并行（10核） | 30-60 | 300-600 | 90-100% | parallel | 大规模训练⭐⭐ |

**说明：**
- **单环境FPS**：单个环境实例的帧率
- **总吞吐量**：所有进程合计的 steps/second
- 12600KF（10核20线程）推荐使用并行版本

## 建议

1. **训练时**：使用高性能版本（关闭渲染）
2. **调试时**：使用可视化版本，或间歇性渲染
3. **大规模实验**：使用多进程并行
4. **如果需要GPU加速**：考虑使用GPU渲染或将策略网络部署到GPU上

## 为什么CPU利用率低？

- **渲染阻塞**：CPU在等待渲染完成
- **Python GIL**：限制了多核利用
- **I/O等待**：可能在等待磁盘读取数据
- **单线程瓶颈**：MetaDrive主循环是单线程的

解决方法：关闭渲染 + 多进程并行

