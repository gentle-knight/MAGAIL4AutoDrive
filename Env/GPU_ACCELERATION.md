# GPU加速指南

## 当前性能瓶颈分析

从测试结果看，即使关闭渲染，FPS仍然只有15-20左右，主要瓶颈是：

### 计算量分析（51辆车）
```
激光雷达计算：
- 前向雷达：80束 × 51车 = 4,080次射线检测
- 侧向雷达：10束 × 51车 = 510次射线检测  
- 车道线雷达：10束 × 51车 = 510次射线检测
合计：5,100次射线检测/帧

红绿灯检测：
- 遍历所有车道 × 51车 = 数千次几何计算
```

**关键问题**：这些计算都是CPU单线程串行的，无法利用多核和GPU！

---

## GPU加速方案

### 方案1：优化激光雷达计算（已实现）✅

**优化内容：**
1. 减少激光束数量：100束 → 52束（减少48%）
2. 优化红绿灯检测：避免遍历所有车道
3. 激光雷达缓存：每N帧才重新计算一次

**预期提升：** 2-4倍（30-60 FPS）

**使用方法：**
```bash
python Env/run_multiagent_env_fast.py
```

---

### 方案2：MetaDrive GPU渲染（有限支持）

**说明：**
MetaDrive基于Panda3D引擎，理论上支持GPU渲染，但：
- GPU主要用于**图形渲染**，不是物理计算
- 激光雷达的射线检测仍在CPU上
- GPU渲染主要加速可视化，不加速训练

**启用方法：**
```python
config = {
    "use_render": True,
    "render_mode": "onscreen",  # 或 "offscreen"
    # Panda3D会自动尝试使用GPU
}
```

**限制：**
- 需要显示器或虚拟显示（Xvfb）
- WSL2环境需要配置X11转发
- 对无渲染训练无帮助

---

### 方案3：使用GPU加速的物理引擎（推荐但需要迁移）

**选项A：Isaac Gym (NVIDIA)**
- 完全在GPU上运行物理模拟和渲染
- 可同时模拟数千个环境
- **缺点**：需要完全重写环境代码，迁移成本高

**选项B：IsaacSim/Omniverse**
- NVIDIA的高级仿真平台
- 支持GPU加速的激光雷达
- **缺点**：学习曲线陡峭，环境配置复杂

**选项C：Brax (Google)**
- JAX驱动，完全在GPU/TPU上运行
- **缺点**：功能有限，不支持复杂场景

---

### 方案4：策略网络GPU加速（推荐）✅

虽然环境仿真在CPU，但可以让**策略网络在GPU上运行**：

```python
import torch

# 创建GPU上的策略模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PolicyNetwork().to(device)

# 批量处理观测
obs_batch = torch.tensor(obs_list).to(device)
with torch.no_grad():
    actions = policy(obs_batch)
actions = actions.cpu().numpy()
```

**优势：**
- 51辆车的推理可以并行
- 如果使用RL训练，GPU加速训练过程
- 不需要修改环境代码

---

### 方案5：多进程并行（最实用）✅

既然单个环境受限于CPU单线程，可以**并行运行多个环境**：

```python
from multiprocessing import Pool
import os

def run_single_env(seed):
    """运行单个环境实例"""
    env = MultiAgentScenarioEnv(config=...)
    obs = env.reset(seed)
    
    for step in range(1000):
        actions = {...}
        obs, rewards, dones, infos = env.step(actions)
        if dones["__all__"]:
            break
    
    env.close()
    return results

# 使用进程池并行运行
if __name__ == "__main__":
    num_processes = os.cpu_count()  # 12600KF有10核20线程
    seeds = list(range(num_processes))
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_env, seeds)
```

**预期提升：** 接近线性（10核 ≈ 10倍吞吐量）

**CPU利用率：** 可达80-100%

---

## 推荐的完整优化方案

### 1. 立即可用（已实现）
```bash
# 使用优化版本，激光束减少+缓存
python Env/run_multiagent_env_fast.py
```
**预期：** 30-60 FPS（2-4倍提升）

### 2. 短期优化（1-2小时）
- 实现多进程并行
- 策略网络迁移到GPU

**预期：** 300-600 FPS（总吞吐量）

### 3. 中期优化（1-2天）
- 使用NumPy矢量化批量处理观测
- 优化Python代码热点（用Cython/Numba）

**预期：** 额外20-30%提升

### 4. 长期方案（1-2周）
- 迁移到Isaac Gym等GPU加速仿真器
- 或使用分布式训练框架（Ray/RLlib）

**预期：** 10-100倍提升

---

## 为什么MetaDrive无法直接使用GPU？

### 架构限制：
1. **物理引擎**：使用Bullet/Panda3D的CPU物理引擎
2. **射线检测**：串行CPU计算，无法并行
3. **Python GIL**：全局解释器锁限制多线程
4. **设计目标**：MetaDrive设计时主要考虑灵活性而非极致性能

### GPU在仿真中的作用：
- ✅ **图形渲染**：绘制画面（但我们训练时不需要）
- ✅ **神经网络推理/训练**：策略模型计算
- ❌ **物理计算**：MetaDrive的物理引擎在CPU
- ❌ **传感器模拟**：激光雷达等在CPU

---

## 检查GPU是否可用

```bash
# 检查NVIDIA GPU
nvidia-smi

# 检查PyTorch GPU支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查MetaDrive渲染设备
python -c "from panda3d.core import GraphicsPipeSelection; print(GraphicsPipeSelection.get_global_ptr().get_default_pipe())"
```

---

## 总结

| 方案 | 实现难度 | 性能提升 | GPU使用 | 推荐度 |
|------|----------|----------|---------|--------|
| 减少激光束 | ⭐ | 2-4x | ❌ | ⭐⭐⭐⭐⭐ |
| 激光雷达缓存 | ⭐ | 1.5-3x | ❌ | ⭐⭐⭐⭐⭐ |
| 多进程并行 | ⭐⭐ | 5-10x | ❌ | ⭐⭐⭐⭐⭐ |
| 策略GPU加速 | ⭐⭐ | 2-5x | ✅ | ⭐⭐⭐⭐ |
| GPU渲染 | ⭐⭐⭐ | 1.2x | ✅ | ⭐⭐ |
| 迁移Isaac Gym | ⭐⭐⭐⭐⭐ | 10-100x | ✅ | ⭐⭐⭐ |

**结论：** 
1. 先用已实现的优化（减少激光束+缓存）
2. 再实现多进程并行
3. 策略网络用GPU训练
4. 如果还不够，考虑迁移到GPU仿真器

