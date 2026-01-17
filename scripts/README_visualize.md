# 模型可视化脚本使用说明

## 功能
使用训练好的MAGAIL模型在环境中运行，并生成俯瞰效果图（top-down view）。

## 使用方法

### 基本用法

```bash
python scripts/visualize_trained_model.py \
    --model_dir runs/magail_0113 \
    --episode 1250 \
    --data_dir data/exp_filtered \
    --num_scenarios 1 \
    --output_dir visualizations
```

### 参数说明

- `--model_dir`: 模型保存目录（例如：`runs/magail_0113`）
- `--episode`: 要加载的episode编号（例如：`1250`）
- `--data_dir`: Waymo数据目录（默认：`data/exp_filtered`）
- `--start_index`: 起始场景索引（默认：`0`）
- `--num_scenarios`: 要运行的场景数量（默认：`1`）
- `--horizon`: 每个episode的最大步数（默认：`200`）
- `--output_dir`: 输出图像保存目录（默认：`visualizations`）
- `--save_all_frames`: 保存所有帧（否则按间隔保存）
- `--save_interval`: 保存帧的间隔，当不使用`--save_all_frames`时生效（默认：`10`）
- `--gif_duration`: GIF每帧持续时间（毫秒），默认50ms（20fps）。值越小，GIF播放越快

### 示例

#### 1. 查看最新训练的模型（episode 1250）
```bash
python scripts/visualize_trained_model.py \
    --model_dir runs/magail_0113 \
    --episode 1250 \
    --num_scenarios 3 \
    --output_dir visualizations/episode_1250
```

#### 2. 保存所有帧（用于制作视频）
```bash
python scripts/visualize_trained_model.py \
    --model_dir runs/magail_0113 \
    --episode 1250 \
    --save_all_frames \
    --output_dir visualizations/episode_1250_all_frames
```

#### 3. 每5步保存一帧
```bash
python scripts/visualize_trained_model.py \
    --model_dir runs/magail_0113 \
    --episode 1250 \
    --save_interval 5 \
    --output_dir visualizations/episode_1250_sparse
```

#### 4. 生成更快的GIF（30fps）
```bash
python scripts/visualize_trained_model.py \
    --model_dir runs/magail_0113 \
    --episode 1250 \
    --gif_duration 33 \
    --output_dir visualizations/episode_1250
```

## 输出

脚本会在指定的输出目录中创建以下文件：
- `scenario_{idx}.gif`: **场景动画GIF**（主要输出）
- `scenario_{idx}_step_{step:04d}.png`: 每个保存步骤的俯瞰图（可选）
- `scenario_{idx}_final.png`: 每个场景的最终状态图

### GIF格式
- 分辨率：1600x900
- 格式：GIF动画
- 包含完整的场景运行过程
- 显示场景编号、步数、智能体数量和奖励信息
- 默认帧率：20fps（可通过`--gif_duration`调整）

### 图像格式
- 分辨率：1600x900
- 格式：PNG
- 包含语义地图和车辆轨迹

## 注意事项

1. **GPU要求**: 脚本需要CUDA支持，如果没有GPU会自动使用CPU（速度较慢）
2. **渲染模式**: 使用MetaDrive的top-down渲染模式，会弹出窗口显示实时渲染
3. **内存占用**: 如果保存所有帧，会占用较多磁盘空间
4. **场景数据**: 确保`--data_dir`指向正确的Waymo数据目录

## 故障排除

### 模型文件不存在
```
FileNotFoundError: 模型文件不存在: runs/magail_0113/model_1250_actor.pth
```
**解决**: 检查模型目录和episode编号是否正确

### 场景数据不存在
```
ValueError: Data directory not found
```
**解决**: 确保`--data_dir`指向正确的数据目录

### 渲染失败
如果遇到渲染相关错误，可以尝试：
- 降低`film_size`参数（在脚本中修改）
- 使用无头模式（需要修改脚本）
