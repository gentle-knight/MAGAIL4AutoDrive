# 日志记录功能使用指南

## 📋 概述

为所有运行脚本添加了日志记录功能，可以将终端输出同时保存到文本文件，方便后续分析和问题排查。

---

## 🎯 功能特点

1. **双向输出**：同时输出到终端和文件，不影响实时查看
2. **自动管理**：使用上下文管理器，自动处理文件开启/关闭
3. **灵活配置**：支持自定义文件名和日志目录
4. **时间戳命名**：默认使用时间戳生成唯一文件名
5. **无缝集成**：只需添加命令行参数，无需修改代码

---

## 🚀 快速使用

### 1. 基础用法

```bash
# 不启用日志（默认）
python Env/run_multiagent_env.py

# 启用日志记录
python Env/run_multiagent_env.py --log

# 或使用短选项
python Env/run_multiagent_env.py -l
```

### 2. 自定义文件名

```bash
# 使用自定义日志文件名
python Env/run_multiagent_env.py --log --log-file=my_test.log

# 测试脚本也支持
python Env/test_lane_filter.py --log --log-file=test_results.log
```

### 3. 组合使用调试和日志

```bash
# 测试脚本：调试模式 + 日志记录
python Env/test_lane_filter.py --debug --log

# 会生成类似：test_debug_20251021_123456.log
```

---

## 📁 日志文件位置

默认日志目录：`Env/logs/`

### 文件命名规则

| 脚本 | 默认文件名格式 | 示例 |
|------|---------------|------|
| `run_multiagent_env.py` | `run_YYYYMMDD_HHMMSS.log` | `run_20251021_143022.log` |
| `run_multiagent_env_fast.py` | `run_fast.log` | `run_fast.log` |
| `test_lane_filter.py` | `test_{mode}_YYYYMMDD_HHMMSS.log` | `test_debug_20251021_143500.log` |

**说明**：
- `YYYYMMDD_HHMMSS` 是时间戳（年月日_时分秒）
- `{mode}` 是测试模式（`standard` 或 `debug`）

---

## 📝 所有支持的脚本

### 1. run_multiagent_env.py（标准运行脚本）

```bash
# 不启用日志
python Env/run_multiagent_env.py

# 启用日志（自动生成时间戳文件名）
python Env/run_multiagent_env.py --log

# 自定义文件名
python Env/run_multiagent_env.py --log --log-file=run_test1.log
```

**日志位置**：`Env/logs/run_YYYYMMDD_HHMMSS.log`

---

### 2. run_multiagent_env_fast.py（高性能版本）

```bash
# 启用日志
python Env/run_multiagent_env_fast.py --log

# 自定义文件名
python Env/run_multiagent_env_fast.py --log --log-file=fast_test.log
```

**日志位置**：`Env/logs/run_fast.log`（默认）

---

### 3. test_lane_filter.py（测试脚本）

```bash
# 标准测试 + 日志
python Env/test_lane_filter.py --log

# 调试测试 + 日志
python Env/test_lane_filter.py --debug --log

# 自定义文件名
python Env/test_lane_filter.py --log --log-file=my_test.log

# 组合使用
python Env/test_lane_filter.py --debug --log --log-file=debug_run.log
```

**日志位置**：
- 标准模式：`Env/logs/test_standard_YYYYMMDD_HHMMSS.log`
- 调试模式：`Env/logs/test_debug_YYYYMMDD_HHMMSS.log`

---

## 💻 编程接口

如果您想在代码中直接使用日志功能：

```python
from logger_utils import setup_logger

# 方式1：使用上下文管理器（推荐）
with setup_logger(log_file="my_log.log", log_dir="logs"):
    print("这条消息会同时输出到终端和文件")
    # 运行您的代码
    # ...

# 方式2：手动管理
from logger_utils import LoggerContext

logger = LoggerContext(log_file="custom.log", log_dir="output")
logger.__enter__()  # 开启日志
print("输出消息")
logger.__exit__(None, None, None)  # 关闭日志
```

---

## 📊 日志内容示例

### 标准运行

```
📝 日志记录已启用
📁 日志文件: Env/logs/run_20251021_143022.log
------------------------------------------------------------
💡 提示: 使用 --log 或 -l 参数启用日志记录
   示例: python run_multiagent_env.py --log
   自定义文件名: python run_multiagent_env.py --log --log-file=my_run.log
------------------------------------------------------------
[INFO] Environment: MultiAgentScenarioEnv
[INFO] MetaDrive version: 0.4.3
...
------------------------------------------------------------
✅ 日志已保存到: Env/logs/run_20251021_143022.log
```

### 调试模式

```
📝 日志记录已启用
📁 日志文件: Env/logs/test_debug_20251021_143500.log
------------------------------------------------------------
🐛 调试模式启用
============================================================

📍 场景信息统计:
  - 总车道数: 123
  - 有红绿灯的车道数: 0
    ⚠️ 场景中没有红绿灯！

🔍 开始车道过滤: 共 51 辆车待检测
...
------------------------------------------------------------
✅ 日志已保存到: Env/logs/test_debug_20251021_143500.log
```

---

## 🔧 高级配置

### 自定义日志目录

```python
from logger_utils import setup_logger

# 指定不同的日志目录
with setup_logger(log_file="test.log", log_dir="my_logs"):
    print("日志会保存到 my_logs/test.log")
```

### 追加模式

```python
from logger_utils import setup_logger

# 追加到现有文件（而不是覆盖）
with setup_logger(log_file="test.log", mode='a'):  # mode='a' 表示追加
    print("这条消息会追加到文件末尾")
```

### 只重定向特定输出

```python
from logger_utils import LoggerContext

# 只重定向stdout，不重定向stderr
logger = LoggerContext(
    log_file="test.log",
    redirect_stdout=True,   # 重定向标准输出
    redirect_stderr=False   # 不重定向错误输出
)
```

---

## 📋 命令行参数总结

| 参数 | 短选项 | 说明 | 示例 |
|------|--------|------|------|
| `--log` | `-l` | 启用日志记录 | `--log` |
| `--log-file=NAME` | 无 | 指定日志文件名 | `--log-file=test.log` |
| `--debug` | `-d` | 启用调试模式（test_lane_filter.py） | `--debug` |

### 参数组合

```bash
# 示例1：标准模式 + 日志
python Env/test_lane_filter.py --log

# 示例2：调试模式 + 日志
python Env/test_lane_filter.py --debug --log

# 示例3：调试 + 自定义文件名
python Env/test_lane_filter.py -d --log --log-file=my_debug.log

# 示例4：所有参数
python Env/test_lane_filter.py --debug --log --log-file=full_test.log
```

---

## 🛠️ 常见问题

### Q1: 日志文件在哪里？

**A**: 默认在 `Env/logs/` 目录下。如果目录不存在，会自动创建。

```bash
# 查看所有日志文件
ls -lh Env/logs/

# 查看最新的日志
ls -lt Env/logs/ | head -5
```

---

### Q2: 如何查看日志内容？

**A**: 使用任何文本编辑器或命令行工具：

```bash
# 方式1：使用cat
cat Env/logs/run_20251021_143022.log

# 方式2：使用less（可翻页）
less Env/logs/run_20251021_143022.log

# 方式3：查看末尾内容
tail -n 50 Env/logs/run_20251021_143022.log

# 方式4：实时监控（适合长时间运行）
tail -f Env/logs/run_20251021_143022.log
```

---

### Q3: 日志文件太多怎么办？

**A**: 可以定期清理旧日志：

```bash
# 删除7天前的日志
find Env/logs/ -name "*.log" -mtime +7 -delete

# 只保留最新的10个日志
cd Env/logs && ls -t *.log | tail -n +11 | xargs rm -f
```

---

### Q4: 日志会影响性能吗？

**A**: 影响很小，因为：
1. 文件I/O是异步的
2. 使用了缓冲区
3. 立即刷新确保数据不丢失

如果追求极致性能，建议训练时不启用日志，只在需要分析时启用。

---

### Q5: 可以同时记录多个脚本的日志吗？

**A**: 可以，每个脚本使用不同的日志文件：

```bash
# 终端1
python Env/run_multiagent_env.py --log --log-file=script1.log

# 终端2（同时运行）
python Env/test_lane_filter.py --log --log-file=script2.log
```

---

## 💡 最佳实践

### 1. 开发阶段

```bash
# 使用调试模式 + 日志，方便排查问题
python Env/test_lane_filter.py --debug --log
```

### 2. 长时间运行

```bash
# 启用日志，避免输出丢失
nohup python Env/run_multiagent_env.py --log > /dev/null 2>&1 &

# 查看实时输出
tail -f Env/logs/run_*.log
```

### 3. 批量实验

```bash
# 为每次实验使用不同的日志文件
for i in {1..5}; do
    python Env/run_multiagent_env.py --log --log-file=exp_${i}.log
done
```

### 4. 性能测试

```bash
# 不启用日志，获得最佳性能
python Env/run_multiagent_env_fast.py
```

---

## 📖 相关文档

- `README.md` - 项目总览
- `DEBUG_GUIDE.md` - 调试功能使用指南
- `CHANGELOG.md` - 更新日志

---

## 🔍 技术细节

### 实现原理

1. **TeeLogger类**：实现同时写入终端和文件
2. **上下文管理器**：自动管理资源（文件打开/关闭）
3. **sys.stdout重定向**：拦截所有print输出
4. **即时刷新**：每次写入后立即刷新，确保数据不丢失

### 源代码

详见 `Env/logger_utils.py`

```python
# 简化示例
class TeeLogger:
    def write(self, message):
        self.terminal.write(message)  # 输出到终端
        self.log_file.write(message)  # 写入文件
        self.log_file.flush()         # 立即刷新
```

---

## ✅ 总结

- ✅ 简单易用：只需添加 `--log` 参数
- ✅ 不影响输出：终端仍可实时查看
- ✅ 自动管理：文件自动开启/关闭
- ✅ 灵活配置：支持自定义文件名和目录
- ✅ 完整记录：包含所有调试信息

立即开始使用：

```bash
python Env/test_lane_filter.py --debug --log
```

