"""
日志记录功能示例
演示如何在自定义脚本中使用日志功能
"""
from logger_utils import setup_logger
from datetime import datetime
import time

def example_without_logging():
    """示例1：不使用日志"""
    print("=" * 60)
    print("示例1：普通输出（不记录日志）")
    print("=" * 60)
    
    print("这是普通的print输出")
    print("只会显示在终端")
    print("不会保存到文件")
    print()


def example_with_logging():
    """示例2：使用日志记录"""
    print("=" * 60)
    print("示例2：使用日志记录")
    print("=" * 60)
    
    # 使用with语句，自动管理日志文件
    with setup_logger(log_file="example_demo.log", log_dir="logs"):
        print("✅ 这条消息会同时输出到终端和文件")
        print("✅ 运行一些计算...")
        
        for i in range(5):
            print(f"  步骤 {i+1}/5: 处理中...")
            time.sleep(0.1)
        
        print("✅ 计算完成！")
    
    print("日志文件已关闭")
    print()


def example_custom_filename():
    """示例3：使用时间戳命名"""
    print("=" * 60)
    print("示例3：自动生成时间戳文件名")
    print("=" * 60)
    
    # log_file=None 会自动生成时间戳文件名
    with setup_logger(log_file=None, log_dir="logs"):
        print("文件名会自动包含时间戳")
        print("适合批量实验，避免覆盖")
    
    print()


def example_append_mode():
    """示例4：追加模式"""
    print("=" * 60)
    print("示例4：追加到现有文件")
    print("=" * 60)
    
    # 第一次写入
    with setup_logger(log_file="append_test.log", log_dir="logs", mode='w'):
        print("第一次写入：这会覆盖文件")
    
    # 第二次写入（追加）
    with setup_logger(log_file="append_test.log", log_dir="logs", mode='a'):
        print("第二次写入：这会追加到文件末尾")
    
    print()


def example_complex_output():
    """示例5：复杂输出（包含颜色、格式）"""
    print("=" * 60)
    print("示例5：复杂输出格式")
    print("=" * 60)
    
    with setup_logger(log_file="complex_output.log", log_dir="logs"):
        # 模拟多种输出格式
        print("\n📊 实验统计：")
        print("  - 实验名称：车道过滤测试")
        print("  - 开始时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("  - 车辆总数：51")
        print("  - 过滤后：45")
        print("\n🚦 红绿灯检测：")
        print("  ✅ 方法1成功：3辆")
        print("  ✅ 方法2成功：2辆")
        print("  ⚠️ 未检测到：40辆")
        print("\n" + "="*50)
        print("实验完成！")
    
    print()


def main():
    """运行所有示例"""
    print("\n" + "🎯 " + "="*56)
    print("日志记录功能完整示例")
    print("="*60 + "\n")
    
    example_without_logging()
    example_with_logging()
    example_custom_filename()
    example_append_mode()
    example_complex_output()
    
    print("="*60)
    print("✅ 所有示例运行完成！")
    print("📁 查看日志文件：ls -lh logs/")
    print("="*60)


if __name__ == "__main__":
    main()

