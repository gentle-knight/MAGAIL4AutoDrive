"""
日志工具模块
提供将终端输出同时保存到文件的功能
"""
import sys
import os
from datetime import datetime


class TeeLogger:
    """
    双向输出类：同时输出到终端和文件
    """
    def __init__(self, filename, mode='w', terminal=None):
        """
        Args:
            filename: 日志文件路径
            mode: 文件打开模式 ('w'=覆盖, 'a'=追加)
            terminal: 原始输出流（通常是sys.stdout或sys.stderr）
        """
        self.terminal = terminal or sys.stdout
        self.log_file = open(filename, mode, encoding='utf-8')
        
    def write(self, message):
        """写入消息到终端和文件"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 立即写入磁盘
        
    def flush(self):
        """刷新缓冲区"""
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()


class LoggerContext:
    """
    日志上下文管理器
    使用with语句自动管理日志的开启和关闭
    """
    def __init__(self, log_file=None, log_dir="logs", mode='w', 
                 redirect_stdout=True, redirect_stderr=True):
        """
        Args:
            log_file: 日志文件名（None则自动生成时间戳文件名）
            log_dir: 日志目录
            mode: 文件打开模式 ('w'=覆盖, 'a'=追加)
            redirect_stdout: 是否重定向标准输出
            redirect_stderr: 是否重定向标准错误
        """
        self.log_dir = log_dir
        self.mode = mode
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"run_{timestamp}.log"
        
        self.log_path = os.path.join(log_dir, log_file)
        
        # 保存原始的stdout和stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 日志对象
        self.stdout_logger = None
        self.stderr_logger = None
        
    def __enter__(self):
        """进入上下文：开启日志"""
        print(f"📝 日志记录已启用")
        print(f"📁 日志文件: {self.log_path}")
        print("-" * 60)
        
        # 创建TeeLogger对象
        if self.redirect_stdout:
            self.stdout_logger = TeeLogger(
                self.log_path, 
                mode=self.mode, 
                terminal=self.original_stdout
            )
            sys.stdout = self.stdout_logger
            
        if self.redirect_stderr:
            self.stderr_logger = TeeLogger(
                self.log_path, 
                mode='a',  # stderr总是追加模式
                terminal=self.original_stderr
            )
            sys.stderr = self.stderr_logger
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：关闭日志"""
        # 恢复原始输出
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # 关闭日志文件
        if self.stdout_logger:
            self.stdout_logger.close()
        if self.stderr_logger:
            self.stderr_logger.close()
            
        print("-" * 60)
        print(f"✅ 日志已保存到: {self.log_path}")
        
        # 返回False表示不抑制异常
        return False


def setup_logger(log_file=None, log_dir="logs", mode='w'):
    """
    快速设置日志记录
    
    Args:
        log_file: 日志文件名（None则自动生成）
        log_dir: 日志目录
        mode: 文件模式 ('w'=覆盖, 'a'=追加)
        
    Returns:
        LoggerContext对象
        
    Example:
        with setup_logger("my_test.log"):
            print("这条消息会同时输出到终端和文件")
    """
    return LoggerContext(log_file=log_file, log_dir=log_dir, mode=mode)


def get_default_log_filename(prefix="run"):
    """
    生成默认的日志文件名（带时间戳）
    
    Args:
        prefix: 文件名前缀
        
    Returns:
        str: 格式为 "prefix_YYYYMMDD_HHMMSS.log"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.log"


if __name__ == "__main__":
    # 测试代码
    print("测试1: 使用默认配置")
    with setup_logger():
        print("这是测试消息1")
        print("这是测试消息2")
    print("日志记录已结束\n")
    
    print("测试2: 使用自定义文件名")
    with setup_logger(log_file="test_custom.log"):
        print("自定义文件名测试")
        for i in range(3):
            print(f"  消息 {i+1}")
    print("完成")

