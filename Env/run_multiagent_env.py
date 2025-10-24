from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader
from logger_utils import setup_logger
import sys
import os

WAYMO_DATA_DIR = r"/home/huangfukk/mdsn"

def main(enable_logging=False, log_file=None):
    """
    主函数
    
    Args:
        enable_logging: 是否启用日志记录到文件
        log_file: 日志文件名（None则自动生成时间戳文件名）
    """
    env = MultiAgentScenarioEnv(
        config={
            # "data_directory": AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False),
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_filtered", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 300,
            "use_render": True,
            "sequential_seed": True,
            "reactive_traffic": True,
            "manual_control": True,
            
            # 车道检测与过滤配置
            "filter_offroad_vehicles": True,  # 启用车道区域过滤，过滤草坪等非车道区域的车辆
            "lane_tolerance": 3.0,  # 车道检测容差（米），可根据需要调整
            "max_controlled_vehicles": None,  # 限制最大车辆数（可选，None表示不限制）
            
            # 调试配置（可选）
            # "debug_lane_filter": True,  # 启用车道过滤详细调试
            # "verbose_reset": True,  # 启用重置详细统计
            # "inherit_expert_velocity": True,  # 继承专家速度
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )

    obs = env.reset(0)
    for step in range(10000):
        actions = {
            aid: env.controlled_agents[aid].policy.act()
            for aid in env.controlled_agents
        }

        obs, rewards, dones, infos = env.step(actions)
        env.render(mode="topdown")

        if dones["__all__"]:
            break

    env.close()


if __name__ == "__main__":
    # 解析命令行参数
    enable_logging = "--log" in sys.argv or "-l" in sys.argv
    
    # 提取自定义日志文件名
    log_file = None
    for arg in sys.argv:
        if arg.startswith("--log-file="):
            log_file = arg.split("=")[1]
            break
    
    if enable_logging:
        # 使用日志记录
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        with setup_logger(log_file=log_file, log_dir=log_dir):
            main(enable_logging=True, log_file=log_file)
    else:
        # 普通运行（只输出到终端）
        print("💡 提示: 使用 --log 或 -l 参数启用日志记录")
        print("   示例: python run_multiagent_env.py --log")
        print("   自定义文件名: python run_multiagent_env.py --log --log-file=my_run.log")
        print("-" * 60)
        main(enable_logging=False)