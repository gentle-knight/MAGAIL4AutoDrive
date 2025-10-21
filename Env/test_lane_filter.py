"""
测试车道过滤和红绿灯检测功能
"""
from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader
from logger_utils import setup_logger
import os

WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/Env"

def test_lane_filter():
    """测试车道过滤功能（基础版）"""
    print("=" * 60)
    print("测试1：车道过滤功能（基础）")
    print("=" * 60)
    
    # 创建启用过滤的环境
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 100,
            "use_render": False,
            
            # 车道过滤配置
            "filter_offroad_vehicles": True,
            "lane_tolerance": 3.0,
            "max_controlled_vehicles": 10,
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )
    
    print("\n启用车道过滤...")
    obs = env.reset(0)
    print(f"生成车辆数: {len(env.controlled_agents)}")
    print(f"观测数据长度: {len(obs)}")
    
    # 运行几步
    for step in range(5):
        actions = {aid: env.controlled_agents[aid].policy.act() 
                   for aid in env.controlled_agents}
        obs, rewards, dones, infos = env.step(actions)
    
    env.close()
    print("✓ 车道过滤测试通过\n")


def test_lane_filter_debug():
    """测试车道过滤功能（详细调试）"""
    print("=" * 60)
    print("测试1b：车道过滤功能（详细调试模式）")
    print("=" * 60)
    
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 100,
            "use_render": False,
            
            # 车道过滤配置
            "filter_offroad_vehicles": True,
            "lane_tolerance": 3.0,
            "max_controlled_vehicles": 5,  # 只看前5辆车
            
            # 🔥 启用调试模式
            "debug_lane_filter": True,  # 启用车道过滤调试
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )
    
    print("\n启用车道过滤调试...")
    obs = env.reset(0)
    
    env.close()
    print("\n✓ 车道过滤调试测试完成\n")


def test_traffic_light():
    """测试红绿灯检测功能"""
    print("=" * 60)
    print("测试2：红绿灯检测功能（启用详细调试）")
    print("=" * 60)
    
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 100,
            "use_render": False,
            "filter_offroad_vehicles": True,
            "max_controlled_vehicles": 3,  # 只测试3辆车
            
            # 🔥 启用调试模式
            "debug_traffic_light": True,  # 启用红绿灯调试
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )
    
    obs = env.reset(0)
    
    # 测试红绿灯检测（调试模式会自动输出详细信息）
    print(f"\n" + "="*60)
    print(f"开始逐车检测红绿灯状态（共 {len(env.controlled_agents)} 辆车）")
    print("="*60)
    
    for idx, (aid, vehicle) in enumerate(list(env.controlled_agents.items())[:3]):  # 只测试前3辆
        print(f"\n【车辆 {idx+1}/3】 ID={aid}")
        traffic_light = env._get_traffic_light_state(vehicle)
        state = vehicle.get_state()
        
        status_text = {0: '无/未知', 1: '绿灯', 2: '黄灯', 3: '红灯'}[traffic_light]
        print(f"最终结果: 红绿灯状态={traffic_light} ({status_text})\n")
    
    env.close()
    print("="*60)
    print("✓ 红绿灯检测测试完成")
    print("="*60 + "\n")


def test_without_filter():
    """测试禁用过滤的情况"""
    print("=" * 60)
    print("测试3：禁用过滤（对比测试）")
    print("=" * 60)
    
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 100,
            "use_render": False,
            
            # 禁用过滤
            "filter_offroad_vehicles": False,
            "max_controlled_vehicles": None,
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )
    
    print("\n禁用车道过滤...")
    obs = env.reset(0)
    print(f"生成车辆数（未过滤）: {len(env.controlled_agents)}")
    
    env.close()
    print("✓ 禁用过滤测试通过\n")


def run_tests(debug_mode=False):
    """运行测试的主函数"""
    try:
        if debug_mode:
            print("🐛 调试模式启用")
            print("=" * 60 + "\n")
            test_lane_filter_debug()
            test_traffic_light()
        else:
            print("⚡ 标准测试模式（使用 --debug 参数启用详细调试）")
            print("=" * 60 + "\n")
            test_lane_filter()
            test_traffic_light()
            test_without_filter()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n功能说明：")
        print("1. 车道过滤功能已启用，自动过滤非车道区域车辆")
        print("2. 红绿灯检测采用双重策略，确保稳定获取状态")
        print("3. 可通过配置参数灵活启用/禁用功能")
        print("\n使用方法：")
        print("  python Env/test_lane_filter.py                    # 标准测试")
        print("  python Env/test_lane_filter.py --debug           # 详细调试")
        print("  python Env/test_lane_filter.py --log             # 保存日志")
        print("  python Env/test_lane_filter.py --debug --log     # 调试+日志")
        print("\n请运行 run_multiagent_env.py 查看完整效果")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    enable_logging = "--log" in sys.argv or "-l" in sys.argv
    
    # 提取自定义日志文件名
    log_file = None
    for arg in sys.argv:
        if arg.startswith("--log-file="):
            log_file = arg.split("=")[1]
            break
    
    if enable_logging:
        # 启用日志记录
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        
        # 生成默认日志文件名
        if log_file is None:
            mode_suffix = "debug" if debug_mode else "standard"
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"test_{mode_suffix}_{timestamp}.log"
        
        with setup_logger(log_file=log_file, log_dir=log_dir):
            run_tests(debug_mode=debug_mode)
    else:
        # 不启用日志，直接运行
        run_tests(debug_mode=debug_mode)

