from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader
from logger_utils import setup_logger
import time
import sys
import os

WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/Env"

def main(enable_logging=False):
    """极致性能优化版本 - 启用所有优化选项"""
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 300,
            
            # 关闭所有渲染
            "use_render": False,
            "render_pipeline": False,
            "image_observation": False,
            "interface_panel": [],
            "manual_control": False,
            "show_fps": False,
            "debug": False,
            
            # 物理引擎优化
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            
            "sequential_seed": True,
            "reactive_traffic": True,
            
            # 车道检测与过滤配置
            "filter_offroad_vehicles": True,  # 过滤非车道区域的车辆
            "lane_tolerance": 3.0,
            "max_controlled_vehicles": 15,  # 限制车辆数以提升性能
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )
    
    # 【关键优化】启用激光雷达缓存
    # 每3帧才重新计算激光雷达，其余帧使用缓存
    # 可将激光雷达计算量减少到原来的1/3
    env.lidar_cache_interval = 3
    
    obs = env.reset(0)
    
    # 性能统计
    start_time = time.time()
    total_steps = 0
    
    print("=" * 60)
    print("极致性能模式")
    print("激光雷达优化：80→40束 (前向), 10→6束 (侧向+车道线)")
    print("激光雷达缓存：每3帧计算一次，中间帧使用缓存")
    print("预期性能提升：3-5倍")
    print("=" * 60)
    
    for step in range(10000):
        actions = {
            aid: env.controlled_agents[aid].policy.act()
            for aid in env.controlled_agents
        }

        obs, rewards, dones, infos = env.step(actions)
        total_steps += 1
        
        # 每100步输出一次性能统计
        if step % 100 == 0 and step > 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            print(f"Step {step:4d}: FPS = {fps:6.2f}, 车辆数 = {len(env.controlled_agents):3d}, "
                  f"平均步时间 = {1000/fps:.2f}ms")

        if dones["__all__"]:
            break

    # 最终统计
    elapsed = time.time() - start_time
    fps = total_steps / elapsed
    print("\n" + "=" * 60)
    print(f"总计: {total_steps} 步")
    print(f"耗时: {elapsed:.2f}s")
    print(f"平均FPS: {fps:.2f}")
    print(f"单步平均耗时: {1000/fps:.2f}ms")
    print("=" * 60)
    
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
        with setup_logger(log_file=log_file or "run_fast.log", log_dir=log_dir):
            main(enable_logging=True)
    else:
        # 普通运行（只输出到终端）
        print("💡 提示: 使用 --log 或 -l 参数启用日志记录")
        print("-" * 60)
        main(enable_logging=False)

