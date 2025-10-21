"""
多进程并行版本 - 充分利用多核CPU
适合大规模数据收集和训练
"""
from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader
import time
import os
from multiprocessing import Pool, cpu_count

WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/Env"


def run_single_env(args):
    """在单个进程中运行一个环境实例"""
    seed, num_steps, worker_id = args
    
    # 创建环境（每个进程独立）
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 300,
            
            # 性能优化
            "use_render": False,
            "render_pipeline": False,
            "image_observation": False,
            "interface_panel": [],
            "manual_control": False,
            "show_fps": False,
            "debug": False,
            
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "sequential_seed": True,
            "reactive_traffic": True,
            
            # 车道检测与过滤配置
            "filter_offroad_vehicles": True,
            "lane_tolerance": 3.0,
            "max_controlled_vehicles": 15,
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )
    
    # 启用激光雷达缓存
    env.lidar_cache_interval = 3
    
    # 运行仿真
    start_time = time.time()
    obs = env.reset(seed)
    total_steps = 0
    total_agents = 0
    
    for step in range(num_steps):
        actions = {
            aid: env.controlled_agents[aid].policy.act()
            for aid in env.controlled_agents
        }
        
        obs, rewards, dones, infos = env.step(actions)
        total_steps += 1
        total_agents += len(env.controlled_agents)
        
        if dones["__all__"]:
            break
    
    elapsed = time.time() - start_time
    fps = total_steps / elapsed if elapsed > 0 else 0
    avg_agents = total_agents / total_steps if total_steps > 0 else 0
    
    env.close()
    
    return {
        'worker_id': worker_id,
        'seed': seed,
        'steps': total_steps,
        'elapsed': elapsed,
        'fps': fps,
        'avg_agents': avg_agents,
    }


def main():
    """主函数：协调多个并行环境"""
    # 获取CPU核心数
    num_cores = cpu_count()
    # 建议使用物理核心数（12600KF是10核20线程，使用10个进程）
    num_workers = min(10, num_cores)
    
    print("=" * 80)
    print(f"多进程并行模式")
    print(f"CPU核心数: {num_cores}")
    print(f"并行进程数: {num_workers}")
    print(f"每个环境运行: 1000步")
    print("=" * 80)
    
    # 准备任务参数
    num_steps_per_env = 1000
    tasks = [(seed, num_steps_per_env, worker_id) 
             for worker_id, seed in enumerate(range(num_workers))]
    
    # 启动多进程池
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_env, tasks)
    
    total_elapsed = time.time() - start_time
    
    # 统计结果
    print("\n" + "=" * 80)
    print("各进程执行结果：")
    print("-" * 80)
    print(f"{'Worker':<8} {'Seed':<6} {'Steps':<8} {'Time(s)':<10} {'FPS':<8} {'平均车辆数':<12}")
    print("-" * 80)
    
    total_steps = 0
    total_fps = 0
    
    for result in results:
        print(f"{result['worker_id']:<8} "
              f"{result['seed']:<6} "
              f"{result['steps']:<8} "
              f"{result['elapsed']:<10.2f} "
              f"{result['fps']:<8.2f} "
              f"{result['avg_agents']:<12.1f}")
        total_steps += result['steps']
        total_fps += result['fps']
    
    print("-" * 80)
    avg_fps_per_env = total_fps / len(results)
    total_throughput = total_steps / total_elapsed
    
    print(f"\n总体统计：")
    print(f"  总步数: {total_steps}")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  单环境平均FPS: {avg_fps_per_env:.2f}")
    print(f"  总吞吐量: {total_throughput:.2f} steps/s")
    print(f"  并行效率: {total_throughput / avg_fps_per_env:.1f}x")
    print("=" * 80)
    
    # 与单进程对比
    print(f"\n性能对比：")
    print(f"  单进程FPS (预估): ~30 FPS")
    print(f"  多进程吞吐量: {total_throughput:.2f} steps/s")
    print(f"  性能提升: {total_throughput / 30:.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()

