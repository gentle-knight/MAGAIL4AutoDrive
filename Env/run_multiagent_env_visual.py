from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader
import time

WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/Env"

def main():
    """带可视化的版本（低FPS，约15帧）"""
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 300,
            
            # 可视化设置（牺牲性能）
            "use_render": True,
            "manual_control": False,
            
            "sequential_seed": True,
            "reactive_traffic": True,
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )

    obs = env.reset(0)
    
    start_time = time.time()
    total_steps = 0
    
    for step in range(10000):
        actions = {
            aid: env.controlled_agents[aid].policy.act()
            for aid in env.controlled_agents
        }

        obs, rewards, dones, infos = env.step(actions)
        env.render(mode="topdown")  # 实时渲染
        
        total_steps += 1
        
        if step % 100 == 0 and step > 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            print(f"Step {step}: FPS = {fps:.2f}, 车辆数 = {len(env.controlled_agents)}")

        if dones["__all__"]:
            break

    elapsed = time.time() - start_time
    fps = total_steps / elapsed
    print(f"\n总计: {total_steps} 步，耗时 {elapsed:.2f}s，平均FPS = {fps:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()

