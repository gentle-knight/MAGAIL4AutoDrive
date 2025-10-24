"""测试禁用红绿灯功能"""
from scenario_env import MultiAgentScenarioEnv
from simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader

WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/Env"

def test_no_traffic_lights():
    """测试禁用红绿灯"""
    print("=" * 60)
    print("测试：禁用红绿灯功能")
    print("=" * 60)
    
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 300,
            "use_render": True,
            "sequential_seed": True,
            "reactive_traffic": True,
            "manual_control": True,
            
            # 车道检测与过滤配置
            "filter_offroad_vehicles": True,
            "lane_tolerance": 3.0,
            "max_controlled_vehicles": 2,
            
            # 禁用红绿灯
            "no_traffic_lights": True,  # 关键配置
            
            # 调试模式
            "debug_lane_filter": False,
            "verbose_reset": False,
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )

    print("\n重置环境...")
    obs = env.reset(0)
    
    # 检查红绿灯管理器状态
    if hasattr(env.engine, 'light_manager') and env.engine.light_manager is not None:
        num_lights = len(env.engine.light_manager._lane_index_to_obj)
        print(f"✓ 红绿灯管理器中的红绿灯数量: {num_lights}")
        if num_lights == 0:
            print("✅ 成功：所有红绿灯已被移除！")
        else:
            print(f"⚠️  警告：仍有 {num_lights} 个红绿灯")
    
    print("\n运行几步测试...")
    for step in range(100):
        actions = {
            aid: env.controlled_agents[aid].policy.act()
            for aid in env.controlled_agents
        }
        
        obs, rewards, dones, infos = env.step(actions)
        env.render(mode="topdown")
        
        if step == 0:
            print(f"步骤 {step}: 环境运行正常")
        
        if dones["__all__"]:
            break
    
    print(f"\n测试完成，共运行 {step+1} 步")
    print("请检查渲染窗口中是否还有红绿灯显示")
    
    env.close()
    print("=" * 60)

if __name__ == "__main__":
    test_no_traffic_lights()

