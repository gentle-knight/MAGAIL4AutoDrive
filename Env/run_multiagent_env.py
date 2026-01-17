from scenario_env import MultiAgentScenarioEnv
from Env.simple_idm_policy import ConstantVelocityPolicy
from metadrive.engine.asset_loader import AssetLoader

WAYMO_DATA_DIR = r"/home/huangfukk/MAGAIL4AutoDrive/data"

def main():
    env = MultiAgentScenarioEnv(
        config={
            # "data_directory": AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False),
            "data_directory": AssetLoader.file_path(WAYMO_DATA_DIR, "exp_converted", unix_style=False),
            "is_multi_agent": True,
            "num_controlled_agents": 3,
            "horizon": 300,
            "use_render": True,
            "sequential_seed": True,
            "reactive_traffic": True,
            "manual_control": True,
        },
        agent2policy=ConstantVelocityPolicy(target_speed=50)
    )

    obs = env.reset(0
                    )
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
    main()