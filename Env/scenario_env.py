import numpy as np
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle, vehicle_class_to_type
import math
import logging
from collections import defaultdict
from typing import Union, Dict, AnyStr
from metadrive.engine.logger import get_logger, set_log_level
from metadrive.type import MetaDriveType


class PolicyVehicle(DefaultVehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
        self.destination = None

    def set_policy(self, policy):
        self.policy = policy

    def set_destination(self, des):
        self.destination = des

    def act(self, observation, policy=None):
        if self.policy is not None:
            return self.policy.act(observation)
        else:
            return self.action_space.sample()

    def before_step(self, action):
        self.last_position = self.position  # 2D vector
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar
        self.last_heading_dir = self.heading
        if action is not None:
            self.last_current_action.append(action)
        self._set_action(action)

    def is_done(self):
        # arrive or crash
        pass


vehicle_class_to_type[PolicyVehicle] = "default"


class MultiAgentScenarioEnv(ScenarioEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(
            data_directory=None,
            num_controlled_agents=3,
            horizon=1000,
        ))
        return config

    def __init__(self, config, agent2policy):
        self.policy = agent2policy
        self.controlled_agents = {}
        self.controlled_agent_ids = []
        self.obs_list = []
        self.round = 0
        super().__init__(config)

    def reset(self, seed: Union[None, int] = None):
        self.round = 0
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)

        self.lazy_init()
        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError("Broken MetaDrive instance.")

        # 记录专家数据中每辆车的位置，接着全部清除，只保留位置等信息，用于后续生成
        _obj_to_clean_this_frame = []
        self.car_birth_info_list = []
        for scenario_id, track in self.engine.traffic_manager.current_traffic_data.items():
            if scenario_id == self.engine.traffic_manager.sdc_scenario_id:
                continue
            else:
                if track["type"] == MetaDriveType.VEHICLE:
                    _obj_to_clean_this_frame.append(scenario_id)
                    valid = track['state']['valid']
                    first_show = np.argmax(valid) if valid.any() else -1
                    last_show = len(valid) - 1 - np.argmax(valid[::-1]) if valid.any() else -1
                    # id，出现时间，出生点坐标，出生朝向，目的地
                    self.car_birth_info_list.append({
                        'id': track['metadata']['object_id'],
                        'show_time': first_show,
                        'begin': (track['state']['position'][first_show, 0], track['state']['position'][first_show, 1]),
                        'heading': track['state']['heading'][first_show],
                        'end': (track['state']['position'][last_show, 0], track['state']['position'][last_show, 1])
                    })

        for scenario_id in _obj_to_clean_this_frame:
            self.engine.traffic_manager.current_traffic_data.pop(scenario_id)

        self.engine.reset()
        self.reset_sensors()
        self.engine.taskMgr.step()

        self.lanes = self.engine.map_manager.current_map.road_network.graph

        if self.top_down_renderer is not None:
            self.top_down_renderer.clear()
            self.engine.top_down_renderer = None

        self.dones = {}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        self.controlled_agents.clear()
        self.controlled_agent_ids.clear()

        super().reset(seed)  # 初始化场景
        self._spawn_controlled_agents()

        return self._get_all_obs()

    def _spawn_controlled_agents(self):
        # ego_vehicle = self.engine.agent_manager.active_agents.get("default_agent")
        # ego_position = ego_vehicle.position if ego_vehicle else np.array([0, 0])
        for car in self.car_birth_info_list:
            if car['show_time'] == self.round:
                agent_id = f"controlled_{car['id']}"

                vehicle = self.engine.spawn_object(
                    PolicyVehicle,
                    vehicle_config={},
                    position=car['begin'],
                    heading=car['heading']
                )
                vehicle.reset(position=car['begin'], heading=car['heading'])

                vehicle.set_policy(self.policy)
                vehicle.set_destination(car['end'])

                self.controlled_agents[agent_id] = vehicle
                self.controlled_agent_ids.append(agent_id)

                # ✅ 关键：注册到引擎的 active_agents，才能参与物理更新
                self.engine.agent_manager.active_agents[agent_id] = vehicle

    def _get_all_obs(self):
        # position, velocity, heading, lidar, navigation, TODO: trafficlight -> list
        self.obs_list = []
        for agent_id, vehicle in self.controlled_agents.items():
            state = vehicle.get_state()

            traffic_light = 0
            for lane in self.lanes.values():
                if lane.lane.point_on_lane(state['position'][:2]):
                    if self.engine.light_manager.has_traffic_light(lane.lane.index):
                        traffic_light = self.engine.light_manager._lane_index_to_obj[lane.lane.index].status
                        if traffic_light == 'TRAFFIC_LIGHT_GREEN':
                            traffic_light = 1
                        elif traffic_light == 'TRAFFIC_LIGHT_YELLOW':
                            traffic_light = 2
                        elif traffic_light == 'TRAFFIC_LIGHT_RED':
                            traffic_light = 3
                        else:
                            traffic_light = 0
                    break

            lidar = self.engine.get_sensor("lidar").perceive(num_lasers=80, distance=30, base_vehicle=vehicle,
                                                             physics_world=self.engine.physics_world.dynamic_world)
            side_lidar = self.engine.get_sensor("side_detector").perceive(num_lasers=10, distance=8,
                                                                          base_vehicle=vehicle,
                                                                          physics_world=self.engine.physics_world.static_world)
            lane_line_lidar = self.engine.get_sensor("lane_line_detector").perceive(num_lasers=10, distance=3,
                                                                                    base_vehicle=vehicle,
                                                                                    physics_world=self.engine.physics_world.static_world)

            obs = (state['position'][:2] + list(state['velocity']) + [state['heading_theta']]
                   + lidar[0] + side_lidar[0] + lane_line_lidar[0] + [traffic_light]
                   + list(vehicle.destination))
            self.obs_list.append(obs)
        return self.obs_list

    def step(self, action_dict: Dict[AnyStr, Union[list, np.ndarray]]):
        self.round += 1

        for agent_id, action in action_dict.items():
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].before_step(action)

        self.engine.step()

        for agent_id in action_dict:
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].after_step()

        self._spawn_controlled_agents()
        obs = self._get_all_obs()
        rewards = {aid: 0.0 for aid in self.controlled_agents}
        dones = {aid: False for aid in self.controlled_agents}
        dones["__all__"] = self.episode_step >= self.config["horizon"]
        infos = {aid: {} for aid in self.controlled_agents}
        return obs, rewards, dones, infos
