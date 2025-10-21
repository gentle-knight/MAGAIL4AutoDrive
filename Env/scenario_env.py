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
            # 车道检测与过滤配置
            filter_offroad_vehicles=True,  # 是否过滤非车道区域的车辆
            lane_tolerance=3.0,  # 车道检测容差（米），用于放宽边界条件
            max_controlled_vehicles=None,  # 最大可控车辆数限制（None表示不限制）
            # 调试模式配置
            debug_traffic_light=False,  # 是否启用红绿灯检测调试输出
            debug_lane_filter=False,  # 是否启用车道过滤调试输出
        ))
        return config

    def __init__(self, config, agent2policy):
        self.policy = agent2policy
        self.controlled_agents = {}
        self.controlled_agent_ids = []
        self.obs_list = []
        self.round = 0
        # 调试模式配置
        self.debug_traffic_light = config.get("debug_traffic_light", False)
        self.debug_lane_filter = config.get("debug_lane_filter", False)
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
        
        # 调试：场景信息统计
        if self.debug_lane_filter or self.debug_traffic_light:
            print(f"\n📍 场景信息统计:")
            print(f"  - 总车道数: {len(self.lanes)}")
            
            # 统计红绿灯数量
            if self.debug_traffic_light:
                traffic_light_lanes = []
                for lane in self.lanes.values():
                    if self.engine.light_manager.has_traffic_light(lane.lane.index):
                        traffic_light_lanes.append(lane.lane.index)
                print(f"  - 有红绿灯的车道数: {len(traffic_light_lanes)}")
                if len(traffic_light_lanes) > 0:
                    print(f"    车道索引: {traffic_light_lanes[:5]}" + 
                          (f" ... 共{len(traffic_light_lanes)}个" if len(traffic_light_lanes) > 5 else ""))
                else:
                    print(f"    ⚠️ 场景中没有红绿灯！")
        
        # 在获取车道信息后，进行车道区域过滤
        total_cars_before = len(self.car_birth_info_list)
        valid_count, filtered_count, filtered_list = self._filter_valid_spawn_positions()
        
        # 输出过滤信息
        if filtered_count > 0:
            self.logger.warning(f"车辆生成位置过滤: 原始 {total_cars_before} 辆, "
                              f"有效 {valid_count} 辆, 过滤 {filtered_count} 辆")
            for filtered_car in filtered_list[:5]:  # 只显示前5个
                self.logger.debug(f"  - 过滤车辆 ID={filtered_car['id']}, "
                                f"位置={filtered_car['position']}, "
                                f"原因={filtered_car['reason']}")
            if filtered_count > 5:
                self.logger.debug(f"  - ... 还有 {filtered_count - 5} 辆车被过滤")
        
        # 限制最大车辆数（在过滤后应用）
        max_vehicles = self.config.get("max_controlled_vehicles", None)
        if max_vehicles is not None and len(self.car_birth_info_list) > max_vehicles:
            self.car_birth_info_list = self.car_birth_info_list[:max_vehicles]
            self.logger.info(f"限制最大车辆数为 {max_vehicles} 辆")
        
        self.logger.info(f"最终生成 {len(self.car_birth_info_list)} 辆可控车辆")

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

    def _is_position_on_lane(self, position, tolerance=None):
        """
        检测给定位置是否在有效车道范围内
        
        Args:
            position: (x, y) 车辆位置坐标
            tolerance: 容差范围（米），用于放宽检测条件。None时使用配置中的默认值
        
        Returns:
            bool: True表示在车道上，False表示在非车道区域（如草坪、停车场等）
        """
        if not hasattr(self, 'lanes') or self.lanes is None:
            if self.debug_lane_filter:
                print(f"    ⚠️ 车道信息未初始化，默认允许")
            return True  # 如果车道信息未初始化，默认允许生成
        
        if tolerance is None:
            tolerance = self.config.get("lane_tolerance", 3.0)
        
        position_2d = (position[0], position[1])
        
        if self.debug_lane_filter:
            print(f"  🔍 检测位置 ({position_2d[0]:.2f}, {position_2d[1]:.2f}), 容差={tolerance}m")
        
        # 方法1：直接检测是否在任一车道上
        checked_lanes = 0
        for lane in self.lanes.values():
            try:
                checked_lanes += 1
                if lane.lane.point_on_lane(position_2d):
                    if self.debug_lane_filter:
                        print(f"    ✅ 在车道上 (车道{lane.lane.index}, 检查了{checked_lanes}条)")
                    return True
            except:
                continue
        
        if self.debug_lane_filter:
            print(f"    ❌ 不在任何车道上 (检查了{checked_lanes}条车道)")
        
        # 方法2：如果严格检测失败，使用容差范围检测（考虑车道边缘）
        # 注释：此方法已被禁用，如需启用请取消注释
        # if tolerance > 0:
        #     for lane in self.lanes.values():
        #         try:
        #             # 计算点到车道中心线的距离
        #             lane_obj = lane.lane
        #             # 获取车道长度并检测最近点
        #             s, lateral = lane_obj.local_coordinates(position_2d)
                    
        #             # 如果横向距离在容差范围内，认为是有效的
        #             if abs(lateral) <= tolerance and 0 <= s <= lane_obj.length:
        #                 return True
        #         except:
        #             continue
        
        return False
    
    def _filter_valid_spawn_positions(self):
        """
        过滤掉生成位置不在有效车道上的车辆信息
        根据配置决定是否执行过滤
        
        Returns:
            tuple: (有效车辆数量, 被过滤车辆数量, 被过滤车辆ID列表)
        """
        # 如果配置中禁用了过滤，直接返回
        if not self.config.get("filter_offroad_vehicles", True):
            if self.debug_lane_filter:
                print(f"🚫 车道过滤已禁用")
            return len(self.car_birth_info_list), 0, []
        
        if self.debug_lane_filter:
            print(f"\n🔍 开始车道过滤: 共 {len(self.car_birth_info_list)} 辆车待检测")
        
        valid_cars = []
        filtered_cars = []
        tolerance = self.config.get("lane_tolerance", 3.0)
        
        for idx, car in enumerate(self.car_birth_info_list):
            if self.debug_lane_filter:
                print(f"\n车辆 {idx+1}/{len(self.car_birth_info_list)}: ID={car['id']}")
            
            if self._is_position_on_lane(car['begin'], tolerance=tolerance):
                valid_cars.append(car)
                if self.debug_lane_filter:
                    print(f"  ✅ 保留")
            else:
                filtered_cars.append({
                    'id': car['id'],
                    'position': car['begin'],
                    'reason': '生成位置不在有效车道上（可能在草坪/停车场等区域）'
                })
                if self.debug_lane_filter:
                    print(f"  ❌ 过滤 (原因: 不在车道上)")
        
        self.car_birth_info_list = valid_cars
        
        if self.debug_lane_filter:
            print(f"\n📊 过滤结果: 保留 {len(valid_cars)} 辆, 过滤 {len(filtered_cars)} 辆")
        
        return len(valid_cars), len(filtered_cars), filtered_cars
    
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

    def _get_traffic_light_state(self, vehicle):
        """
        获取车辆当前位置的红绿灯状态（优化版）
        
        解决问题：
        1. 部分红绿灯状态为None的问题 - 添加异常处理和默认值
        2. 车道分段导致无法获取红绿灯的问题 - 优先使用导航模块，失败时回退到遍历
        
        Returns:
            int: 0=无红绿灯, 1=绿灯, 2=黄灯, 3=红灯
        """
        traffic_light = 0
        state = vehicle.get_state()
        position_2d = state['position'][:2]
        
        if self.debug_traffic_light:
            print(f"\n🚦 检测车辆红绿灯 - 位置: ({position_2d[0]:.1f}, {position_2d[1]:.1f})")
        
        try:
            # 方法1：优先尝试从车辆导航模块获取当前车道（更高效）
            if hasattr(vehicle, 'navigation') and vehicle.navigation is not None:
                current_lane = vehicle.navigation.current_lane
                
                if self.debug_traffic_light:
                    print(f"  方法1-导航模块:")
                    print(f"    current_lane = {current_lane}")
                    print(f"    lane_index = {current_lane.index if current_lane else 'None'}")
                
                if current_lane:
                    has_light = self.engine.light_manager.has_traffic_light(current_lane.index)
                    
                    if self.debug_traffic_light:
                        print(f"    has_traffic_light = {has_light}")
                    
                    if has_light:
                        status = self.engine.light_manager._lane_index_to_obj[current_lane.index].status
                        
                        if self.debug_traffic_light:
                            print(f"    status = {status}")
                        
                        if status == 'TRAFFIC_LIGHT_GREEN':
                            if self.debug_traffic_light:
                                print(f"  ✅ 方法1成功: 绿灯")
                            return 1
                        elif status == 'TRAFFIC_LIGHT_YELLOW':
                            if self.debug_traffic_light:
                                print(f"  ✅ 方法1成功: 黄灯")
                            return 2
                        elif status == 'TRAFFIC_LIGHT_RED':
                            if self.debug_traffic_light:
                                print(f"  ✅ 方法1成功: 红灯")
                            return 3
                        elif status is None:
                            if self.debug_traffic_light:
                                print(f"  ⚠️ 方法1: 红绿灯状态为None")
                            return 0
                    else:
                        if self.debug_traffic_light:
                            print(f"    该车道没有红绿灯")
                else:
                    if self.debug_traffic_light:
                        print(f"    导航模块current_lane为None")
            else:
                if self.debug_traffic_light:
                    has_nav = hasattr(vehicle, 'navigation')
                    nav_not_none = vehicle.navigation is not None if has_nav else False
                    print(f"  方法1-导航模块: 不可用 (hasattr={has_nav}, not_none={nav_not_none})")
                    
        except Exception as e:
            if self.debug_traffic_light:
                print(f"  ❌ 方法1异常: {type(e).__name__}: {e}")
            pass
        
        try:
            # 方法2：遍历所有车道查找（兜底方案，处理车道分段问题）
            if self.debug_traffic_light:
                print(f"  方法2-遍历车道: 开始遍历 {len(self.lanes)} 条车道")
            
            found_lane = False
            checked_lanes = 0
            
            for lane in self.lanes.values():
                try:
                    checked_lanes += 1
                    if lane.lane.point_on_lane(position_2d):
                        found_lane = True
                        if self.debug_traffic_light:
                            print(f"    ✓ 找到车辆所在车道: {lane.lane.index} (检查了{checked_lanes}条)")
                        
                        has_light = self.engine.light_manager.has_traffic_light(lane.lane.index)
                        if self.debug_traffic_light:
                            print(f"    has_traffic_light = {has_light}")
                        
                        if has_light:
                            status = self.engine.light_manager._lane_index_to_obj[lane.lane.index].status
                            if self.debug_traffic_light:
                                print(f"    status = {status}")
                            
                            if status == 'TRAFFIC_LIGHT_GREEN':
                                if self.debug_traffic_light:
                                    print(f"  ✅ 方法2成功: 绿灯")
                                return 1
                            elif status == 'TRAFFIC_LIGHT_YELLOW':
                                if self.debug_traffic_light:
                                    print(f"  ✅ 方法2成功: 黄灯")
                                return 2
                            elif status == 'TRAFFIC_LIGHT_RED':
                                if self.debug_traffic_light:
                                    print(f"  ✅ 方法2成功: 红灯")
                                return 3
                            elif status is None:
                                if self.debug_traffic_light:
                                    print(f"  ⚠️ 方法2: 红绿灯状态为None")
                                return 0
                        else:
                            if self.debug_traffic_light:
                                print(f"    该车道没有红绿灯")
                        break
                except:
                    continue
            
            if self.debug_traffic_light and not found_lane:
                print(f"    ⚠️ 未找到车辆所在车道 (检查了{checked_lanes}条)")
                
        except Exception as e:
            if self.debug_traffic_light:
                print(f"  ❌ 方法2异常: {type(e).__name__}: {e}")
            pass
        
        if self.debug_traffic_light:
            print(f"  结果: 返回 {traffic_light} (无红绿灯/未知)")
        
        return traffic_light
    
    def _get_all_obs(self):
        # position, velocity, heading, lidar, navigation, TODO: trafficlight -> list
        self.obs_list = []
        for agent_id, vehicle in self.controlled_agents.items():
            state = vehicle.get_state()

            # 使用优化后的红绿灯检测方法
            traffic_light = self._get_traffic_light_state(vehicle)

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
