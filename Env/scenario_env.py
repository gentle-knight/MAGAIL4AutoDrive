"""
多智能体场景环境 (MultiAgentScenarioEnv)

==================================
配置参数说明 (写在最前面)
==================================

基础配置：
    data_directory (str): 专家数据目录路径
    num_controlled_agents (int): 默认可控智能体数量，默认3
    horizon (int): 每个回合的最大步数，默认1000

车道检测与过滤配置：
    filter_offroad_vehicles (bool): 是否过滤非车道区域的车辆，默认True
        - True: 过滤掉在草坪、停车场等非车道区域生成的车辆
        - False: 保留所有车辆
    lane_tolerance (float): 车道检测容差（米），默认3.0
        - 用于放宽车道检测的边界条件
    max_controlled_vehicles (int|None): 最大可控车辆数限制，默认None
        - None: 不限制车辆数量
        - int: 限制最多生成的车辆数

场景对象配置：
    no_traffic_lights (bool): 是否禁用红绿灯渲染和逻辑，默认False
        - True: 完全移除场景中的红绿灯
        - False: 保留红绿灯（按数据集原样生成）

专家数据继承配置：
    inherit_expert_velocity (bool): 是否继承专家数据中车辆的初始速度，默认False
        - True: 车辆生成时使用专家数据中的速度
        - False: 车辆生成时速度为0（由策略控制）

调试模式配置：
    debug_lane_filter (bool): 车道过滤详细调试输出，默认False
        - 输出每个车辆位置的车道检测详细过程
    verbose_reset (bool): 重置时输出详细统计信息，默认False
        - 输出场景统计、过滤详情等

使用示例：
    env = MultiAgentScenarioEnv(
        config={
            "data_directory": "path/to/data",
            "max_controlled_vehicles": 10,
            "inherit_expert_velocity": True,  # 继承专家速度
            "no_traffic_lights": True,         # 禁用红绿灯
            "verbose_reset": True,             # 详细输出
        },
        agent2policy=your_policy
    )

==================================
整体逻辑和处理流程
==================================

1. 初始化阶段：
   - 继承MetaDrive的ScenarioEnv基类
   - 配置多智能体参数（车辆数量、调试模式等）
   - 接收策略映射字典(agent2policy)

2. 环境重置阶段 (reset方法)：
   - 解析专家数据，提取车辆生成信息(car_birth_info_list)
   - 清理原始交通数据，只保留车辆位置、朝向、目的地、速度（可选）
   - 禁用红绿灯（如果配置）
   - 初始化地图和车道信息
   - 执行车道过滤(_filter_valid_spawn_positions)，移除非车道区域的车辆
   - 限制最大车辆数量
   - 生成可控智能体(_spawn_controlled_agents)

3. 观测获取阶段 (_get_all_obs方法)：
   - 遍历所有可控车辆
   - 获取车辆状态（位置、速度、朝向）
   - 检测红绿灯状态(_get_traffic_light_state)
   - 获取激光雷达数据（前向、侧向、车道线检测）
   - 组装完整观测向量

4. 环境步进阶段 (step方法)：
   - 执行所有智能体的动作
   - 更新物理引擎状态
   - 生成新的智能体（按时间步）
   - 返回新的观测、奖励、完成状态

核心功能模块：
- PolicyVehicle: 可控制策略的车辆类
- 车道检测与过滤: 确保车辆只在有效车道上生成
- 多智能体管理: 动态生成和管理可控车辆
- 专家速度继承: 可选地继承专家数据中的初始速度
"""

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
    """
    可控制策略的车辆类
    
    继承自MetaDrive的DefaultVehicle，增加了策略控制和目标设置功能。
    用于多智能体环境中的可控车辆，支持自定义策略和目的地。
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化策略车辆
        
        Args:
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.policy = None  # 车辆的控制策略
        self.destination = None  # 车辆的目标目的地

    def set_policy(self, policy):
        """
        设置车辆的控制策略
        
        Args:
            policy: 控制策略对象，必须实现act(observation)方法
        """
        self.policy = policy

    def set_destination(self, des):
        """
        设置车辆的目标目的地
        
        Args:
            des: 目标位置坐标 (x, y)
        """
        self.destination = des

    def act(self, observation, policy=None):
        """
        根据观测获取动作
        
        Args:
            observation: 环境观测数据
            policy: 可选的外部策略，如果提供则优先使用
            
        Returns:
            动作向量，如果无策略则返回随机动作
        """
        if self.policy is not None:
            return self.policy.act(observation)
        else:
            return self.action_space.sample()

    def before_step(self, action):
        """
        执行动作前的状态记录
        
        在每步执行前记录当前状态，用于后续的状态追踪和分析。
        
        Args:
            action: 即将执行的动作
        """
        self.last_position = self.position  # 记录当前位置 (2D向量)
        self.last_velocity = self.velocity  # 记录当前速度 (2D向量)
        self.last_speed = self.speed  # 记录当前速度大小 (标量)
        self.last_heading_dir = self.heading  # 记录当前朝向
        if action is not None:
            self.last_current_action.append(action)  # 记录动作历史
        self._set_action(action)  # 设置动作到车辆

    def is_done(self):
        """
        检查车辆是否完成任务
        
        目前为空实现，可根据需要添加到达目的地或碰撞检测逻辑
        
        Returns:
            bool: True表示任务完成，False表示继续执行
        """
        # arrive or crash
        pass


# 将PolicyVehicle注册为默认车辆类型
vehicle_class_to_type[PolicyVehicle] = "default"


class MultiAgentScenarioEnv(ScenarioEnv):
    """
    多智能体场景环境
    
    基于MetaDrive的ScenarioEnv扩展，支持多智能体强化学习训练。
    主要功能包括：
    1. 从专家数据中提取车辆信息并生成可控智能体
    2. 车道检测与过滤，确保车辆在有效区域生成
    3. 红绿灯状态检测，为智能体提供交通信号信息
    4. 多智能体观测、动作和奖励管理
    """
    
    @classmethod
    def default_config(cls):
        """
        获取环境的默认配置
        
        继承父类配置并添加多智能体相关的配置参数
        
        配置参数说明：
        - data_directory: 专家数据目录路径
        - num_controlled_agents: 默认可控智能体数量
        - horizon: 每个回合的最大步数
        
        车道检测与过滤配置：
        - filter_offroad_vehicles: 是否过滤非车道区域的车辆
        - lane_tolerance: 车道检测容差（米）
        - max_controlled_vehicles: 最大可控车辆数限制（None表示不限制）
        
        场景对象配置：
        - no_traffic_lights: 禁用红绿灯渲染和逻辑
        
        专家数据继承配置：
        - inherit_expert_velocity: 是否继承专家数据中车辆的初始速度（默认False）
        
        调试模式配置：
        - debug_traffic_light: 红绿灯检测详细调试输出
        - debug_lane_filter: 车道过滤详细调试输出
        - verbose_reset: 重置时输出详细统计信息
        
        Returns:
            dict: 包含所有配置参数的字典
        """
        config = super().default_config()
        config.update(dict(
            # 基础配置
            data_directory=None,
            num_controlled_agents=3,
            horizon=1000,
            
            # 车道检测与过滤配置
            filter_offroad_vehicles=True,
            lane_tolerance=3.0,
            max_controlled_vehicles=None,
            
            # 场景对象配置
            no_traffic_lights=False,
            
            # 专家数据继承配置
            inherit_expert_velocity=False,
            
            # 调试模式配置
            debug_lane_filter=False,
            verbose_reset=False,
        ))
        return config

    def __init__(self, config, agent2policy):
        """
        初始化多智能体场景环境
        
        Args:
            config: 环境配置字典，包含各种参数设置
            agent2policy: 智能体ID到策略的映射字典
        """
        self.policy = agent2policy  # 智能体策略映射
        self.controlled_agents = {}  # 可控智能体字典 {agent_id: vehicle}
        self.controlled_agent_ids = []  # 可控智能体ID列表
        self.obs_list = []  # 观测数据列表
        self.round = 0  # 当前时间步
        
        # 调试模式配置
        self.debug_lane_filter = config.get("debug_lane_filter", False)
        
        # 调用父类初始化
        super().__init__(config)

    def reset(self, seed: Union[None, int] = None):
        """
        重置环境到初始状态
        
        这是环境的核心重置方法，执行以下步骤：
        1. 初始化日志系统
        2. 解析专家数据，提取车辆生成信息
        3. 清理原始交通数据
        4. 初始化地图和车道信息
        5. 执行车道过滤和车辆数量限制
        6. 生成可控智能体
        7. 返回初始观测
        
        Args:
            seed: 随机种子，用于环境重置时的随机性控制
            
        Returns:
            list: 所有智能体的初始观测数据
        """
        self.round = 0  # 重置时间步计数器
        
        # 初始化日志系统
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)

        # 延迟初始化MetaDrive引擎
        self.lazy_init()
        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError("Broken MetaDrive instance.")
        
        # 在场景加载前禁用红绿灯（如果配置中启用了no_traffic_lights选项）
        if self.config.get("no_traffic_lights", False):
            # 重写红绿灯管理器的方法，阻止创建和使用红绿灯
            if hasattr(self.engine, 'light_manager') and self.engine.light_manager is not None:
                self.engine.light_manager.before_reset = lambda *args, **kwargs: None
                self.engine.light_manager.after_reset = lambda *args, **kwargs: None
                self.engine.light_manager.before_step = lambda *args, **kwargs: None
                self.engine.light_manager.get_traffic_light = lambda *args, **kwargs: None
                self.logger.info("已禁用红绿灯管理器")

        # 步骤1：解析专家数据，提取车辆生成信息
        # 记录专家数据中每辆车的位置，接着全部清除，只保留位置等信息，用于后续生成
        _obj_to_clean_this_frame = []  # 需要清理的对象ID列表
        self.car_birth_info_list = []  # 车辆生成信息列表
        self.expert_trajectories = {}  # 专家数据轨迹字典
        
        for scenario_id, track in self.engine.traffic_manager.current_traffic_data.items():
            # 跳过自车（SDC - Self Driving Car）
            if scenario_id == self.engine.traffic_manager.sdc_scenario_id:
                continue
            
            # 只处理车辆类型的对象
            if track["type"] == MetaDriveType.VEHICLE:
                _obj_to_clean_this_frame.append(scenario_id)
                valid = track['state']['valid']  # 车辆有效性标记
                # 找到车辆首次出现和最后出现的时间步
                first_show = np.argmax(valid) if valid.any() else -1
                last_show = len(valid) - 1 - np.argmax(valid[::-1]) if valid.any() else -1
                
                if first_show == -1 or last_show == -1:
                    continue            
                object_id = track["metadata"]["object_id"]
                
                # 提取完整轨迹数据(只使用确认存在的字段)
                trajectory_data = {
                    "object_id": object_id,
                    "scenario_id": scenario_id,
                    "valid_mask": valid[first_show:last_show+1].copy(),  # 有效性掩码
                    "positions": track["state"]["position"][first_show:last_show+1].copy(),  # (T, 3)
                    "headings": track["state"]["heading"][first_show:last_show+1].copy(),    # (T,)
                    "velocities": track["state"]["velocity"][first_show:last_show+1].copy(), # (T, 2)
                    "timesteps": np.arange(first_show, last_show+1),  # 时间戳
                    "start_timestep": first_show,
                    "end_timestep": last_show,
                    "length": last_show - first_show + 1
                }

                # 可选:如果数据中有车辆尺寸信息,则添加
                # 方法1: 尝试从state中获取
                if "length" in track["state"]:
                    trajectory_data["vehicle_length"] = track["state"]["length"][first_show]
                if "width" in track["state"]:
                    trajectory_data["vehicle_width"] = track["state"]["width"][first_show]
                if "height" in track["state"]:
                    trajectory_data["vehicle_height"] = track["state"]["height"][first_show]

                # 方法2: 尝试从metadata中获取
                if "vehicle_length" not in trajectory_data and "length" in track.get("metadata", {}):
                    trajectory_data["vehicle_length"] = track["metadata"]["length"]
                if "vehicle_width" not in trajectory_data and "width" in track.get("metadata", {}):
                    trajectory_data["vehicle_width"] = track["metadata"]["width"]
                if "vehicle_height" not in trajectory_data and "height" in track.get("metadata", {}):
                    trajectory_data["vehicle_height"] = track["metadata"]["height"]

                # 方法3: 使用默认值(如果以上都没有)
                if "vehicle_length" not in trajectory_data:
                    trajectory_data["vehicle_length"] = 4.5  # MetaDrive默认车长
                if "vehicle_width" not in trajectory_data:
                    trajectory_data["vehicle_width"] = 2.0   # MetaDrive默认车宽
                if "vehicle_height" not in trajectory_data:
                    trajectory_data["vehicle_height"] = 1.5  # MetaDrive默认车高

                                
                # 存储到专家轨迹字典
                self.expert_trajectories[object_id] = trajectory_data

                # 提取车辆关键信息
                car_info = {
                    'id': track['metadata']['object_id'],
                    'show_time': first_show,
                    'begin': (track['state']['position'][first_show, 0], track['state']['position'][first_show, 1]),
                    'heading': track['state']['heading'][first_show],
                    'end': (track['state']['position'][last_show, 0], track['state']['position'][last_show, 1])
                }
                
                # 如果配置要求继承专家速度，则提取初始速度
                if self.config.get("inherit_expert_velocity", False):
                    velocity = track['state']['velocity'][first_show]
                    car_info['velocity'] = (velocity[0], velocity[1])
                
                self.car_birth_info_list.append(car_info)
            # 非车辆对象（如红绿灯、行人等）保留，不清理

        # 清理车辆原始交通数据，释放内存（保留红绿灯等其他对象）
        for scenario_id in _obj_to_clean_this_frame:
            self.engine.traffic_manager.current_traffic_data.pop(scenario_id)

        # 步骤2：重置MetaDrive引擎和传感器
        self.engine.reset()
        self.reset_sensors()
        self.engine.taskMgr.step()

        # 步骤3：获取地图车道信息
        self.lanes = self.engine.map_manager.current_map.road_network.graph
        
        # 调试：场景信息统计（仅在verbose_reset模式下输出）
        if self.config.get("verbose_reset", False):
            print(f"\n📍 场景信息统计:")
            print(f"  - 总车道数: {len(self.lanes)}")
            
            # 统计红绿灯数量（如果未禁用红绿灯）
            if not self.config.get("no_traffic_lights", False):
                traffic_light_lanes = []
                for lane in self.lanes.values():
                    if self.engine.light_manager.has_traffic_light(lane.lane.index):
                        traffic_light_lanes.append(lane.lane.index)
                print(f"  - 有红绿灯的车道数: {len(traffic_light_lanes)}")
                if len(traffic_light_lanes) > 5:
                    print(f"    车道索引示例: {traffic_light_lanes[:5]} ...")
            else:
                print(f"  - 红绿灯: 已禁用")
        
        # 步骤4：执行车道区域过滤
        total_cars_before = len(self.car_birth_info_list)
        valid_count, filtered_count, filtered_list = self._filter_valid_spawn_positions()
        
        # 输出过滤信息（仅在有过滤时输出）
        if filtered_count > 0:
            if self.config.get("verbose_reset", False):
                self.logger.warning(f"车辆生成位置过滤: 原始 {total_cars_before} 辆, "
                                  f"有效 {valid_count} 辆, 过滤 {filtered_count} 辆")
                for filtered_car in filtered_list[:3]:
                    self.logger.debug(f"  过滤车辆 ID={filtered_car['id']}, "
                                    f"位置={filtered_car['position']}, "
                                    f"原因={filtered_car['reason']}")
                if filtered_count > 3:
                    self.logger.debug(f"  ... 还有 {filtered_count - 3} 辆车被过滤")
            else:
                self.logger.info(f"车辆过滤: {total_cars_before} 辆 -> {valid_count} 辆 (过滤 {filtered_count} 辆)")
        
        # 步骤5：限制最大车辆数（在过滤后应用）
        max_vehicles = self.config.get("max_controlled_vehicles", None)
        if max_vehicles is not None and len(self.car_birth_info_list) > max_vehicles:
            original_count = len(self.car_birth_info_list)
            self.car_birth_info_list = self.car_birth_info_list[:max_vehicles]
            if self.config.get("verbose_reset", False):
                self.logger.info(f"限制最大车辆数: {original_count} 辆 -> {max_vehicles} 辆")
        
        # 最终统计
        if self.config.get("verbose_reset", False):
            self.logger.info(f"✓ 最终生成 {len(self.car_birth_info_list)} 辆可控车辆")

        # 清理渲染器
        if self.top_down_renderer is not None:
            self.top_down_renderer.clear()
            self.engine.top_down_renderer = None

        # 初始化回合相关变量
        self.dones = {}  # 智能体完成状态
        self.episode_rewards = defaultdict(float)  # 回合奖励累积
        self.episode_lengths = defaultdict(int)  # 回合长度累积

        # 清空可控智能体
        self.controlled_agents.clear()
        self.controlled_agent_ids.clear()

        # 步骤6：调用父类重置并生成可控智能体
        super().reset(seed)  # 初始化场景
        self._spawn_controlled_agents()

        # 步骤7：返回初始观测
        return self._get_all_obs()

    def _is_position_on_lane(self, position, tolerance=None):
        """
        检测给定位置是否在有效车道范围内
        
        这个函数用于验证车辆生成位置是否在合法的车道上，避免在草坪、停车场等
        非车道区域生成车辆。支持两种检测方法：
        1. 严格检测：直接使用MetaDrive的point_on_lane方法
        2. 容差检测：考虑车道边缘的容差范围（当前已禁用）
        
        Args:
            position: (x, y) 车辆位置坐标
            tolerance: 容差范围（米），用于放宽检测条件。None时使用配置中的默认值
        
        Returns:
            bool: True表示在车道上，False表示在非车道区域（如草坪、停车场等）
        """
        # 检查车道信息是否已初始化
        if not hasattr(self, 'lanes') or self.lanes is None:
            if self.debug_lane_filter:
                print(f"    ⚠️ 车道信息未初始化，默认允许")
            return True  # 如果车道信息未初始化，默认允许生成
        
        # 设置容差参数
        if tolerance is None:
            tolerance = self.config.get("lane_tolerance", 3.0)
        
        position_2d = (position[0], position[1])
        
        if self.debug_lane_filter:
            print(f"  🔍 检测位置 ({position_2d[0]:.2f}, {position_2d[1]:.2f}), 容差={tolerance}m")
        
        # 方法1：直接检测是否在任一车道上
        # 遍历所有车道，使用MetaDrive的point_on_lane方法进行精确检测
        checked_lanes = 0
        for lane in self.lanes.values():
            try:
                checked_lanes += 1
                if lane.lane.point_on_lane(position_2d):
                    if self.debug_lane_filter:
                        print(f"    ✅ 在车道上 (车道{lane.lane.index}, 检查了{checked_lanes}条)")
                    return True
            except:
                # 如果检测过程中出现异常，继续检查下一条车道
                continue
        
        if self.debug_lane_filter:
            print(f"    ❌ 不在任何车道上 (检查了{checked_lanes}条车道)")
        
        # 方法2：如果严格检测失败，使用容差范围检测（考虑车道边缘）
        # 注释：此方法已被禁用，如需启用请取消注释
        # 该方法通过计算点到车道中心线的横向距离来判断是否在容差范围内
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
        
        这个函数是车道过滤的核心实现，用于确保所有生成的车辆都在合法的车道上。
        它会遍历所有车辆生成信息，使用_is_position_on_lane方法检测每个位置，
        过滤掉在草坪、停车场等非车道区域的车辆。
        
        过滤过程包括：
        1. 检查配置是否启用过滤
        2. 遍历所有车辆生成信息
        3. 对每个车辆位置进行车道检测
        4. 分离有效和无效的车辆
        5. 更新车辆生成列表
        
        Returns:
            tuple: (有效车辆数量, 被过滤车辆数量, 被过滤车辆详细信息列表)
        """
        # 检查配置是否启用车道过滤
        if not self.config.get("filter_offroad_vehicles", True):
            if self.debug_lane_filter:
                print(f"🚫 车道过滤已禁用")
            return len(self.car_birth_info_list), 0, []
        
        if self.debug_lane_filter:
            print(f"\n🔍 开始车道过滤: 共 {len(self.car_birth_info_list)} 辆车待检测")
        
        # 初始化过滤结果
        valid_cars = []  # 有效车辆列表
        filtered_cars = []  # 被过滤车辆列表
        tolerance = self.config.get("lane_tolerance", 3.0)
        
        # 遍历所有车辆生成信息进行检测
        for idx, car in enumerate(self.car_birth_info_list):
            if self.debug_lane_filter:
                print(f"\n车辆 {idx+1}/{len(self.car_birth_info_list)}: ID={car['id']}")
            
            # 检测车辆生成位置是否在有效车道上
            if self._is_position_on_lane(car['begin'], tolerance=tolerance):
                valid_cars.append(car)
                if self.debug_lane_filter:
                    print(f"  ✅ 保留")
            else:
                # 记录被过滤的车辆信息
                filtered_cars.append({
                    'id': car['id'],
                    'position': car['begin'],
                    'reason': '生成位置不在有效车道上（可能在草坪/停车场等区域）'
                })
                if self.debug_lane_filter:
                    print(f"  ❌ 过滤 (原因: 不在车道上)")
        
        # 更新车辆生成列表为过滤后的结果
        self.car_birth_info_list = valid_cars
        
        if self.debug_lane_filter:
            print(f"\n📊 过滤结果: 保留 {len(valid_cars)} 辆, 过滤 {len(filtered_cars)} 辆")
        
        return len(valid_cars), len(filtered_cars), filtered_cars
    
    def _spawn_controlled_agents(self):
        """
        生成可控智能体车辆
        
        根据当前时间步和车辆生成信息，动态生成需要出现的可控车辆。
        每个车辆都会被分配策略和目标目的地，并注册到MetaDrive引擎中
        参与物理仿真。
        
        生成过程：
        1. 遍历所有车辆生成信息
        2. 检查车辆是否应该在当前时间步出现
        3. 创建PolicyVehicle实例
        4. 设置车辆策略和目的地
        5. 注册到环境管理和引擎中
        """
        # 注释：可以获取自车位置用于相对位置计算（当前未使用）
        # ego_vehicle = self.engine.agent_manager.active_agents.get("default_agent")
        # ego_position = ego_vehicle.position if ego_vehicle else np.array([0, 0])
        
        # 遍历所有车辆生成信息
        for car in self.car_birth_info_list:
            # 检查车辆是否应该在当前时间步出现
            if car['show_time'] == self.round:
                # 生成智能体ID
                agent_id = f"controlled_{car['id']}"

                # 在MetaDrive引擎中生成车辆对象
                vehicle = self.engine.spawn_object(
                    PolicyVehicle,  # 使用自定义的策略车辆类
                    vehicle_config={},  # 车辆配置（使用默认）
                    position=car['begin'],  # 车辆生成位置
                    heading=car['heading']  # 车辆生成朝向
                )
                # 重置车辆状态到指定位置和朝向
                vehicle.reset(position=car['begin'], heading=car['heading'])
                
                # 如果配置要求继承专家速度，则设置初始速度
                if 'velocity' in car and self.config.get("inherit_expert_velocity", False):
                    vehicle.set_velocity(car['velocity'])

                # 设置车辆的控制策略和目标
                vehicle.set_policy(self.policy)  # 设置策略
                vehicle.set_destination(car['end'])  # 设置目的地

                # 注册到环境管理
                self.controlled_agents[agent_id] = vehicle
                self.controlled_agent_ids.append(agent_id)

                # ✅ 关键：注册到引擎的 active_agents，才能参与物理更新
                # 这是MetaDrive引擎识别和管理智能体的关键步骤
                self.engine.agent_manager.active_agents[agent_id] = vehicle

    def _get_all_obs(self):
        """
        获取所有可控智能体的观测数据
        
        这是环境的核心观测函数，为每个可控智能体组装完整的观测向量。
        观测数据包括：
        1. 车辆状态信息：位置、速度、朝向
        2. 传感器数据：激光雷达（前向、侧向、车道线检测）
        3. 导航信息：目标目的地
        
        观测向量结构：
        - position[2]: 车辆位置 (x, y)
        - velocity[2]: 车辆速度 (vx, vy)  
        - heading[1]: 车辆朝向角度
        - lidar[80]: 前向激光雷达数据 (80个激光束，30米范围)
        - side_lidar[10]: 侧向激光雷达数据 (10个激光束，8米范围)
        - lane_line_lidar[10]: 车道线检测数据 (10个激光束，3米范围)
        - destination[2]: 目标目的地 (x, y)
        
        Returns:
            list: 所有智能体的观测数据列表
        """
        self.obs_list = []  # 清空观测列表
        
        # 遍历所有可控智能体
        for agent_id, vehicle in self.controlled_agents.items():
            # 获取车辆基本状态信息
            state = vehicle.get_state()

            # 获取激光雷达传感器数据
            # 前向激光雷达：80个激光束，30米检测距离，用于障碍物检测
            lidar = self.engine.get_sensor("lidar").perceive(
                num_lasers=80, 
                distance=30, 
                base_vehicle=vehicle,
                physics_world=self.engine.physics_world.dynamic_world
            )
            
            # 侧向激光雷达：10个激光束，8米检测距离，用于侧向障碍物检测
            side_lidar = self.engine.get_sensor("side_detector").perceive(
                num_lasers=10, 
                distance=8,
                base_vehicle=vehicle,
                physics_world=self.engine.physics_world.static_world
            )
            
            # 车道线检测激光雷达：10个激光束，3米检测距离，用于车道线识别
            lane_line_lidar = self.engine.get_sensor("lane_line_detector").perceive(
                num_lasers=10, 
                distance=3,
                base_vehicle=vehicle,
                physics_world=self.engine.physics_world.static_world
            )

            # 组装完整的观测向量
            obs = (state['position'][:2] +  # 位置 (x, y)
                   list(state['velocity']) +  # 速度 (vx, vy)
                   [state['heading_theta']] +  # 朝向角度
                   lidar[0] +  # 前向激光雷达数据
                   side_lidar[0] +  # 侧向激光雷达数据
                   lane_line_lidar[0] +  # 车道线检测数据
                   list(vehicle.destination))  # 目标目的地 (x, y)
            
            self.obs_list.append(obs)
        
        return self.obs_list

    def step(self, action_dict: Dict[AnyStr, Union[list, np.ndarray]]):
        """
        执行环境的一个时间步
        
        这是环境的核心步进函数，执行以下操作序列：
        1. 更新时间步计数器
        2. 执行所有智能体的动作（before_step）
        3. 更新MetaDrive物理引擎状态
        4. 执行智能体动作后的处理（after_step）
        5. 生成新的智能体（按时间步）
        6. 获取新的观测数据
        7. 计算奖励和完成状态
        8. 返回环境状态
        
        Args:
            action_dict: 智能体动作字典 {agent_id: action}
            
        Returns:
            tuple: (观测数据, 奖励字典, 完成状态字典, 信息字典)
                - obs: 所有智能体的观测数据列表
                - rewards: 每个智能体的奖励 {agent_id: reward}
                - dones: 每个智能体的完成状态 {agent_id: done, "__all__": episode_done}
                - infos: 每个智能体的额外信息 {agent_id: info}
        """
        # 步骤1：更新时间步计数器
        self.round += 1

        # 步骤2：执行所有智能体的动作（动作执行前处理）
        for agent_id, action in action_dict.items():
            if agent_id in self.controlled_agents:
                # 记录车辆状态并设置动作
                self.controlled_agents[agent_id].before_step(action)

        # 步骤3：更新MetaDrive物理引擎状态
        # 这是核心的物理仿真步骤，所有车辆状态都会根据动作更新
        self.engine.step()

        # 步骤4：执行智能体动作后的处理
        for agent_id in action_dict:
            if agent_id in self.controlled_agents:
                # 执行动作后的状态更新（如果有after_step方法）
                self.controlled_agents[agent_id].after_step()

        # 步骤5：生成新的智能体（按时间步动态生成）
        self._spawn_controlled_agents()
        
        # 步骤6：获取新的观测数据
        obs = self._get_all_obs()
        
        # 步骤7：计算奖励和完成状态
        # 初始化所有智能体的奖励为0（可根据需要实现奖励计算逻辑）
        rewards = {aid: 0.0 for aid in self.controlled_agents}
        
        # 初始化所有智能体的完成状态为False（可根据需要实现完成条件）
        dones = {aid: False for aid in self.controlled_agents}
        
        # 检查整个回合是否结束（达到最大步数）
        dones["__all__"] = self.episode_step >= self.config["horizon"]
        
        # 初始化所有智能体的额外信息为空字典
        infos = {aid: {} for aid in self.controlled_agents}
        
        # 步骤8：返回环境状态
        return obs, rewards, dones, infos
