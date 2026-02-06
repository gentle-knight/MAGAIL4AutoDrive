import logging
import numpy as np
from collections import defaultdict
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.type import MetaDriveType
from Env.scenario_env import MultiAgentScenarioEnv, PolicyVehicle
from Env.inverse_dynamics import InverseDynamics

class ExpertReplayEnv(MultiAgentScenarioEnv):
    def __init__(self, config=None):
        # Allow passing config without agent2policy since we don't use policies for replay
        if config is None:
            config = {}
        # Ensure we don't simulate physics for the controlled agents in the traditional sense
        # but we still need the engine to run
        super().__init__(config, agent2policy={}) 
        self.inverse_dynamics = InverseDynamics()
        self.expert_tracks = {}
        # Replay SDC/ego ("default_agent" in MetaDrive) as well; otherwise it will keep default action=0 and look stuck.
        self.replay_sdc = self.config.get("replay_sdc", True)
        self.sdc_track = None
        self.sdc_vehicle = None
        self.sdc_agent_id = "default_agent"
        
    def reset(self, seed=None):
        self.round = 0
        if self.logger is None:
            from metadrive.engine.logger import get_logger, set_log_level
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.INFO)
            set_log_level(log_level)

        self.lazy_init()
        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError("Broken MetaDrive instance.")
            
        self.background_vehicles = {}
        self.expert_tracks = {}
        self.sdc_track = None
        self.sdc_vehicle = None

        # 在加载新场景前，必须清除上一轮通过 spawn_object 生成的物体，否则 engine.reset() 内 _object_clean_check 会报错
        # 从 engine 当前对象中按名称筛选（与 manager 无关的对象需在此清理），并强制销毁
        ids_to_clear = []
        for oid, obj in self.engine.get_objects().items():
            name = getattr(obj, "name", None) or getattr(obj, "id", None)
            if name and (str(name).startswith("controlled_") or str(name).startswith("bg_")):
                ids_to_clear.append(oid)
        if ids_to_clear:
            self.engine.clear_objects(ids_to_clear, force_destroy=True)
        self.controlled_agents.clear()
        self.controlled_agent_ids.clear()
        for aid in list(self.engine.agent_manager.active_agents.keys()):
            if aid.startswith("bg_") or aid.startswith("controlled_"):
                self.engine.agent_manager.active_agents.pop(aid, None)

        if self.replay_sdc and hasattr(self.engine, "traffic_manager"):
            sdc_sid = self.engine.traffic_manager.sdc_scenario_id
            self.sdc_track = self.engine.traffic_manager.current_traffic_data.get(sdc_sid, None)

        from Env.utils import filter_traffic_tracks_to_birth_lists
        traffic_data = self.engine.traffic_manager.current_traffic_data
        car_birth_info_list, self.background_vehicles, obj_to_clean = filter_traffic_tracks_to_birth_lists(
            traffic_data,
            self.engine.traffic_manager.sdc_scenario_id,
            self.engine.map_manager,
        )
        for entry in car_birth_info_list:
            sid = entry["scenario_id"]
            if sid in traffic_data:
                self.expert_tracks[sid] = traffic_data[sid]
        self.car_birth_info_list = car_birth_info_list
        for scenario_id in obj_to_clean:
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
        
        # We skip calling super().reset() to avoid double reset
        # But we need to ensure ScenarioEnv-specific setup is done if any.
        # ScenarioEnv.reset() basically does engine.reset() and some cleanup.
        # We covered most of it.
        
        self._spawn_controlled_agents()
        self._spawn_all_background_vehicles_at_init()

        # Ensure SDC/ego is moved to the correct initial expert state.
        if self.replay_sdc:
            self.sdc_vehicle = self.engine.agent_manager.active_agents.get(self.sdc_agent_id, None)
            if self.sdc_vehicle is not None and self.sdc_track is not None:
                valid = self.sdc_track["state"]["valid"]
                t0 = int(np.argmax(valid)) if valid.any() else 0
                pos0 = self.sdc_track["state"]["position"][t0]
                heading0 = self.sdc_track["state"]["heading"][t0]
                vel0 = self.sdc_track["state"]["velocity"][t0]
                self.sdc_vehicle.set_position(pos0)
                self.sdc_vehicle.set_heading_theta(heading0)
                self.sdc_vehicle.set_velocity(vel0)
        
        return self._get_all_obs()

    def _spawn_all_background_vehicles_at_init(self):
        """Spawn all static background vehicles once at reset (no show_time filter; no removal by valid)."""
        for sid, car in self.background_vehicles.items():
            bg_id = f"bg_{car['id']}"
            if bg_id in self.engine.agent_manager.active_agents:
                continue
            vehicle_config = {}
            if 'length' in car and 'width' in car:
                vehicle_config = {
                    "length": car['length'],
                    "width": car['width']
                }
            v = self.engine.spawn_object(
                DefaultVehicle,
                name=bg_id,
                vehicle_config=vehicle_config,
                position=car['begin'],
                heading=car['heading']
            )
            v.set_velocity([0, 0])
            self.engine.agent_manager.active_agents[bg_id] = v
            v.valid_mask = car.get('valid')
            v.start_t = car['show_time']

    def _update_background_vehicles(self):
        # Static vehicles are spawned once at init and never removed (no spawn/remove by show_time or valid).
        pass

    def _spawn_controlled_agents(self):
        for car in self.car_birth_info_list:
            if car['show_time'] == self.round:
                agent_id = f"controlled_{car['id']}"
                
                # Check if we already have this agent (shouldn't happen with unique IDs but safety check)
                if agent_id in self.controlled_agents:
                    continue
                
                # Handling ID flickering / merging
                # If this ID is new, check if there's an existing agent very close to its start position
                # that just disappeared? (Not implemented here, complex logic)
                # But we can check if there's an overlap with existing agents?
                # For now, just spawn.
                
                # Read vehicle type/size if available
                vehicle_config = {}
                if 'length' in car and 'width' in car:
                     vehicle_config = {
                         "length": car['length'],
                         "width": car['width']
                     }

                vehicle = self.engine.spawn_object(
                    PolicyVehicle,
                    name=agent_id,
                    vehicle_config=vehicle_config,
                    position=car['begin'],
                    heading=car['heading']
                )
                vehicle.reset(position=car['begin'], heading=car['heading'])
                
                # We don't set policy or destination in the same way, or maybe we do for compatibility
                vehicle.set_destination(car['end'])
                
                # Store extra info for replay
                vehicle.expert_track = self.expert_tracks[car['scenario_id']]
                vehicle.original_id = car['id']
                
                self.controlled_agents[agent_id] = vehicle
                self.controlled_agent_ids.append(agent_id)
                self.engine.agent_manager.active_agents[agent_id] = vehicle

    def step(self, action_dict=None):
        # We ignore input action_dict for the purpose of controlling agents
        # Instead, we calculate what the action *should* be
        
        self.round += 1
        expert_actions = {}
        
        # 1. Update state of all controlled agents to the current timestep (self.round)
        #    and compute action from (self.round-1) to (self.round).
        #    Wait, usually step() moves T -> T+1.
        #    Current state is T. We want to move to T+1.
        #    So we need state at T and T+1.
        
        # Identify agents that are done (valid=0 at T+1 or T+1 >= length)
        agents_to_remove = []

        # Update SDC/ego first (otherwise it will stay still with default action=0)
        if self.replay_sdc and self.sdc_vehicle is not None and self.sdc_track is not None:
            next_step = self.round
            curr_step = self.round - 1
            if next_step < len(self.sdc_track["state"]["position"]) and self.sdc_track["state"]["valid"][next_step]:
                curr_state = {
                    "position": self.sdc_track["state"]["position"][curr_step],
                    "heading": self.sdc_track["state"]["heading"][curr_step],
                    "velocity": self.sdc_track["state"]["velocity"][curr_step],
                }
                next_state = {
                    "position": self.sdc_track["state"]["position"][next_step],
                    "heading": self.sdc_track["state"]["heading"][next_step],
                    "velocity": self.sdc_track["state"]["velocity"][next_step],
                }
                action, _ = self.inverse_dynamics.compute_action(curr_state, next_state, dt=0.1)
                expert_actions[self.sdc_agent_id] = action
                self.sdc_vehicle.set_position(next_state["position"])
                self.sdc_vehicle.set_heading_theta(next_state["heading"])
                self.sdc_vehicle.set_velocity(next_state["velocity"])
                self.sdc_vehicle.last_expert_action = action

        for agent_id, vehicle in self.controlled_agents.items():
            track = vehicle.expert_track
            # current_step = self.round - 1 # Since we incremented at start
            # But vehicle is currently at state corresponding to self.round - 1.
            # We want to move it to self.round.
            
            # Check bounds
            next_step = self.round
            curr_step = self.round - 1
            
            if next_step >= len(track['state']['position']):
                agents_to_remove.append(agent_id)
                continue
                
            valid = track['state']['valid'][next_step]
            if not valid:
                agents_to_remove.append(agent_id)
                continue
            
            # Get states
            curr_pos = track['state']['position'][curr_step]
            next_pos = track['state']['position'][next_step]
            curr_heading = track['state']['heading'][curr_step]
            next_heading = track['state']['heading'][next_step]
            curr_vel = track['state']['velocity'][curr_step]
            next_vel = track['state']['velocity'][next_step]
            
            # Prepare state dicts for Inverse Dynamics
            curr_state = {
                'position': curr_pos,
                'heading': curr_heading,
                'velocity': curr_vel
            }
            next_state = {
                'position': next_pos,
                'heading': next_heading,
                'velocity': next_vel
            }
            
            # Calculate action
            action, raw_info = self.inverse_dynamics.compute_action(curr_state, next_state, dt=0.1) # Waymo is 10Hz?
            expert_actions[agent_id] = action
            
            # Force update vehicle state
            vehicle.set_position(next_pos)
            vehicle.set_heading_theta(next_heading)
            vehicle.set_velocity(next_vel)
            
            # Also record this action in the vehicle for later retrieval if needed
            vehicle.last_expert_action = action
            
        # Remove finished agents
        for agent_id in agents_to_remove:
            vehicle = self.controlled_agents[agent_id]
            self.controlled_agents.pop(agent_id)
            self.controlled_agent_ids.remove(agent_id)
            self.engine.agent_manager.active_agents.pop(agent_id, None)
            
            self.engine.clear_objects([vehicle.id])

        # Step physics world to update sensors/collision detection
        # We don't need full integration, but we need to update the physics world state
        self.engine.taskMgr.step()
        
        # Spawn new agents for this turn
        self._spawn_controlled_agents()
        self._update_background_vehicles()
        
        # Get observations
        obs = self._get_all_obs()
        
        rewards = {aid: 0.0 for aid in self.controlled_agents}
        dones = {aid: False for aid in self.controlled_agents}
        dones["__all__"] = (self.round >= self.config["horizon"]) or (len(self.controlled_agents) == 0 and self.round > 190) # Waymo scenarios are usually ~198 steps (20s @ 10Hz) or 90 steps (9s)
        
        infos = {aid: {"expert_action": expert_actions.get(aid, np.zeros(2))} for aid in self.controlled_agents}
        
        return obs, rewards, dones, infos

    def _get_all_obs(self):
        # Implement custom observation: 30m range, 10 nearest vehicles
        obs_dict = {}
        
        for agent_id, vehicle in self.controlled_agents.items():
            # 1. Ego State
            ego_state = [
                vehicle.position[0], vehicle.position[1],
                vehicle.velocity[0], vehicle.velocity[1],
                vehicle.heading_theta
            ]
            
            # 2. Neighbors
            neighbors = []
            # Iterate through all vehicles in the engine
            candidates = []
            for other_id, other_vehicle in self.engine.agent_manager.active_agents.items():
                if other_id == agent_id:
                    continue
                
                dist = np.linalg.norm(vehicle.position - other_vehicle.position)
                if dist < 30.0:
                    candidates.append((dist, other_vehicle))
            
            # Sort by distance
            candidates.sort(key=lambda x: x[0])
            
            # Take top 10
            top_10 = candidates[:10]
            
            neighbor_feats = []
            for _, neighbor in top_10:
                neighbor_feats.extend([
                    neighbor.position[0] - vehicle.position[0], # Relative pos
                    neighbor.position[1] - vehicle.position[1],
                    neighbor.velocity[0], # Absolute vel? or Relative? Usually relative in MultiAgent
                    neighbor.velocity[1]
                ])
                
            # Pad if < 10
            missing = 10 - len(top_10)
            if missing > 0:
                neighbor_feats.extend([0.0] * (4 * missing))
                
            # Flatten
            obs = np.array(ego_state + neighbor_feats, dtype=np.float32)
            obs_dict[agent_id] = obs
            
        return obs_dict
