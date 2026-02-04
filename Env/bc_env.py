from Env.scenario_env import MultiAgentScenarioEnv
from Env.utils import filter_traffic_tracks_to_birth_lists
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
import numpy as np


class BCScenarioEnv(MultiAgentScenarioEnv):
    """
    Environment for Behavior Cloning Evaluation.
    Uses the same 45-dim observation as ExpertReplayEnv:
    - Ego State (5): x, y, vx, vy, heading
    - Neighbors (40): 10 nearest * (rel_x, rel_y, vx, vy)

    Spawns background (static) vehicles so that observation distribution matches expert data collection:
    expert data is generated with ExpertReplayEnv which includes bg_* in active_agents, so the policy
    was trained on obs that can include those neighbors. Demo should use the same scene for consistency.
    """
    def reset(self, seed=None):
        # Clear background vehicles from previous episode so engine.reset() passes _object_clean_check
        if getattr(self, "engine", None) is not None:
            ids_bg = [
                oid for oid, obj in self.engine.get_objects().items()
                if (getattr(obj, "name", None) or getattr(obj, "id", None) or "").startswith("bg_")
            ]
            if ids_bg:
                self.engine.clear_objects(ids_bg, force_destroy=True)
            for aid in list(self.engine.agent_manager.active_agents.keys()):
                if aid.startswith("bg_"):
                    self.engine.agent_manager.active_agents.pop(aid, None)
        obs = super().reset(seed=seed)
        self._spawn_background_vehicles()
        return self._get_all_obs()

    def _build_birth_lists_from_traffic(self):
        """Same lane/static filter as expert data; return background_vehicles so we spawn them (match training obs)."""
        car_birth_info_list, background_vehicles, obj_to_clean, stats = filter_traffic_tracks_to_birth_lists(
            self.engine.traffic_manager.current_traffic_data,
            self.engine.traffic_manager.sdc_scenario_id,
            self.engine.map_manager,
            return_stats=True,
        )
        if stats["n_controlled"] == 0 and stats["n_total"] > 0:
            print(
                "[BCScenarioEnv] 0 controlled agents: total_vehicles={}, off_lane={}, static={}, no_valid={}.".format(
                    stats["n_total"],
                    stats["n_off_lane"],
                    stats["n_static"],
                    stats["n_no_valid"],
                )
            )
        return car_birth_info_list, background_vehicles, obj_to_clean

    def _spawn_background_vehicles(self):
        """Spawn static background vehicles so they appear in active_agents and thus in obs (same as ExpertReplayEnv)."""
        for sid, car in self.background_vehicles.items():
            if car["show_time"] != self.round:
                continue
            bg_id = f"bg_{car['id']}"
            if bg_id in self.engine.agent_manager.active_agents:
                continue
            vehicle_config = {}
            if "length" in car and "width" in car:
                vehicle_config = {"length": car["length"], "width": car["width"]}
            v = self.engine.spawn_object(
                DefaultVehicle,
                name=bg_id,
                vehicle_config=vehicle_config,
                position=car["begin"],
                heading=car["heading"],
            )
            v.set_velocity([0, 0])
            self.engine.agent_manager.active_agents[bg_id] = v
            v.valid_mask = car["valid"]
            v.start_t = car["show_time"]

    def _update_background_vehicles(self):
        self._spawn_background_vehicles()
        to_remove = []
        objects_to_clear = []
        for aid, v in self.engine.agent_manager.active_agents.items():
            if not aid.startswith("bg_"):
                continue
            if hasattr(v, "valid_mask"):
                if self.round >= len(v.valid_mask) or not v.valid_mask[self.round]:
                    to_remove.append(aid)
                    objects_to_clear.append(v)
        for aid in to_remove:
            self.engine.agent_manager.active_agents.pop(aid, None)
        if objects_to_clear:
            self.engine.clear_objects([v.id for v in objects_to_clear])

    def step(self, action_dict):
        self.round += 1
        for agent_id, action in action_dict.items():
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].before_step(action)
        self.engine.step()
        self.engine.after_step()
        for agent_id in action_dict:
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].after_step()
        self._spawn_controlled_agents()
        self._update_background_vehicles()
        obs = self._get_all_obs()
        rewards = {aid: 0.0 for aid in self.controlled_agents}
        dones = {aid: False for aid in self.controlled_agents}
        dones["__all__"] = self.episode_step >= self.config["horizon"]
        infos = {aid: {} for aid in self.controlled_agents}
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
            # Use engine.agent_manager.active_agents to find neighbors
            # Note: This includes background vehicles if they are in active_agents
            for other_id, other_vehicle in self.engine.agent_manager.active_agents.items():
                if other_id == agent_id:
                    continue
                
                # Check if vehicle is valid/active
                # (MetaDrive manages active_agents, so they should be active)
                
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
                    neighbor.velocity[0], # Absolute vel
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
