from Env.scenario_env import MultiAgentScenarioEnv
from Env.hbbc_background_policy import HBBCBackgroundController
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
    def _init_hbbc_background(self):
        self.enable_hbbc_background = bool(self.config.get("enable_hbbc_background", False))
        self.hbbc_dynamic_agents = {}
        self._spawned_dynamic_bg_ids = set()
        self.hbbc_controller = None
        if not self.enable_hbbc_background:
            return
        self.hbbc_controller = HBBCBackgroundController(
            model_path=self.config.get("hbbc_model_path", "models/hbbc/hbbc.pt"),
            device=self.config.get("hbbc_inference_device", "cpu"),
            latent_mode=self.config.get("hbbc_latent_mode", "per_vehicle_fixed"),
            latent_json_path=self.config.get("hbbc_latent_json_path"),
            seed=int(self.config.get("seed", 0)),
            dt=float(self.config.get("hbbc_dt", 0.1)),
        )
        self.hbbc_controller.reset_episode()

    def _move_excess_controlled_to_hbbc_background(self):
        if not self.enable_hbbc_background:
            return
        keep_n = int(self.config.get("num_controlled_agents", 0))
        keep_n = max(0, keep_n)
        ordered_ids = list(self.controlled_agents.keys())
        keep_ids = set(ordered_ids[:keep_n])
        move_ids = [aid for aid in ordered_ids if aid not in keep_ids]
        for aid in move_ids:
            self.hbbc_dynamic_agents[aid] = self.controlled_agents[aid]
            self.controlled_agents.pop(aid, None)
            if aid in self.controlled_agent_ids:
                self.controlled_agent_ids.remove(aid)
        self._spawned_dynamic_bg_ids.update(move_ids)

    def _apply_hbbc_before_step(self):
        if not self.enable_hbbc_background or not self.hbbc_dynamic_agents:
            return
        batch = []
        for aid, vehicle in self.hbbc_dynamic_agents.items():
            object_id = getattr(vehicle, "original_id", None) or aid.replace("controlled_", "", 1)
            batch.append((aid, vehicle, str(object_id) if object_id is not None else None, aid))
        actions = self.hbbc_controller.infer_actions(batch)
        for aid, vehicle in self.hbbc_dynamic_agents.items():
            action = actions.get(aid, np.zeros(2, dtype=np.float32))
            vehicle.before_step(action)

    def _apply_hbbc_after_step(self):
        if not self.enable_hbbc_background:
            return
        for vehicle in self.hbbc_dynamic_agents.values():
            vehicle.after_step()

    def reset(self, seed=None):
        self._init_hbbc_background()
        # Clear background vehicles from previous episode so engine.reset() passes _object_clean_check
        if getattr(self, "engine", None) is not None:
            ids_bg = [
                oid for oid, obj in self.engine.get_objects().items()
                if (getattr(obj, "name", None) or getattr(obj, "id", None) or "").startswith(("bg_", "controlled_"))
            ]
            if ids_bg:
                self.engine.clear_objects(ids_bg, force_destroy=True)
            for aid in list(self.engine.agent_manager.active_agents.keys()):
                if aid.startswith("bg_") or aid.startswith("controlled_"):
                    self.engine.agent_manager.active_agents.pop(aid, None)
        obs = super().reset(seed=seed)
        self._move_excess_controlled_to_hbbc_background()
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
        """Spawn all static background vehicles once at reset (no show_time filter; same as ExpertReplayEnv)."""
        for sid, car in self.background_vehicles.items():
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
            v.valid_mask = car.get("valid")
            v.start_t = car.get("show_time")

    def _update_background_vehicles(self):
        # Static vehicles are spawned once at init and never removed.
        pass

    def step(self, action_dict):
        if action_dict is None:
            action_dict = {}
        self.round += 1
        for agent_id, action in action_dict.items():
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].before_step(action)
        self._apply_hbbc_before_step()
        self.engine.step()
        self.engine.after_step()
        for agent_id in action_dict:
            if agent_id in self.controlled_agents:
                self.controlled_agents[agent_id].after_step()
        self._spawn_controlled_agents()
        self._move_excess_controlled_to_hbbc_background()
        self._apply_hbbc_after_step()
        self._update_background_vehicles()
        obs = self._get_all_obs()

        # Reward shaping for evaluation/rollout monitoring (BC training itself doesn't use env reward).
        speed_coef = float(self.config.get("reward_speed_coef", 0.05))
        collision_distance = float(self.config.get("collision_distance", 6.0))
        collision_penalty = float(self.config.get("collision_penalty", 100.0))

        # Pre-collect all active vehicles (includes background vehicles).
        active_agents = list(self.engine.agent_manager.active_agents.items())

        rewards = {}
        infos = {}
        for aid, vehicle in self.controlled_agents.items():
            # Speed reward
            speed = getattr(vehicle, "speed", None)
            if speed is None:
                speed = float(np.linalg.norm(vehicle.velocity))
            r_speed = speed_coef * float(speed)

            # Near-collision penalty (distance-based, simulator-agnostic)
            min_dist = float("inf")
            for other_id, other_vehicle in active_agents:
                if other_id == aid:
                    continue
                try:
                    dist = float(np.linalg.norm(vehicle.position - other_vehicle.position))
                except Exception:
                    continue
                if dist < min_dist:
                    min_dist = dist

            near_collision = bool(min_dist < collision_distance)
            r_collision = -collision_penalty if near_collision else 0.0

            rewards[aid] = float(r_speed + r_collision)
            infos[aid] = {
                "near_collision": near_collision,
                "min_dist": (min_dist if np.isfinite(min_dist) else None),
                "r_speed": float(r_speed),
                "r_collision": float(r_collision),
            }
        dones = {aid: False for aid in self.controlled_agents}
        dones["__all__"] = self.episode_step >= self.config["horizon"]
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
