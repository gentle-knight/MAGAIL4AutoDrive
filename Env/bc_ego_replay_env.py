"""
Single-agent BC evaluation environment: only ego (SDC) is controlled by the policy;
other vehicles are replayed from expert trajectories (same as data collection).
"""
import numpy as np
from Env.expert_replay_env import ExpertReplayEnv
from Env.hbbc_background_policy import HBBCBackgroundController


class BCEgoReplayEnv(ExpertReplayEnv):
    """
    For single-agent BC evaluation: controlled_agents exposes only SDC (default_agent).
    Other vehicles are still spawned and replayed by expert; internally we keep them
    in _replay_agents so step() can update them.
    """

    def reset(self, seed=None):
        obs = super().reset(seed=seed)
        self.enable_hbbc_background = bool(self.config.get("enable_hbbc_background", False))
        self.hbbc_controller = None
        self._hbbc_runtime_logged = False
        if self.enable_hbbc_background:
            self.hbbc_controller = HBBCBackgroundController(
                model_path=self.config.get("hbbc_model_path", "models/hbbc/hbbc.pt"),
                device=self.config.get("hbbc_inference_device", "cpu"),
                latent_mode=self.config.get("hbbc_latent_mode", "per_vehicle_fixed"),
                latent_json_path=self.config.get("hbbc_latent_json_path"),
                seed=int(self.config.get("seed", 0)),
                dt=float(self.config.get("hbbc_dt", 0.1)),
            )
            self.hbbc_controller.reset_episode()
        # Expose only SDC as the controlled agent for the evaluator
        self._replay_agents = dict(self.controlled_agents)
        if self.replay_sdc and self.sdc_vehicle is not None:
            self.controlled_agents = {self.sdc_agent_id: self.sdc_vehicle}
            self.controlled_agent_ids = [self.sdc_agent_id]
        else:
            self.controlled_agents = {}
            self.controlled_agent_ids = []
        return self._get_all_obs()

    def _get_all_obs(self):
        """Return only ego (SDC) observation so evaluator has a single agent."""
        if not self.controlled_agents or self.sdc_vehicle is None:
            return {}
        obs = self._obs_for_vehicle(self.sdc_vehicle, exclude_agent_id=self.sdc_agent_id)
        return {self.sdc_agent_id: obs}

    def step(self, action_dict=None):
        self.round += 1
        expert_actions = {}
        agents_to_remove = []

        # SDC: use policy action if provided, else expert replay
        if self.replay_sdc and self.sdc_vehicle is not None and self.sdc_track is not None:
            policy_action = None
            if action_dict and self.sdc_agent_id in action_dict:
                policy_action = np.asarray(action_dict[self.sdc_agent_id], dtype=np.float64)
            next_step = self.round
            curr_step = self.round - 1
            if next_step < len(self.sdc_track["state"]["position"]) and self.sdc_track["state"]["valid"][next_step]:
                curr_state = {
                    "position": self.sdc_track["state"]["position"][curr_step],
                    "heading": self.sdc_track["state"]["heading"][curr_step],
                    "velocity": self.sdc_track["state"]["velocity"][curr_step],
                }
                if policy_action is not None:
                    next_state = self.inverse_dynamics.apply_action(curr_state, policy_action, dt=0.1)
                    expert_actions[self.sdc_agent_id] = policy_action
                else:
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
                self.sdc_vehicle.last_expert_action = expert_actions[self.sdc_agent_id]

        # Replay other vehicles: restore full controlled_agents for internal logic
        self.controlled_agents = dict(self._replay_agents)
        self.controlled_agent_ids = list(self.controlled_agents.keys())
        hbbc_batch = []
        hbbc_curr_states = {}
        for agent_id, vehicle in self.controlled_agents.items():
            track = vehicle.expert_track
            next_step = self.round
            if next_step >= len(track["state"]["position"]):
                agents_to_remove.append(agent_id)
                continue
            if not track["state"]["valid"][next_step]:
                agents_to_remove.append(agent_id)
                continue
            if self.enable_hbbc_background and self.hbbc_controller is not None:
                # HBBC autonomous rollout: use vehicle's own previous-step state
                curr_state = {
                    "position": np.asarray(vehicle.position, dtype=np.float64),
                    "heading": float(vehicle.heading_theta),
                    "velocity": np.asarray(vehicle.velocity, dtype=np.float64),
                }
                object_id = str(getattr(vehicle, "original_id", agent_id))
                hbbc_batch.append((agent_id, vehicle, object_id, agent_id))
                hbbc_curr_states[agent_id] = curr_state
            else:
                curr_step = self.round - 1
                curr_state = {
                    "position": track["state"]["position"][curr_step],
                    "heading": track["state"]["heading"][curr_step],
                    "velocity": track["state"]["velocity"][curr_step],
                }
                next_state = {
                    "position": track["state"]["position"][next_step],
                    "heading": track["state"]["heading"][next_step],
                    "velocity": track["state"]["velocity"][next_step],
                }
                action, _ = self.inverse_dynamics.compute_action(curr_state, next_state, dt=0.1)
                expert_actions[agent_id] = action
                vehicle.set_position(next_state["position"])
                vehicle.set_heading_theta(next_state["heading"])
                vehicle.set_velocity(next_state["velocity"])
                vehicle.last_expert_action = action

        if hbbc_batch and self.hbbc_controller is not None:
            hbbc_actions = self.hbbc_controller.infer_actions(hbbc_batch)
            if not self._hbbc_runtime_logged:
                print(f"[HBBC] background policy active, current dynamic agents: {len(hbbc_batch)}")
                self._hbbc_runtime_logged = True
            for agent_id, _, _, _ in hbbc_batch:
                curr_state = hbbc_curr_states[agent_id]
                action = hbbc_actions[agent_id]
                next_state = self.inverse_dynamics.apply_action(curr_state, action, dt=0.1)
                expert_actions[agent_id] = action
                vehicle = self.controlled_agents[agent_id]
                vehicle.set_position(next_state["position"])
                vehicle.set_heading_theta(next_state["heading"])
                vehicle.set_velocity(next_state["velocity"])
                try:
                    vehicle.last_current_action.append(action)
                except Exception:
                    pass
                vehicle.last_expert_action = action
        for agent_id in agents_to_remove:
            vehicle = self.controlled_agents[agent_id]
            self.controlled_agents.pop(agent_id)
            self.controlled_agent_ids.remove(agent_id)
            self.engine.agent_manager.active_agents.pop(agent_id, None)
            self.engine.clear_objects([vehicle.id])
            if self.hbbc_controller is not None:
                self.hbbc_controller.remove_vehicle(agent_id)
        self.engine.taskMgr.step()
        self._spawn_controlled_agents()
        self._update_background_vehicles()
        self._replay_agents = dict(self.controlled_agents)
        # Expose only SDC again
        if self.replay_sdc and self.sdc_vehicle is not None:
            self.controlled_agents = {self.sdc_agent_id: self.sdc_vehicle}
            self.controlled_agent_ids = [self.sdc_agent_id]
        else:
            self.controlled_agents = {}
            self.controlled_agent_ids = []

        obs = self._get_all_obs()
        rewards = {}
        infos = {aid: {"expert_action": expert_actions.get(aid, np.zeros(2))} for aid in self.controlled_agents}
        if self.sdc_agent_id in self.controlled_agents and self.sdc_vehicle is not None:
            speed_coef = float(self.config.get("reward_speed_coef", 0.05))
            collision_distance = float(self.config.get("collision_distance", 6.0))
            collision_penalty = float(self.config.get("collision_penalty", 100.0))
            speed = float(np.linalg.norm(self.sdc_vehicle.velocity))
            r_speed = speed_coef * speed
            min_dist = float("inf")
            for other_id, other_vehicle in self.engine.agent_manager.active_agents.items():
                if other_id == self.sdc_agent_id:
                    continue
                try:
                    d = float(np.linalg.norm(self.sdc_vehicle.position - other_vehicle.position))
                    min_dist = min(min_dist, d)
                except Exception:
                    continue
            near_collision = min_dist < collision_distance
            r_collision = -collision_penalty if near_collision else 0.0
            rewards[self.sdc_agent_id] = r_speed + r_collision
            infos[self.sdc_agent_id].update(
                near_collision=near_collision,
                min_dist=min_dist if np.isfinite(min_dist) else None,
                r_speed=r_speed,
                r_collision=r_collision,
            )
        dones = {aid: False for aid in self.controlled_agents}
        dones["__all__"] = self.round >= self.config["horizon"] or (len(self._replay_agents) == 0 and self.round > 190)
        return obs, rewards, dones, infos
