import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from Env.hbbc_actor_critic import ActorCritic


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _normalize_eps(eps: np.ndarray) -> np.ndarray:
    eps = np.asarray(eps, dtype=np.float32).reshape(-1)
    if eps.shape[0] != 6:
        raise ValueError(f"latent_eps must be 6-dim, got {eps.shape[0]}")
    norm = float(np.linalg.norm(eps))
    if norm < 1e-8:
        eps = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        eps = eps / norm
    return np.clip(eps, -1.0, 1.0)


def _normalize_c(latent_c: np.ndarray) -> np.ndarray:
    c = np.asarray(latent_c, dtype=np.float32).reshape(-1)
    if c.shape[0] != 4:
        raise ValueError(f"latent_c must be 4-dim, got {c.shape[0]}")
    idx = int(np.argmax(c))
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot


def _sample_latent(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    eps = _normalize_eps(rng.randn(6).astype(np.float32))
    mode = int(rng.randint(0, 4))
    c = np.zeros(4, dtype=np.float32)
    c[mode] = 1.0
    return eps, c


@dataclass
class VehicleStateCache:
    last_heading_theta: Optional[float] = None
    last_action: Tuple[float, float] = (0.0, 0.0)
    last_speed_km_h: Optional[float] = None


class HBBCModelWrapper:
    _cache: Dict[Tuple[str, str], "HBBCModelWrapper"] = {}

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = os.path.abspath(model_path)
        self.device = torch.device(device)
        self.model = self._load_model()

    @classmethod
    def get(cls, model_path: str, device: str = "cpu") -> "HBBCModelWrapper":
        key = (os.path.abspath(model_path), str(torch.device(device)))
        if key not in cls._cache:
            cls._cache[key] = HBBCModelWrapper(model_path=key[0], device=key[1])
        return cls._cache[key]

    def _load_model(self) -> ActorCritic:
        model = ActorCritic(
            num_actor_obs=18,
            num_critic_obs=18,
            num_actions=2,
            latent_c_dim=4,
            latent_eps_dim=6,
            use_style_latent=True,
        ).to(self.device)
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["actor_critic"] if isinstance(ckpt, dict) and "actor_critic" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"HBBC checkpoint missing required keys for {self.model_path}: {missing}"
            )
        if unexpected:
            print(f"[HBBC] ignore extra checkpoint keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        model.eval()
        return model

    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).to(self.device)
            actions = self.model.act_inference(obs_t).cpu().numpy()
        return np.clip(actions, -1.0, 1.0)


class HBBCLatentManager:
    def __init__(self, mode: str = "per_vehicle_fixed", seed: int = 0, latent_json_path: Optional[str] = None):
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.latent_json_path = latent_json_path
        self.manual_object_latent: Dict[str, Dict[str, np.ndarray]] = {}
        self.manual_agent_latent: Dict[str, Dict[str, np.ndarray]] = {}
        self.manual_global_latent: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.vehicle_latent: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._episode_latent: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._load_manual_latent_json()

    def reset_episode(self):
        self.vehicle_latent.clear()
        self._episode_latent = None
        if self.mode == "per_episode_reset":
            self._episode_latent = _sample_latent(self.rng)

    def _load_manual_latent_json(self):
        if not self.latent_json_path:
            return
        path = os.path.abspath(self.latent_json_path)
        if not os.path.exists(path):
            print(f"[HBBC] latent json not found: {path}, fallback to random sampling.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[HBBC] failed to load latent json ({path}): {e}. fallback to random sampling.")
            return

        object_section = data.get("object_id", {})
        agent_section = data.get("agent_id", {})
        global_section = data.get("global")

        if global_section is not None:
            parsed = self._parse_one_latent(global_section, "global")
            if parsed is not None:
                self.manual_global_latent = (parsed["latent_eps"], parsed["latent_c"])

        for key, value in object_section.items():
            parsed = self._parse_one_latent(value, f"object_id:{key}")
            if parsed is not None:
                self.manual_object_latent[str(key)] = parsed
        for key, value in agent_section.items():
            parsed = self._parse_one_latent(value, f"agent_id:{key}")
            if parsed is not None:
                self.manual_agent_latent[str(key)] = parsed

    @staticmethod
    def _parse_one_latent(value: dict, name: str) -> Optional[Dict[str, np.ndarray]]:
        if not isinstance(value, dict):
            print(f"[HBBC] invalid latent entry ({name}): expect dict.")
            return None
        try:
            eps = _normalize_eps(value["latent_eps"])
            c = _normalize_c(value["latent_c"])
            return {"latent_eps": eps, "latent_c": c}
        except Exception as e:
            print(f"[HBBC] invalid latent entry ({name}): {e}")
            return None

    def _lookup_manual(self, object_id: Optional[str], agent_id: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if object_id is not None and object_id in self.manual_object_latent:
            e = self.manual_object_latent[object_id]["latent_eps"]
            c = self.manual_object_latent[object_id]["latent_c"]
            return e, c
        if agent_id is not None and agent_id in self.manual_agent_latent:
            e = self.manual_agent_latent[agent_id]["latent_eps"]
            c = self.manual_agent_latent[agent_id]["latent_c"]
            return e, c
        if self.manual_global_latent is not None:
            return self.manual_global_latent
        return None

    def get_latent(self, vehicle_key: str, object_id: Optional[str], agent_id: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        manual = self._lookup_manual(object_id=object_id, agent_id=agent_id)
        if manual is not None:
            return manual
        if self.mode == "per_episode_reset":
            if self._episode_latent is None:
                self._episode_latent = _sample_latent(self.rng)
            return self._episode_latent
        if vehicle_key not in self.vehicle_latent:
            self.vehicle_latent[vehicle_key] = _sample_latent(self.rng)
        return self.vehicle_latent[vehicle_key]


class HBBCBackgroundController:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        latent_mode: str = "per_vehicle_fixed",
        latent_json_path: Optional[str] = None,
        seed: int = 0,
        dt: float = 0.1,
    ):
        self.model = HBBCModelWrapper.get(model_path=model_path, device=device)
        self.latent_mgr = HBBCLatentManager(mode=latent_mode, seed=seed, latent_json_path=latent_json_path)
        self.dt = float(dt)
        self.vehicle_state: Dict[str, VehicleStateCache] = {}

    def reset_episode(self):
        self.latent_mgr.reset_episode()
        self.vehicle_state.clear()

    def remove_vehicle(self, vehicle_key: str):
        self.vehicle_state.pop(vehicle_key, None)
        self.latent_mgr.vehicle_latent.pop(vehicle_key, None)

    def _build_base_state(self, vehicle, vehicle_key: str) -> np.ndarray:
        state = self.vehicle_state.get(vehicle_key)
        if state is None:
            state = VehicleStateCache()
            self.vehicle_state[vehicle_key] = state

        speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0))
        max_speed_km_h = float(getattr(vehicle, "max_speed_km_h", 120.0))
        veh_vel = np.clip((speed_km_h + 1.0) / (max_speed_km_h + 1.0), 0.0, 1.0)

        heading_theta = float(getattr(vehicle, "heading_theta", 0.0))
        if state.last_heading_theta is None:
            yaw_rate = 0.0
        else:
            yaw_rate = _wrap_to_pi(heading_theta - state.last_heading_theta) / self.dt
        yaw_rate = float(np.clip(yaw_rate, -5.0, 5.0))

        current_action = getattr(vehicle, "current_action", None)
        if current_action is None:
            last_action_0, last_action_1 = state.last_action
        else:
            try:
                last_action_0, last_action_1 = float(current_action[0]), float(current_action[1])
            except Exception:
                last_action_0, last_action_1 = state.last_action

        state.last_heading_theta = heading_theta
        state.last_speed_km_h = speed_km_h
        state.last_action = (last_action_0, last_action_1)

        obs = np.array(
            [
                0.0,
                0.0,
                0.0,
                veh_vel,
                0.0,
                yaw_rate * 0.5,
                last_action_0,
                last_action_1,
            ],
            dtype=np.float32,
        )
        return obs

    def build_obs(self, vehicle, vehicle_key: str, object_id: Optional[str], agent_id: Optional[str]) -> np.ndarray:
        base = self._build_base_state(vehicle, vehicle_key=vehicle_key)
        eps, c = self.latent_mgr.get_latent(vehicle_key=vehicle_key, object_id=object_id, agent_id=agent_id)
        return np.concatenate([base, eps, c], axis=-1).astype(np.float32)

    def infer_actions(self, batch: List[Tuple[str, object, Optional[str], Optional[str]]]) -> Dict[str, np.ndarray]:
        if not batch:
            return {}
        obs_list = []
        vehicle_ids = []
        for vehicle_key, vehicle, object_id, agent_id in batch:
            obs_list.append(self.build_obs(vehicle, vehicle_key=vehicle_key, object_id=object_id, agent_id=agent_id))
            vehicle_ids.append(vehicle_key)
        actions = self.model.act_batch(np.stack(obs_list, axis=0))
        out = {}
        for idx, key in enumerate(vehicle_ids):
            out[key] = actions[idx].astype(np.float32)
        return out
