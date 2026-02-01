from Env.scenario_env import MultiAgentScenarioEnv
import numpy as np

class BCScenarioEnv(MultiAgentScenarioEnv):
    """
    Environment for Behavior Cloning Evaluation.
    Uses the same 45-dim observation as ExpertReplayEnv:
    - Ego State (5): x, y, vx, vy, heading
    - Neighbors (40): 10 nearest * (rel_x, rel_y, vx, vy)
    """
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
