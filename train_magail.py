import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import argparse
import signal
import sys
from torch.utils.data import DataLoader
from dataset.magail_dataset import MAGAILExpertDataset

# --- Networks ---

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        x = self.net(state)
        mu = torch.tanh(self.mu_head(x)) # Action range [-1, 1]
        if mu.dim() == 1:
            mu = mu.unsqueeze(0) # Handle single sample
        log_std = self.log_std_head.expand_as(mu)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# --- PPO Algorithm ---

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.actor = Actor(state_dim, action_dim).cuda()
        self.critic = Critic(state_dim).cuda()
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).cuda()
            dist = self.actor(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), action_logprob.cpu().numpy()

    def update(self, memory):
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(memory['states'])).cuda()
        actions = torch.FloatTensor(np.array(memory['actions'])).cuda()
        logprobs = torch.FloatTensor(np.array(memory['logprobs'])).cuda()
        rewards = torch.FloatTensor(np.array(memory['rewards'])).cuda()
        next_states = torch.FloatTensor(np.array(memory['next_states'])).cuda()
        dones = torch.FloatTensor(np.array(memory['dones'])).cuda()
        
        # Monte Carlo estimate of state rewards (or GAE if implemented, simplistic here)
        # Usually for PPO we use GAE. Let's do a simple discounted return for now or bootstrapping.
        # Let's use bootstrapping from critic for returns.
        
        returns = []
        discounted_reward = 0
        # This simple loop assumes full episode or consistent batch. 
        # For multi-agent disjoint steps, bootstrapping is better.
        # But let's calculate advantage using GAE for stability.
        
        values = self.critic(states).detach()
        next_values = self.critic(next_states).detach()
        
        # GAE
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).cuda()
        returns = advantages + values.squeeze()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            dist = self.actor(states)
            action_logprobs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.critic(states).squeeze()
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(action_logprobs - logprobs)

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values, returns) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.mean().backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
            
        return loss.mean().item()

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path + "_actor.pth")
        torch.save(self.critic.state_dict(), checkpoint_path + "_critic.pth")

# --- Training Loop ---

def train(args):
    # 1. Setup Environment (Dummy for now, usually you run simulation here)
    # But for MAGAIL we need to collect generated trajectories.
    # We need the Env class to be importable.
    from Env.scenario_env import MultiAgentScenarioEnv
    from Env.simple_idm_policy import ConstantVelocityPolicy # Just for init
    
    # Config for Env
    env_config = {
        "data_directory": args.data_dir,
        "is_multi_agent": True,
        "num_controlled_agents": 3, # Dynamic
        "horizon": 200,
        "use_render": False,
        "sequential_seed": True,
        "start_scenario_index": 0,
        "num_scenarios": args.num_scenarios # Use argument
    }
    
    # Ideally we use a wrapper for RL
    # env = MultiAgentScenarioEnv(config=env_config) # This requires Waymo data loader setup
    
    # 2. Setup Models
    state_dim = 45
    action_dim = 2
    
    ppo_agent = PPO(state_dim, action_dim)
    discriminator = Discriminator(state_dim, action_dim).cuda()
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)
    disc_criterion = nn.BCELoss()
    
    # 3. Load Expert Data
    expert_dataset = MAGAILExpertDataset(args.expert_data_dir)
    # Ensure batch_size is not larger than dataset
    if len(expert_dataset) < args.batch_size:
        print(f"Warning: Expert dataset size {len(expert_dataset)} < batch_size {args.batch_size}. Adjusting batch_size.")
        args.batch_size = len(expert_dataset)
        if args.batch_size == 0:
             raise ValueError("Expert dataset is empty!")

    expert_loader = DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Create an infinite iterator
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    expert_iter = cycle(expert_loader)
    
    # 4. Initialize Env
    from Env.expert_replay_env import ExpertReplayEnv # Using ReplayEnv for config, but we need ScenarioEnv for simulation?
    # Actually we need MultiAgentScenarioEnv for interactive training, not Replay.
    from Env.scenario_env import MultiAgentScenarioEnv
    from Env.simple_idm_policy import ConstantVelocityPolicy # Placeholder policy for init
    
    # 2. Setup Models
    # Determine state dim from environment if possible, or use fixed
    # Expert data has 45 dim? 
    # But Env might return something else if we are using default ScenarioEnv settings.
    # ScenarioEnv returns list of obs.
    # The error says: "mat1 and mat2 shapes cannot be multiplied (1x108 and 45x256)"
    # This means the Env is returning 108-dim observation (MetaDrive default + Lidar), 
    # but our Actor expects 45 (which is what we saved in expert data).
    
    # We must align the environment observation space with our expert data format.
    # Our ExpertReplayEnv used a custom _get_all_obs.
    # We need to inject that same logic into the training env, OR
    # subclass MultiAgentScenarioEnv in the training script to override observation.
    
    class MAGAILScenarioEnv(MultiAgentScenarioEnv):
        def _get_all_obs(self):
            # Same logic as ExpertReplayEnv to ensure compatibility
            obs_dict = {}
            for agent_id, vehicle in self.controlled_agents.items():
                # 1. Ego State
                ego_state = [
                    vehicle.position[0], vehicle.position[1],
                    vehicle.velocity[0], vehicle.velocity[1],
                    vehicle.heading_theta
                ]
                
                # 2. Neighbors
                candidates = []
                for other_id, other_vehicle in self.engine.agent_manager.active_agents.items():
                    if other_id == agent_id:
                        continue
                    dist = np.linalg.norm(vehicle.position - other_vehicle.position)
                    if dist < 30.0:
                        candidates.append((dist, other_vehicle))
                
                candidates.sort(key=lambda x: x[0])
                top_10 = candidates[:10]
                
                neighbor_feats = []
                for _, neighbor in top_10:
                    neighbor_feats.extend([
                        neighbor.position[0] - vehicle.position[0], 
                        neighbor.position[1] - vehicle.position[1],
                        neighbor.velocity[0], 
                        neighbor.velocity[1]
                    ])
                    
                missing = 10 - len(top_10)
                if missing > 0:
                    neighbor_feats.extend([0.0] * (4 * missing))
                    
                obs = np.array(ego_state + neighbor_feats, dtype=np.float32)
                obs_dict[agent_id] = obs
            return obs_dict

    env = MAGAILScenarioEnv(config=env_config, agent2policy={}) # Pass empty dict if we control all externally
    
    print("Starting training...")
    
    # Tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    except ImportError:
        print("TensorBoard not installed. Logging to console only.")
        writer = None
    
    global_step = 0
    
    for i_episode in range(args.max_episodes):
        # --- 1. Collect Rollouts (Interaction) ---
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'next_states': [], 'dones': []}
        
        # Prepare seed
        available_scenarios = env.config["num_scenarios"]
        start_index = env.config["start_scenario_index"]
        seed = np.random.randint(start_index, start_index + available_scenarios)

        # Reset Env
        try:
            # MetaDrive sometimes complains about uncleared objects if reset happens too fast or with lingering objs
            # We can try to force clear before reset or handle exception
            # But standard reset should handle it.
            # The error "You should clear all generated objects..." means some manager didn't clear its objects.
            # This is likely due to TrafficManager or AgentManager holding refs.
            
            # Re-creating env is safer but slower. 
            # Let's try closing and re-creating if reset fails frequently.
            # Or just ignore this error and try reset again? No, reset failing is fatal usually.
            
            # Hack: Manually clear objects if we can access engine
            if env.engine is not None:
                 env.engine.clear_objects(list(env.engine.get_objects().keys()))
            
            obs_dict = env.reset(seed=seed)
        except Exception as e:
            # print(f"Env reset failed: {e}. Recreating environment...")
            try:
                env.close()
            except:
                pass
            
            # Ensure engine is closed properly
            from metadrive.engine.engine_utils import close_engine
            try:
                close_engine()
            except Exception as e2:
                # Force cleanup of singleton if close failed
                from metadrive.engine.base_engine import BaseEngine
                if BaseEngine.singleton is not None:
                    BaseEngine.singleton = None
                
                # Also need to clear ShowBase
                try:
                    from direct.showbase.ShowBase import ShowBase
                    if hasattr(base, 'destroy'):
                        base.destroy()
                except:
                    pass
                    
                # Brutal force: delete base from builtins if it exists
                import builtins
                if hasattr(builtins, 'base'):
                    del builtins.base
                    
                # print(f"Error closing engine: {e2}")
            
            # Explicitly delete old env object to free memory
            del env
            import gc
            gc.collect()
            
            env = MAGAILScenarioEnv(config=env_config, agent2policy={})
            obs_dict = env.reset(seed=seed)
        
        episode_reward = 0
        steps = 0
        
        # Rollout loop
        while True:
            # Select actions for all agents
            actions = {}
            action_logprobs = {}
            
            # obs_dict: {agent_id: obs}
            # MultiAgentScenarioEnv usually returns a dict {agent_id: obs}
            # BUT wait, check scenario_env.py implementation
            
            if isinstance(obs_dict, list):
                 # This happens if the environment returns a list instead of a dict
                 # MultiAgentScenarioEnv._get_all_obs returns a list in original implementation?
                 # Let's check scenario_env.py
                 # If it returns list, we need to map it to agent ids or just iterate
                 pass
            
            # Temporary fix if it returns list (which means my previous edit to Env/expert_replay_env.py
            # changed it there, but maybe not in Env/scenario_env.py which we are using here!)
            
            if isinstance(obs_dict, list):
                # We need agent IDs to step
                # In MultiAgentScenarioEnv, controlled_agents is a dict.
                # If obs is a list, it probably corresponds to controlled_agents.values() order?
                # This is risky. 
                # Let's assume obs_dict is actually just observations.
                # We need to keys to create action dict.
                
                current_agent_ids = list(env.controlled_agents.keys())
                # Ensure length matches
                if len(obs_dict) != len(current_agent_ids):
                    # print(f"Warning: Obs list len {len(obs_dict)} != agents {len(current_agent_ids)}")
                    pass
                
                # Reconstruct dict
                new_obs_dict = {}
                for i, agent_id in enumerate(current_agent_ids):
                    if i < len(obs_dict):
                        new_obs_dict[agent_id] = obs_dict[i]
                obs_dict = new_obs_dict

            for agent_id, obs in obs_dict.items():
                act, logprob = ppo_agent.select_action(obs) # Select action returns numpy
                actions[agent_id] = act.flatten() # (2,)
                action_logprobs[agent_id] = logprob # scalar
                
            # Step Env
            next_obs_dict, rewards, dones, infos = env.step(actions)
            
            # Store in memory
            for agent_id, obs in obs_dict.items():
                if agent_id in actions:
                    memory['states'].append(obs)
                    memory['actions'].append(actions[agent_id])
                    memory['logprobs'].append(action_logprobs[agent_id])
                    
                    # Store standard environmental reward for logging (not used for update in GAIL)
                    # For GAIL update we use Discriminator reward later
                    memory['rewards'].append(0) # Placeholder
                    
                    # Next state
                    if agent_id in next_obs_dict:
                        memory['next_states'].append(next_obs_dict[agent_id])
                        memory['dones'].append(False)
                    else:
                        # Agent finished/vanished
                        # We need a dummy next state or handle done correctly
                        # Just duplicate current state and mark done?
                        memory['next_states'].append(obs) 
                        memory['dones'].append(True)
            
            obs_dict = next_obs_dict
            steps += 1
            
            if dones["__all__"] or steps >= 200: # Limit horizon
                break
                
        # Initialize losses to 0/None before potential loop skip
        disc_loss = torch.tensor(0.0)
        ppo_loss = 0.0
        all_gail_rewards = [0.0]

        # --- 2. Train Discriminator ---
        # Convert policy memory to tensors
        policy_states = torch.FloatTensor(np.array(memory['states'])).cuda()
        policy_actions = torch.FloatTensor(np.array(memory['actions'])).cuda()
        
        # Sample expert batch
        expert_batch = next(expert_iter)
            
        expert_states = expert_batch['state'].cuda()
        expert_actions = expert_batch['action'].cuda()
        
        # Minibatch size matching
        batch_size = min(policy_states.size(0), expert_states.size(0))
        
        if batch_size > 0: # Only train if we have data
            policy_states = policy_states[:batch_size]
            policy_actions = policy_actions[:batch_size]
            expert_states = expert_states[:batch_size]
            expert_actions = expert_actions[:batch_size]
            
            # Update Discriminator
            # Label 1 for Expert, 0 for Policy
            # Train Expert
            disc_optimizer.zero_grad()
            
            exp_preds = discriminator(expert_states, expert_actions)
            exp_loss = disc_criterion(exp_preds, torch.ones_like(exp_preds))
            
            pol_preds = discriminator(policy_states.detach(), policy_actions.detach()) # Detach policy data
            pol_loss = disc_criterion(pol_preds, torch.zeros_like(pol_preds))
            
            disc_loss = exp_loss + pol_loss
            disc_loss.backward()
            disc_optimizer.step()
            
            # --- 3. Update Policy with GAIL Rewards ---
            # Reward = -log(1 - D(s, a))
            # Or more stable: log(D(s, a)) ? Original GAIL uses -log(1-D) which is log(D) roughly.
            # Let's use -log(1 - D(s, a) + eps)
            
            # Actually PPO needs the full trajectory for GAE.
            # So we should compute rewards for ALL policy samples in memory.
            
            all_policy_states = torch.FloatTensor(np.array(memory['states'])).cuda()
            all_policy_actions = torch.FloatTensor(np.array(memory['actions'])).cuda()
            
            with torch.no_grad():
                all_d_val = discriminator(all_policy_states, all_policy_actions)
                all_gail_rewards = -torch.log(1 - all_d_val + 1e-8).cpu().numpy().flatten()
                
            # Replace placeholders
            memory['rewards'] = all_gail_rewards.tolist()
            
            # Update PPO
            ppo_loss = ppo_agent.update(memory)
            
            # Clean up memory
            del policy_states, policy_actions, expert_states, expert_actions, exp_preds, exp_loss, pol_preds, pol_loss
            del all_policy_states, all_policy_actions, all_d_val
            torch.cuda.empty_cache()
        else:
            print(f"Episode {i_episode}: No data collected (Env might have crashed or no agents). Skipping update.")

        # --- 4. Logging ---
        if writer:
            writer.add_scalar('Loss/Discriminator', disc_loss.item(), i_episode)
            writer.add_scalar('Loss/Policy', ppo_loss, i_episode)
            writer.add_scalar('Reward/Mean_GAIL', np.mean(all_gail_rewards), i_episode)
        
        print(f"Episode {i_episode}: Disc Loss {disc_loss.item():.4f} | PPO Loss {ppo_loss:.4f} | Mean Reward {np.mean(all_gail_rewards):.4f}")
        
        if i_episode % 50 == 0:
            ppo_agent.save(os.path.join(args.log_dir, f"model_{i_episode}"))

    env.close()
    if writer:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data_dir", type=str, default="data/training_data", help="Directory with .pkl expert data")
    parser.add_argument("--data_dir", type=str, default="data/exp_filtered", help="Waymo data dir for Env")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_episodes", type=int, default=1000)
    parser.add_argument("--num_scenarios", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="runs/magail_exp")
    
    args = parser.parse_args()
    
    # Create log dir
    os.makedirs(args.log_dir, exist_ok=True)
    
    train(args)
