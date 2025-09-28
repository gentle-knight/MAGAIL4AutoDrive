import torch
from torch import nn
from .bert import Bert


DISC_LOGIT_INIT_SCALE = 1.0


class GAILDiscrim(Bert):

    def __init__(self, input_dim, reward_i_coef=1.0, reward_t_coef=1.0, normalizer=None, device=None):
        super().__init__(input_dim=input_dim, output_dim=1, TANH=False)
        self.device = device
        self.reward_t_coef = reward_t_coef
        self.reward_i_coef = reward_i_coef
        self.normalizer = normalizer

    def calculate_reward(self, states_gail, next_states_gail, rewards_t):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        states_gail = states_gail.clone()
        next_states_gail = next_states_gail.clone()
        states = torch.cat([states_gail, next_states_gail], dim=-1)
        with torch.no_grad():
            if self.normalizer is not None:
                states = self.normalizer.normalize_torch(states, self.device)
            rewards_t = self.reward_t_coef * rewards_t
            d = self.forward(states)
            prob = 1 / (1 + torch.exp(-d))
            rewards_i = self.reward_i_coef * (
                -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device))))
            rewards = rewards_t + rewards_i
        return rewards, rewards_t / (self.reward_t_coef + 1e-10), rewards_i / (self.reward_i_coef + 1e-10)

    def get_disc_logit_weights(self):
        return torch.flatten(self.classifier.weight)

    def get_disc_weights(self):
        weights = []
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.classifier.weight))
        return weights
