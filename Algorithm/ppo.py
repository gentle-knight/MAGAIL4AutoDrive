import os
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from buffer import RolloutBuffer
from bert import Bert
from policy import StateIndependentPolicy
from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self, state_shape, device, gamma):

        self.learning_steps = 0
        self.state_shape = state_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state_list):
        action_list = []
        log_pi_list = []
        if type(state_list).__module__ != "torch":
            state_list = torch.tensor(state_list, dtype=torch.float, device=self.device)
        with torch.no_grad():
            for state in state_list:
                action, log_pi = self.actor.sample(state.unsqueeze(0))
                action_list.append(action.cpu().numpy()[0])
                log_pi_list.append(log_pi.item())
        return action_list, log_pi_list

    def exploit(self, state_list):
        action_list = []
        state_list = torch.tensor(state_list, dtype=torch.float, device=self.device)
        with torch.no_grad():
            for state in state_list:
                action = self.actor(state.unsqueeze(0))
                action_list.append(action.cpu().numpy()[0])
        return action_list

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self, writer, total_steps):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


class PPO(Algorithm):

    def __init__(self, state_shape, device, gamma=0.995, rollout_length=2048,
                 units_actor=(64, 64), epoch_ppo=10, clip_eps=0.2,
                 lambd=0.97, max_grad_norm=1.0, desired_kl=0.01, surrogate_loss_coef=2.,
                 value_loss_coef=5., entropy_coef=0., bounds_loss_coef=10., lr_actor=1e-3, lr_critic=1e-3,
                 lr_disc=1e-3, auto_lr=True, use_adv_norm=True, max_steps=10000000):
        super().__init__(state_shape, device, gamma)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_disc = lr_disc
        self.auto_lr = auto_lr

        self.use_adv_norm = use_adv_norm

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = Bert(
            input_dim=state_shape,
            output_dim=1
        ).to(device)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.surrogate_loss_coef = surrogate_loss_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.bounds_loss_coef = bounds_loss_coef
        self.max_steps = max_steps

        self.optim_actor = Adam([{'params': self.actor.parameters()}], lr=lr_actor)
        # self.optim_actor = Adam([
        #         {'params': self.actor.net.f_net.parameters(), 'lr': lr_actor},
        #         {'params': self.actor.net.k_net.parameters(), 'lr': lr_actor/3}])
        self.optim_critic = Adam([{'params': self.critic.parameters()}], lr=lr_critic)

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state_list, state_gail):
        state_list = torch.tensor(state_list, dtype=torch.float, device=self.device)
        state_gail = torch.tensor(state_gail, dtype=torch.float, device=self.device)
        action_list, log_pi_list = self.explore(state_list)
        next_state, reward, terminated, truncated, info = env.step(np.array(action_list))
        next_state_gail = env.state_gail
        done = terminated or truncated

        means = self.actor.means.detach().cpu().numpy()[0]
        stds = (self.actor.log_stds.exp()).detach().cpu().numpy()[0]

        self.buffer.append(state_list, state_gail, action_list, reward, done, terminated, log_pi_list,
                           next_state, next_state_gail, means, stds)

        if done:
            next_state = env.reset()
            next_state_gail = env.state_gail

        return next_state, next_state_gail, info

    def update(self, writer, total_steps):
        pass

    def update_ppo(self, states, actions, rewards, dones, tm_dones, log_pi_list, next_states, mus, sigmas, writer,
                   total_steps):
        with torch.no_grad():
            values = self.critic(states.detach())
            next_values = self.critic(next_states.detach())

        targets, gaes = self.calculate_gae(
            values, rewards, dones, tm_dones, next_values, self.gamma, self.lambd)

        state_list = states.permute(1, 0, 2)
        action_list = actions.permute(1, 0, 2)

        for i in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            for state, action, log_pi in state_list, action_list, log_pi_list:
                self.update_actor(state, action, log_pi, gaes, mus, sigmas, writer)

        # self.lr_decay(total_steps, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        loss_critic = loss_critic * self.value_loss_coef

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'Loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, mus_old, sigmas_old, writer):
        self.optim_actor.zero_grad()
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mus = self.actor.means
        sigmas = (self.actor.log_stds.exp()).repeat(mus.shape[0], 1)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        loss_actor = loss_actor * self.surrogate_loss_coef
        if self.auto_lr:
            # desired_kl: 0.01
            with torch.inference_mode():
                kl = torch.sum(torch.log(sigmas / sigmas_old + 1.e-5) +
                               (torch.square(sigmas_old) + torch.square(mus_old - mus)) /
                               (2.0 * torch.square(sigmas)) - 0.5, axis=-1)

                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.lr_actor = max(1e-5, self.lr_actor / 1.5)
                    self.lr_critic = max(1e-5, self.lr_critic / 1.5)
                    self.lr_disc = max(1e-5, self.lr_disc / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.lr_actor = min(1e-2, self.lr_actor * 1.5)
                    self.lr_critic = min(1e-2, self.lr_critic * 1.5)
                    self.lr_disc = min(1e-2, self.lr_disc * 1.5)

                for param_group in self.optim_actor.param_groups:
                    param_group['lr'] = self.lr_actor
                for param_group in self.optim_critic.param_groups:
                    param_group['lr'] = self.lr_critic
                for param_group in self.optim_d.param_groups:
                    param_group['lr'] = self.lr_disc

        loss = loss_actor  # + b_loss * 0 - self.entropy_coef * entropy * 0

        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'Loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'Loss/entropy', entropy.item(), self.learning_steps)
            writer.add_scalar(
                'Loss/learning_rate', self.lr_actor, self.learning_steps)

    def lr_decay(self, total_steps, writer):
        lr_a_now = max(1e-5, self.lr_actor * (1 - total_steps / self.max_steps))
        lr_c_now = max(1e-5, self.lr_critic * (1 - total_steps / self.max_steps))
        lr_d_now = max(1e-5, self.lr_disc * (1 - total_steps / self.max_steps))
        for p in self.optim_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optim_critic.param_groups:
            p['lr'] = lr_c_now
        for p in self.optim_d.param_groups:
            p['lr'] = lr_d_now

        writer.add_scalar(
            'Loss/learning_rate', lr_a_now, self.learning_steps)

    def calculate_gae(self, values, rewards, dones, tm_dones, next_values, gamma, lambd):
        """
            Calculate the advantage using GAE
            'tm_dones=True' means dead or win, there is no next state s'
            'dones=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps).
            When calculating the adv, if dones=True, gae=0
            Reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/ppo_continuous.py
        """
        with torch.no_grad():
            # Calculate TD errors.
            deltas = rewards + gamma * next_values * (1 - tm_dones) - values
            # Initialize gae.
            gaes = torch.empty_like(rewards)

            # Calculate gae recursively from behind.
            gaes[-1] = deltas[-1]
            for t in reversed(range(rewards.size(0) - 1)):
                gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

            v_target = gaes + values
            if self.use_adv_norm:
                gaes = (gaes - gaes.mean()) / (gaes.std(dim=0) + 1e-8)

        return v_target, gaes

    def save_models(self, save_dir):
        pass
