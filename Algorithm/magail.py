import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .disc import GAILDiscrim
from .ppo import PPO
from .utils import Normalizer


class MAGAIL(PPO):
    def __init__(self, buffer_exp, input_dim, device,
                 disc_coef=20.0, disc_grad_penalty=0.1, disc_logit_reg=0.25, disc_weight_decay=0.0005,
                 lr_disc=1e-3, epoch_disc=50, batch_size=1000, use_gail_norm=True
                 ):
        super().__init__(state_shape=input_dim, device=device)
        self.learning_steps = 0
        self.learning_steps_disc = 0

        self.disc = GAILDiscrim(input_dim=input_dim)
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_coef = disc_coef
        self.disc_logit_reg = disc_logit_reg
        self.disc_weight_decay = disc_weight_decay
        self.lr_disc = lr_disc
        self.epoch_disc = epoch_disc
        self.optim_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_disc)

        self.normalizer = None
        if use_gail_norm:
            self.normalizer = Normalizer(self.state_shape[0]*2)

        self.batch_size = batch_size
        self.buffer_exp = buffer_exp

    def update_disc(self, states, states_exp, writer):
        states_cp = states.clone()
        states_exp_cp = states_exp.clone()

        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states_cp)
        logits_exp = self.disc(states_exp_cp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = 0.5 * (loss_pi + loss_exp)

        # logit reg
        logit_weights = self.disc.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))

        # grad penalty
        sample_expert = states_exp_cp
        sample_expert.requires_grad = True
        disc = self.disc.linear(self.disc.trunk(sample_expert))
        ones = torch.ones(disc.size(), device=disc.device)
        disc_demo_grad = torch.autograd.grad(disc, sample_expert,
                                             grad_outputs=ones,
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        grad_pen_loss = torch.mean(disc_demo_grad)

        # weight decay
        disc_weights = self.disc.get_disc_weights()
        disc_weights = torch.cat(disc_weights, dim=-1)
        disc_weight_decay = torch.sum(torch.square(disc_weights))

        loss = self.disc_coef * loss_disc + self.disc_grad_penalty * grad_pen_loss + \
               self.disc_logit_reg * disc_logit_loss + self.disc_weight_decay * disc_weight_decay

        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('Loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()

            writer.add_scalar('Acc/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('Acc/acc_exp', acc_exp, self.learning_steps)

    def update(self, writer, total_steps):
        self.learning_steps += 1
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy trajectories.
            samples_policy = self.buffer.sample(self.batch_size)
            states, next_states = samples_policy[1], samples_policy[-3]
            states = torch.cat([states, next_states], dim=-1)

            # Samples from expert demonstrations.
            samples_expert = self.buffer_exp.sample(self.batch_size)
            states_exp, next_states_exp = samples_expert[0], samples_expert[1]
            states_exp = torch.cat([states_exp, next_states_exp], dim=-1)

            if self.normalizer is not None:
                with torch.no_grad():
                    states = self.normalizer.normalize_torch(states, self.device)
                    states_exp = self.normalizer.normalize_torch(states_exp, self.device)

            # Update discriminator and us encoder.
            self.update_disc(states, states_exp, writer)

            # Calulates the running mean and std of a data stream
            if self.normalizer is not None:
                self.normalizer.update(states.cpu().numpy())
                self.normalizer.update(states_exp.cpu().numpy())

        states, actions, rewards, dones, tm_dones, log_pis, next_states, mus, sigmas = self.buffer.get()

        # Calculate rewards.
        rewards, rewards_t, rewards_i = self.disc.calculate_reward(states, next_states, rewards)

        writer.add_scalar('Reward/rewards', rewards_t.mean().item() + rewards_i.mean().item(),
                          self.learning_steps)
        writer.add_scalar('Reward/rewards_t', rewards_t.mean().item(), self.learning_steps)
        writer.add_scalar('Reward/rewards_i', rewards_i.mean().item(), self.learning_steps)

        # Update PPO using estimated rewards.
        self.update_ppo(states, actions, rewards, dones, tm_dones, log_pis, next_states, mus, sigmas, writer,
                        total_steps)
        self.buffer.clear()
        return rewards_t.mean().item() + rewards_i.mean().item()

    def save_models(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'disc': self.disc.state_dict(),
            'optim_actor': self.optim_actor.state_dict(),
            'optim_critic': self.optim_critic.state_dict(),
            'optim_d': self.optim_d.state_dict()
        }, os.path.join(path, 'model.pth'))

    def load_models(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location='cuda:0')
        self.actor.load_state_dict(loaded_dict['actor'])
        self.critic.load_state_dict(loaded_dict['critic'])
        self.disc.load_state_dict(loaded_dict['disc'])
        if load_optimizer:
            self.optim_actor.load_state_dict(loaded_dict['optim_actor'])
            self.optim_critic.load_state_dict(loaded_dict['optim_critic'])
            self.optim_d.load_state_dict(loaded_dict['optim_d'])
