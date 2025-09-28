import os
import numpy as np
import torch


class RolloutBuffer:
    # TODO: state and action are list
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size

        self.states = torch.empty((self.buffer_size, *state_shape), dtype=torch.float, device=device)
        # self.states_gail = torch.empty((self.buffer_size, *state_gail_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((self.buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((self.buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((self.buffer_size, 1), dtype=torch.int, device=device)
        self.tm_dones = torch.empty((self.buffer_size, 1), dtype=torch.int, device=device)
        self.log_pis = torch.empty((self.buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((self.buffer_size, *state_shape), dtype=torch.float, device=device)
        # self.next_states_gail = torch.empty((self.buffer_size, *state_gail_shape), dtype=torch.float, device=device)
        self.means = torch.empty((self.buffer_size, *action_shape), dtype=torch.float, device=device)
        self.stds = torch.empty((self.buffer_size, *action_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, tm_dones, log_pi, next_state, next_state_gail, means, stds):
        self.states[self._p].copy_(state)
        # self.states_gail[self._p].copy_(state_gail)
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = int(done)
        self.tm_dones[self._p] = int(tm_dones)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        # self.next_states_gail[self._p].copy_(torch.from_numpy(next_state_gail))
        self.means[self._p].copy_(torch.from_numpy(means))
        self.stds[self._p].copy_(torch.from_numpy(stds))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        idxes = slice(0, self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.tm_dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes],
            self.means[idxes],
            self.stds[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.tm_dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes],
            self.means[idxes],
            self.stds[idxes]
        )

    def clear(self):
        self.states[:, :] = 0
        self.actions[:, :] = 0
        self.rewards[:, :] = 0
        self.dones[:, :] = 0
        self.tm_dones[:, :] = 0
        self.log_pis[:, :] = 0
        self.next_states[:, :] = 0
        self.means[:, :] = 0
        self.stds[:, :] = 0
