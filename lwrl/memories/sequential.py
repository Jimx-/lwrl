import random
import torch

import numpy as np
import scipy.io as sio

from lwrl.memories import Memory


class SequentialMemory(Memory):
    def __init__(self, max_length, history_length=1):
        super().__init__()

        self.max_length = max_length
        self.obs_buffer = None
        self.history_length = history_length
        self.current = 0
        self.count = 0
        self.prestates = None

    def add(self, obs, action, reward, done):
        if len(obs.shape) > 1:
            obs = np.transpose(obs, (2, 0, 1))

        if self.obs_buffer is None:
            # infer action type
            try:
                action_type = action.dtype
                if action_type is torch.float32:
                    action_type = np.float32
            except AttributeError:
                action_type = type(action)
            self.action_buffer = np.empty(self.max_length, dtype=action_type)

            self.reward_buffer = np.empty(self.max_length, dtype=np.float32)
            self.obs_buffer = np.empty(
                (self.max_length, *obs.shape), dtype=obs.dtype)
            self.terminal_buffer = np.empty(self.max_length, dtype=bool)
            self.dim = obs.shape

        self.action_buffer[self.current] = action
        self.reward_buffer[self.current] = reward
        self.obs_buffer[self.current, ...] = obs
        self.terminal_buffer[self.current] = done
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.max_length

    def _get_obs(self, index):
        index = index % self.count
        if index >= self.history_length - 1:
            return self.obs_buffer[(index - self.history_length + 1):(
                index + 1), ...]
        else:
            indices = [(index - i) % self.count
                       for i in reversed(range(self.history_length))]
            return self.obs_buffer[indices]

    def sample(self, size):
        if self.prestates is None:
            state_shape = (size, self.history_length, *self.dim)
            self.prestates = np.empty(state_shape, dtype=self.obs_buffer.dtype)
            self.poststates = np.empty(
                state_shape, dtype=self.obs_buffer.dtype)

        indices = []
        i = 0
        while i < size:
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if index >= self.current and index - self.history_length < self.current:
                    continue
                if self.terminal_buffer[(
                        index - self.history_length):index].any():
                    continue
                break

            self.prestates[i, ...] = self._get_obs(index - 1)
            self.poststates[i, ...] = self._get_obs(index)

            indices.append(index)

            i += 1

        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.terminal_buffer[indices]

        if len(self.dim) == 1:
            shape = (size, self.dim[0])
        else:
            shape = (size, -1, self.dim[1], self.dim[2])
        return (self.prestates.reshape(shape), actions, rewards,
                self.poststates.reshape(shape), dones)

    def size(self):
        return self.count

    def save(self, filename):
        sio.savemat(
            filename,
            mdict=dict(
                observations=self.obs_buffer,
                actions=self.action_buffer,
                rewards=self.reward_buffer,
                terminals=self.terminal_buffer))
