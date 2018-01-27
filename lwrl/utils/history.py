import numpy as np

class History(object):
    def __init__(self, history_length=1):
        self.history_length = history_length
        self._empty = True
        self.history = None

    def add(self, obs):
        if len(obs.shape) > 1:
            obs = np.transpose(obs, (2, 0, 1))

        if self.history is None:
            self.history = np.zeros((self.history_length, *obs.shape), dtype=np.uint8)
            self.dim = obs.shape[1:]

        self.history[:-1] = self.history[1:]
        self.history[-1, ...] = obs
        self._empty = False

    def reset(self):
        self.history *= 0
        self._empty = True

    def get(self):
        return self.history.reshape((-1, *self.dim))

    def empty(self):
        return self._empty
