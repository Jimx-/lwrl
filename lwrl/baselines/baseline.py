import torch


class Baseline:
    """
    Baseline value functions for policy gradient
    """

    def predict(self, obs_batch):
        raise NotImplementedError

    def loss(self, obs_batch, reward):
        prediction = self.predict(obs_batch)
        delta = prediction - reward
        return torch.norm(delta, p=2, dim=-1)
