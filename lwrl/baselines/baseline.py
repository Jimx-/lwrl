import torch.nn.functional as F


class Baseline:
    """
    Baseline value functions for policy gradient
    """

    def predict(self, obs_batch):
        raise NotImplementedError

    def loss(self, obs_batch, reward):
        prediction = self.predict(obs_batch)
        return (prediction - reward).pow(2).mean()
