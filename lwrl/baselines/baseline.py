import torch.nn.functional as F


class Baseline:
    """
    Baseline value functions for policy gradient
    """

    def predict(self, obs_batch):
        raise NotImplementedError

    def reference(self, obs_batch, reward):
        return None

    def loss(self, obs_batch, reward, reference=None):
        prediction = self.predict(obs_batch)
        return F.mse_loss(prediction, reward)
