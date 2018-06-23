import torch.nn as nn

from lwrl.baselines import Baseline
from lwrl.models.networks import Network


class NetworkBaseline(Network, Baseline):
    def __init__(self, network_spec):
        super().__init__(network_spec)

        self.linear = nn.Linear(
            in_features=self.output_size()[0], out_features=1)

    def predict(self, obs_batch):
        phi = self.network(obs_batch)
        prediction = self.linear(phi)
        return prediction.squeeze(1)
