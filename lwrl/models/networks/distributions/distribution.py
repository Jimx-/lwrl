import torch
import torch.nn as nn

import numpy as np

from lwrl.models.networks import Network
import lwrl.utils.th_helper as H


class DistributionNetwork(nn.Module):
    def __init__(self, network_spec, **kwargs):
        super().__init__()
        self.network = Network(network_spec)
        out_features = self.network.output_size()[0]
        self._create_distribution(out_features, **kwargs)

    def _create_distribution(self, feature_dim, **kwargs):
        pass

    def forward(self, x):
        x = self.network(x)
        return self._forward_distribution(x)

    def _forward_distribution(self, phi):
        raise NotImplementedError

    def sample(self, dist_params, deterministic):
        raise NotImplementedError

    def log_prob(self, dist_params, actions):
        raise NotImplementedError
