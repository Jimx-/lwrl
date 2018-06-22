from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.beta import Beta

from lwrl.models.networks.distributions import DistributionNetwork


class BetaDistributionNetwork(DistributionNetwork):
    """
    Beta distribution for bounded continuous actions
    """

    def __init__(self, network_spec, shape, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.shape = shape
        action_size = reduce(lambda x, y: x * y, shape, 1)

        super().__init__(network_spec)

        self.alpha = nn.Linear(self.output_size()[0], action_size)
        self.beta = nn.Linear(self.output_size()[0], action_size)

    def forward(self, x):
        phi = super().forward(x)
        alpha = self.alpha(phi).cpu()
        beta = self.beta(phi).cpu()
        distribution = Beta(concentration1=alpha, concentration0=beta)
        return distribution

    def sample(self, distribution, deterministic):
        if deterministic:
            sampled = distribution.mean
        else:
            sampled = distribution.sample()

        result = self.min_value + (self.max_value - self.min_value) * sampled
        return result[0]

    def log_prob(self, distribution, action):
        return distribution.log_prob(action).unsqueeze(-1)
