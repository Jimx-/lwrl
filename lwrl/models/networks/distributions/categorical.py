import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from lwrl.models.networks.distributions import DistributionNetwork


class CategoricalDistributionNetwork(DistributionNetwork):
    """
    Categorical distribution for discrete actions
    """

    def __init__(self, network_spec, num_actions):
        self.num_actions = num_actions
        super().__init__(network_spec)

        self.logits = nn.Linear(self.output_size()[0], num_actions)

    def forward(self, x):
        logits = self.logits(super().forward(x))
        distribution = Categorical(logits=logits)
        return distribution

    def sample(self, distribution, deterministic):
        if deterministic:
            logits = distribution.logits
            _, definite = logits.max(-1)
            return definite.item()

        return distribution.sample().item()

    def log_prob(self, distribution, action):
        return distribution.log_prob(action)
