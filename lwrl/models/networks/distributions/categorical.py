import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

import lwrl.utils.th_helper as H
from lwrl.models.networks.distributions import DistributionNetwork


class CategoricalDistributionNetwork(DistributionNetwork):
    def _create_distribution(self, feature_dim, shape=(), num_actions=None):
        self.shape = shape
        self.num_actions = num_actions

        action_size = int(np.prod(shape) * num_actions)
        self.logits = nn.Linear(feature_dim, action_size)

    def _forward_distribution(self, phi):
        # softmax policy
        logits = self.logits(phi)
        logits = logits.view(-1, *self.shape, self.num_actions)

        state_values = logits.exp().sum(dim=-1).log()
        # probability of action is proportional to exponentiated weight
        probs = torch.clamp(F.softmax(logits, dim=-1), min=1e-6)
        # log probability
        logits = torch.log(probs)
        dist = Categorical(logits=logits)

        return logits, probs, state_values, dist

    def sample(self, dist_params, deterministic):
        logits, _, _, dist = dist_params

        if deterministic:
            _, indices = logits.max(-1)
            actions = indices
        else:
            # sample from the distribution
            actions = dist.sample(self.shape)

        return actions

    def log_prob(self, dist_params, actions):
        _, _, _, dist = dist_params
        return dist.log_prob(actions)

    def entropy(self, dist_params):
        _, _, _, dist = dist_params
        return dist.entropy()
