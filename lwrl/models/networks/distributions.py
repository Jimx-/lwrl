import torch
import torch.nn as nn
import torch.nn.functional as F

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

        return logits, probs, state_values

    def sample(self, dist_params, deterministic):
        logits, _, _ = dist_params

        if deterministic:
            # select action with maximal log probability
            _, indices = logits.data.max(-1)
            actions = indices
        else:
            # sample from the distribution
            uniform = H.Variable(H.float_tensor(logits.size()).uniform_(1e-6, 1-1e-6), volatile=True)
            gumbel = -torch.log(-torch.log(uniform))
            _, indices = (logits + gumbel).data.max(-1)
            actions = indices

        return actions

    def log_prob(self, dist_params, actions):
        logits, _, _ = dist_params
        return logits.gather(1, actions.unsqueeze(1))
