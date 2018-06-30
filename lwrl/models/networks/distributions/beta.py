import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

import numpy as np

import lwrl.utils.th_helper as H
from lwrl.models.networks.distributions import DistributionNetwork


class BetaDistributionNetwork(DistributionNetwork):
    def _create_distribution(self,
                             feature_dim,
                             shape=(),
                             min_value=None,
                             max_value=None):
        self.shape = shape
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        self.log_eps = np.log(1e-6)

        action_size = int(np.prod(shape))
        # mu ~ Beta(mu | alpha, beta)
        self.alpha = nn.Linear(feature_dim, action_size)
        self.beta = nn.Linear(feature_dim, action_size)

    def _forward_distribution(self, phi):
        alpha = self.alpha(phi).clamp(min=self.log_eps, max=-self.log_eps)
        alpha = torch.log(torch.exp(alpha) + 1.) + 1.

        beta = self.beta(phi).clamp(min=self.log_eps, max=-self.log_eps)
        beta = torch.log(torch.exp(beta) + 1.) + 1.

        alpha = alpha.view(-1, *self.shape)
        beta = beta.view(-1, *self.shape)

        #alpha_beta = torch.clamp(alpha + beta, min=1e-6)
        dist = Beta(concentration0=beta.cpu(), concentration1=alpha.cpu())

        return alpha, beta, dist

    def sample(self, dist_params, deterministic):
        alpha, _, dist = dist_params

        if deterministic:
            # use mean as action
            samples = dist.mean
        else:
            samples = dist.rsample(alpha.size())

        actions = self.min_value + (self.max_value - self.min_value) * samples
        return actions.view(alpha.size()).type(H.float_tensor)

    def log_prob(self, dist_params, actions):
        _, _, dist = dist_params
        actions = (actions.data - self.min_value) / (
            self.max_value - self.min_value)
        with torch.no_grad():
            actions = H.Variable(torch.clamp(actions, max=1 - 1e-6)).cpu()
        log_prob = dist.log_prob(actions).type(H.float_tensor)
        return log_prob
