import torch.nn as nn
import torch.nn.functional as F

from lwrl.models.networks import Network


class DeepQNetwork(nn.Module):
    def __init__(self, network_spec, num_actions=18):
        super().__init__()
        self.network = Network(network_spec)
        out_features = self.network.output_size()[0]
        self._init_q_head(out_features, num_actions)

    def _init_q_head(self, feature_dim, num_actions):
        self.fc5 = nn.Linear(feature_dim, num_actions)

    def forward(self, x):
        x = self.network(x)
        return self._forward_q(x)

    def _forward_q(self, phi):
        return self.fc5(phi)


class DuelingDQN(DeepQNetwork):
    def _init_q_head(self, feature_dim, num_actions):
        self.fc_value = nn.Linear(feature_dim, 1)
        self.fc_advantage = nn.Linear(feature_dim, num_actions)

    def _forward_q(self, phi):
        value = self.fc_value(phi)
        advantage = self.fc_advantage(phi)
        return value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
