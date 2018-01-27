import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self._init_q_head(512, num_actions)

    def _init_q_head(self, feature_dim, num_actions):
        self.fc5 = nn.Linear(feature_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
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
        return value.expand_as(advantage) + (advantage - advantage.mean(1).expand_as(advantage))
