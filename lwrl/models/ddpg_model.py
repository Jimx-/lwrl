import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import lwrl.utils.th_helper as H
from lwrl.models import DistributionModel


class DDPGCriticModel(nn.Module):
    def __init__(self, state_spec, action_spec, hidden1=400, hidden2=300):
        super().__init__()

        state_shape = state_spec['shape']
        #action_shape = action_spec['shape']
        assert len(state_shape) == 1
        self.action_size = 1

        self.fc1 = nn.Linear(state_shape[0], hidden1)
        self.fc2 = nn.Linear(hidden1 + self.action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        nn.init.uniform_(self.fc3.weight)
        nn.init.uniform_(self.fc3.bias)

    def forward(self, s, a):
        a = a.view(-1, self.action_size)
        out = F.relu(self.fc1(s))
        out = F.relu(self.fc2(torch.cat([out, a], 1)))
        out = self.fc3(out)
        return out.squeeze()


class DDPGModel(DistributionModel):
    def __init__(self,
                 state_spec,
                 action_spec,
                 network_spec,
                 exploration_schedule,
                 optimizer,
                 saver_spec,
                 discount_factor,
                 update_target_freq,
                 update_target_weight,
                 critic_network_spec,
                 critic_optimizer,
                 state_preprocess_pipeline=None):
        self.network_spec = network_spec
        self.critic_network_spec = critic_network_spec
        self.critic_optimizer = critic_optimizer

        self.update_target_freq = update_target_freq
        self.update_target_weight = update_target_weight

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            network_spec=network_spec,
            exploration_schedule=exploration_schedule,
            optimizer=optimizer,
            saver_spec=saver_spec,
            discount_factor=discount_factor,
            state_preprocess_pipeline=state_preprocess_pipeline,
            require_deterministic=True)

    def init_model(self):
        super().init_model()

        self.target_network = self.create_network(
            self.network_spec, self.action_spec).type(H.float_tensor)

        hidden1 = self.critic_network_spec['hidden1']
        hidden2 = self.critic_network_spec['hidden2']
        self.critic_network = DDPGCriticModel(
            self.state_spec,
            self.action_spec,
            hidden1=hidden1,
            hidden2=hidden2).type(H.float_tensor)
        self.target_critic_network = DDPGCriticModel(
            self.state_spec,
            self.action_spec,
            hidden1=hidden1,
            hidden2=hidden2).type(H.float_tensor)
        self.critic_optimizer = H.optimizer_dict[self.critic_optimizer[
            'type']](self.critic_network.parameters(),
                     **self.critic_optimizer['args'])

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_critic_network.load_state_dict(
            self.critic_network.state_dict())

    def get_target_network_action(self, obs, random_action):
        with torch.no_grad():
            dist_param = self.target_network(H.Variable(obs))
        action = self.target_network.sample(
            dist_param,
            deterministic=(not random_action) or self.require_deterministic)
        return action

    def predict_target_q(self, obs_batch, action_batch, reward_batch,
                         neg_done_mask):
        q_value = self.target_critic_network(obs_batch, action_batch)

        return reward_batch + neg_done_mask * self.discount_factor * q_value

    def update_target_model(self, target_model, model):
        for target_param, param in zip(target_model.parameters(),
                                       model.parameters()):
            target_param.data.copy_(
                (1 - self.update_target_weight) * param.data +
                self.update_target_weight * target_param.data)

    def update(self, obs_batch, action_batch, reward_batch, next_obs_batch,
               done_mask):
        obs_batch = self.preprocess_state(
            H.Variable(torch.from_numpy(obs_batch).type(H.float_tensor)))
        next_obs_batch = self.preprocess_state(
            H.Variable(torch.from_numpy(next_obs_batch).type(H.float_tensor)))
        if self.action_spec['type'] == 'int':
            action_batch = H.Variable(torch.from_numpy(action_batch).long())
        else:
            action_batch = H.Variable(torch.from_numpy(action_batch))
        reward_batch = H.Variable(torch.from_numpy(reward_batch))
        neg_done_mask = H.Variable(
            torch.from_numpy(1.0 - done_mask).type(H.float_tensor))

        if H.use_cuda:
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        # predict action using target network
        next_target_actions = self.get_target_network_action(
            next_obs_batch, random_action=False)

        # predict Q values for next states
        next_q_values = self.predict_target_q(
            next_obs_batch, next_target_actions, reward_batch,
            neg_done_mask).detach()

        q_values = self.critic_network(obs_batch, action_batch)
        #critic_loss = (q_values - next_q_values).pow(2).mean()
        critic_loss = F.mse_loss(q_values, next_q_values)

        # update critic
        self.critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        predicted_actions = self.get_action(
            obs_batch, random_action=False, update=True)
        actor_loss = -self.critic_network(obs_batch, predicted_actions).mean()
        self.network.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        self.num_updates += 1

        # target networks <- online networks
        if self.num_updates % self.update_target_freq == 0:
            self.update_target_model(self.target_network, self.network)
            self.update_target_model(self.target_critic_network,
                                     self.critic_network)
