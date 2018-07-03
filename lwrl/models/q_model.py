import random

import numpy as np
import torch

import lwrl.utils.th_helper as H
from lwrl.models import Model


class QModel(Model):
    def __init__(self,
                 state_spec,
                 action_spec,
                 network_cls,
                 network_spec,
                 exploration_schedule,
                 optimizer,
                 saver_spec,
                 discount_factor,
                 clip_error,
                 update_target_freq,
                 double_q_learning=False,
                 state_preprocess_pipeline=None):
        self.network_cls = network_cls
        self.network_spec = network_spec

        self.clip_error = clip_error
        self.update_target_freq = update_target_freq
        self.double_q_learning = double_q_learning

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            exploration_schedule=exploration_schedule,
            optimizer=optimizer,
            saver_spec=saver_spec,
            discount_factor=discount_factor,
            state_preprocess_pipeline=state_preprocess_pipeline,
        )

    def init_model(self):
        super().init_model()

        self.q_network = self.network_cls(
            self.network_spec, self.action_spec['num_actions']).type(
                H.float_tensor)
        self.target_network = self.network_cls(
            self.network_spec, self.action_spec['num_actions']).type(
                H.float_tensor)

        self.optimizer = self.optimizer_builder(self.q_network.parameters())

    def predict(self, obs):
        obs = self.preprocess_state(
            torch.from_numpy(obs).type(H.float_tensor).unsqueeze(0))
        return self.q_network(H.Variable(obs)).data

    def get_action(self, obs, random_action, update):
        with torch.no_grad():
            return self.q_network(H.Variable(obs)).data.max(1)[1].cpu()[0]

    def update(self, obs_batch, action_batch, reward_batch, next_obs_batch,
               done_mask):
        obs_batch = self.preprocess_state(
            H.Variable(torch.from_numpy(obs_batch).type(H.float_tensor)))
        next_obs_batch = self.preprocess_state(
            H.Variable(torch.from_numpy(next_obs_batch).type(H.float_tensor)))
        action_batch = H.Variable(torch.from_numpy(action_batch).long())
        reward_batch = H.Variable(torch.from_numpy(reward_batch))
        neg_done_mask = H.Variable(
            torch.from_numpy(1.0 - done_mask).type(H.float_tensor))

        if H.use_cuda:
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        # minimize (Q(s, a) - (r + gamma * max Q(s', a'; w'))^2
        q_values = self.q_network(obs_batch).gather(
            1, action_batch.unsqueeze(1))  # Q(s, a; w)

        if self.double_q_learning:
            _, next_state_actions = self.q_network(next_obs_batch).max(
                1, keepdim=True)
            next_max_q_values = self.target_network(next_obs_batch).gather(
                1, next_state_actions).squeeze().detach()
        else:
            next_max_q_values = self.target_network(
                next_obs_batch).detach().max(1)[0]  # max Q(s', a'; w')

        td_error = self.calculate_td_error(q_values, next_max_q_values,
                                           reward_batch, neg_done_mask)
        clipped_td_error = td_error.clamp(-self.clip_error, self.clip_error)
        grad = clipped_td_error * -1.0

        self.optimizer.zero_grad()
        q_values.backward(grad.data)

        self.optimizer.step()
        self.num_updates += 1

        # target networks <- online networks
        if self.num_updates % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def calculate_td_error(self, q_values, next_q_values, rewards,
                           neg_done_mask):
        next_q_values = neg_done_mask * next_q_values
        target_q_values = rewards + self.discount_factor * next_q_values  # r + gamma * max Q(s', a'; w')
        return target_q_values.unsqueeze(1) - q_values

    def save(self, timestep):
        self.saver.save({
            'global_step': timestep,
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, timestep)

    def restore(self):
        checkpoint = self.saver.restore()
        self.global_step = checkpoint['global_step']
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
