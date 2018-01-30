import random
import torch

from lwrl.models import Model
import lwrl.utils.th_helper as H


class QModel(Model):
    def __init__(
            self,
            state_space,
            action_space,
            network_cls,
            network_spec,
            exploration_schedule,
            optimizer_spec,
            saver_spec,
            discount_factor,
            clip_error,
            update_target_freq,
            double_q_learning=False
    ):
        self.network_cls = network_cls
        self.network_spec = network_spec

        self.clip_error = clip_error
        self.update_target_freq = update_target_freq
        self.double_q_learning = double_q_learning

        super().__init__(state_space, action_space, exploration_schedule, optimizer_spec, saver_spec, discount_factor)

    def init_model(self):
        super().init_model()

        self.q_network = self.network_cls(self.network_spec, self.num_actions).type(H.float_tensor)
        self.target_network = self.network_cls(self.network_spec, self.num_actions).type(H.float_tensor)

        self.optimizer = self.optimizer_builder(self.q_network.parameters())

    def act(self, obs, random_action=True):
        # epsilon-greedy action selection
        eps = self.exploration_schedule.value(self.timestep)
        if not random_action:
            eps = 0.05
        if random.random() < eps:
            return self.action_space.sample()
        else:
            obs = torch.from_numpy(obs).type(H.float_tensor).unsqueeze(0) / 255.0
            with torch.no_grad():
                return self.q_network(H.Variable(obs)).data.max(1)[1].cpu()[0]

    def update(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_mask):
        obs_batch = H.Variable(torch.from_numpy(obs_batch).type(H.float_tensor) / 255.0)
        next_obs_batch = H.Variable(torch.from_numpy(next_obs_batch).type(H.float_tensor) / 255.0)
        action_batch = H.Variable(torch.from_numpy(action_batch).long())
        reward_batch = H.Variable(torch.from_numpy(reward_batch))
        neg_done_mask = H.Variable(torch.from_numpy(1.0 - done_mask).type(H.float_tensor))

        if H.use_cuda:
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        # minimize (Q(s, a) - (r + gamma * max Q(s', a'; w'))^2
        q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)) # Q(s, a; w)
        next_max_q_values = self.target_network(next_obs_batch).detach().max(1)[0] # max Q(s', a'; w')
        next_q_values = neg_done_mask * next_max_q_values
        target_q_values = reward_batch + self.discount_factor * next_q_values # r + gamma * max Q(s', a'; w')
        td_error = target_q_values.unsqueeze(1) - q_values
        clipped_td_error = td_error.clamp(-self.clip_error, self.clip_error)
        grad = clipped_td_error * -1.0

        self.optimizer.zero_grad()
        q_values.backward(grad.data)

        self.optimizer.step()
        self.num_updates += 1

        # target networks <- online networks
        if self.num_updates % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

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
