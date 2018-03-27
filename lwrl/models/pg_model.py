import torch

from lwrl.models import DistributionModel
import lwrl.utils.th_helper as H


class PGModel(DistributionModel):
    def __init__(
            self,
            state_spec,
            action_spec,
            network_spec,
            exploration_schedule,
            optimizer,
            saver_spec,
            discount_factor,
            state_preprocess_pipeline=None
    ):
        self.network_spec = network_spec

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            network_spec=network_spec,
            exploration_schedule=exploration_schedule,
            require_deterministic=False,
            optimizer=optimizer,
            saver_spec=saver_spec,
            discount_factor=discount_factor,
            state_preprocess_pipeline=state_preprocess_pipeline
        )

    def init_model(self):
        super().init_model()

    def estimate_rewards(self, rewards, neg_done_mask):
        batch_size = rewards.shape[0]
        for i in range(batch_size - 1, -1, -1):
            rewards[i] = rewards[i] + (rewards[i + 1] if i != batch_size - 1 else 0.0) * neg_done_mask[i] * self.discount_factor

        return rewards.unsqueeze(1)

    def update(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_mask):
        obs_batch = self.preprocess_state(H.Variable(torch.from_numpy(obs_batch).type(H.float_tensor)))
        next_obs_batch = self.preprocess_state(H.Variable(torch.from_numpy(next_obs_batch).type(H.float_tensor)))
        if self.action_spec['type'] == 'int':
            action_batch = H.Variable(torch.from_numpy(action_batch).long())
        else:
            action_batch = H.Variable(torch.from_numpy(action_batch))
        reward_batch = H.Variable(torch.from_numpy(reward_batch))
        neg_done_mask = H.Variable(torch.from_numpy(1.0 - done_mask).type(H.float_tensor))

        if H.use_cuda:
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        estimated_rewards = self.estimate_rewards(reward_batch, neg_done_mask)
        loss = self.calculate_loss(obs_batch, action_batch, estimated_rewards, next_obs_batch, neg_done_mask)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.num_updates += 1

    def calculate_loss(self, obs_batch, action_batch, reward_batch, next_obs_batch, neg_done_mask):
        raise NotImplementedError
