import torch

import lwrl.utils.th_helper as H
from lwrl.models import DistributionModel
from lwrl.baselines import baseline_factory
from lwrl.optimizers import optimizer_factory


class PGModel(DistributionModel):
    def __init__(self,
                 state_spec,
                 action_spec,
                 network_spec,
                 exploration_schedule,
                 optimizer,
                 saver_spec,
                 discount_factor,
                 state_preprocess_pipeline=None,
                 baseline_mode=None,
                 baseline_spec=None,
                 baseline_optimizer=None,
                 entropy_regularization=None,
                 gae_lambda=None):
        self.network_spec = network_spec

        self.baseline_mode = baseline_mode
        self.baseline_spec = baseline_spec
        self.baseline_optimizer_spec = baseline_optimizer

        assert gae_lambda is None or (0.0 <= gae_lambda <= 1.0
                                      and baseline_mode is not None)
        self.gae_lambda = gae_lambda

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            network_spec=network_spec,
            exploration_schedule=exploration_schedule,
            require_deterministic=False,
            optimizer=optimizer,
            saver_spec=saver_spec,
            discount_factor=discount_factor,
            state_preprocess_pipeline=state_preprocess_pipeline,
            entropy_regularization=entropy_regularization)

    def init_model(self):
        super().init_model()

        if self.baseline_spec is not None:
            self.baseline = baseline_factory(**self.baseline_spec).type(
                H.float_tensor)

        self.baseline_optimizer = None
        if self.baseline_optimizer_spec is not None:
            self.baseline_optimizer = optimizer_factory(
                self.baseline_optimizer_spec['type'],
                self.baseline.parameters(),
                **self.baseline_optimizer_spec['args'])

    def calculate_cumulative_rewards(self, rewards, neg_done_mask,
                                     discount_factor):
        batch_size = rewards.shape[0]
        for i in range(batch_size - 1, -1, -1):
            rewards[i] = rewards[i] + (rewards[i + 1]
                                       if i != batch_size - 1 else 0.0
                                       ) * neg_done_mask[i] * discount_factor
        return rewards

    def estimate_rewards(self, obs_batch, rewards, neg_done_mask):
        if self.baseline_mode is None:
            # calculate cumulative reward
            return self.calculate_cumulative_rewards(rewards, neg_done_mask,
                                                     self.discount_factor)

        elif self.baseline_mode == 'states':
            state_value = self.baseline.predict(obs_batch)

        if self.gae_lambda is None:
            rewards = self.calculate_cumulative_rewards(
                rewards, neg_done_mask, self.discount_factor)
            advantage = rewards - state_value
        else:
            next_state_value = torch.zeros_like(state_value)
            next_state_value[:-1] = next_state_value[1:]
            next_state_value[-1] = 0.0
            next_state_value = next_state_value * neg_done_mask
            td_residual = rewards + self.discount_factor * next_state_value - state_value
            gae_discount = self.discount_factor * self.gae_lambda
            advantage = self.calculate_cumulative_rewards(
                td_residual, neg_done_mask, gae_discount)

        return advantage

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

        estimated_rewards = self.estimate_rewards(obs_batch, reward_batch,
                                                  neg_done_mask)
        if self.baseline_optimizer is not None:
            estimated_rewards = estimated_rewards.detach()

        # optimize the actor model
        loss_arguments = dict(
            obs_batch=obs_batch,
            action_batch=action_batch,
            reward_batch=estimated_rewards,
            next_obs_batch=next_obs_batch,
            neg_done_mask=neg_done_mask,
        )

        self.optimizer.step(
            self.total_loss,
            loss_arguments,
            fn_reference=self.calculate_reference)

        # optimize the critic model
        if self.baseline_optimizer is not None:
            cumulative_rewards = self.calculate_cumulative_rewards(
                reward_batch, neg_done_mask, self.discount_factor)
            baseline_loss_arguments = dict(
                obs_batch=obs_batch,
                reward=cumulative_rewards,
            )

            self.baseline_optimizer.step(
                self.baseline.loss,
                baseline_loss_arguments,
                fn_reference=self.baseline.reference)

        self.num_updates += 1

    def total_loss(self,
                   obs_batch,
                   action_batch,
                   reward_batch,
                   next_obs_batch,
                   neg_done_mask,
                   reference=None):
        loss = self.calculate_loss(obs_batch, action_batch, reward_batch,
                                   next_obs_batch, neg_done_mask, reference)
        reg_loss = self.regularization_loss(obs_batch)
        return loss + reg_loss

    def calculate_loss(self,
                       obs_batch,
                       action_batch,
                       reward_batch,
                       next_obs_batch,
                       neg_done_mask,
                       reference=None):
        raise NotImplementedError

    def calculate_reference(self, obs_batch, action_batch, reward_batch,
                            next_obs_batch, neg_done_mask):
        return None
