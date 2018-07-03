import torch
from lwrl.models import PGModel


class PGProbRatioModel(PGModel):
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
                 subsampling_fraction=0.1,
                 optimization_steps=50,
                 likelihood_ratio_clipping=None):
        self.subsampling_fraction = subsampling_fraction
        self.optimization_steps = optimization_steps

        self.likelihood_ratio_clipping = likelihood_ratio_clipping

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            network_spec=network_spec,
            exploration_schedule=exploration_schedule,
            optimizer=optimizer,
            saver_spec=saver_spec,
            discount_factor=discount_factor,
            state_preprocess_pipeline=state_preprocess_pipeline,
            baseline_mode=baseline_mode,
            baseline_spec=baseline_spec,
            baseline_optimizer=baseline_optimizer,
            subsampling_fraction=subsampling_fraction,
            optimization_steps=optimization_steps)

    def calculate_reference(self, obs_batch, action_batch, reward_batch,
                            next_obs_batch, neg_done_mask):
        dist_params = self.network(obs_batch)
        old_log_prob = self.network.log_prob(dist_params,
                                             action_batch).detach()
        return old_log_prob

    def calculate_loss(self,
                       obs_batch,
                       action_batch,
                       reward_batch,
                       next_obs_batch,
                       neg_done_mask,
                       reference=None):

        dist_params = self.network(obs_batch)
        log_prob = self.network.log_prob(dist_params, action_batch)

        if reference is None:
            old_log_prob = log_prob.detach()
        else:
            old_log_prob = reference

        prob_ratio = torch.exp(log_prob - old_log_prob)

        if self.likelihood_ratio_clipping is None:
            return (-prob_ratio * reward_batch).mean()
        else:
            clipped_prob_ratio = torch.clamp(
                prob_ratio, 1.0 / (1.0 + self.likelihood_ratio_clipping),
                1.0 + self.likelihood_ratio_clipping)

            return -torch.min(prob_ratio * reward_batch,
                              clipped_prob_ratio * reward_batch).mean()
