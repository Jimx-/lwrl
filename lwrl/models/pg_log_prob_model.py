from lwrl.models import PGModel


class PGLogProbModel(PGModel):
    def calculate_loss(self,
                       obs_batch,
                       action_batch,
                       reward_batch,
                       next_obs_batch,
                       neg_done_mask,
                       reference=None):
        dist_params = self.network(obs_batch)
        log_prob = self.network.log_prob(dist_params, action_batch)

        return (-log_prob * reward_batch).mean()
