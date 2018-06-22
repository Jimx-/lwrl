from lwrl.models.networks import Network


class DistributionNetwork(Network):
    def __init__(self, network_spec):
        super().__init__(network_spec)

    def sample(self, dist_params, deterministic):
        raise NotImplementedError

    def log_prob(self, dist_params, action):
        raise NotImplementedError
