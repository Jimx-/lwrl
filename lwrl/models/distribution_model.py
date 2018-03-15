from torch import nn

from lwrl.models import Model
from lwrl.models.networks import CategoricalDistributionNetwork
import lwrl.utils.th_helper as H


class DistributionModel(Model):
    def __init__(
            self,
            state_spec,
            action_spec,
            network_spec,
            exploration_schedule,
            require_deterministic,
            optimizer=None,
            saver_spec=None,
            discount_factor=0.99,
            state_preprocess_pipeline=None,
    ):
        self.network_spec = network_spec
        self.require_deterministic = require_deterministic

        super().__init__(
            state_spec,
            action_spec,
            exploration_schedule,
            optimizer,
            saver_spec,
            discount_factor,
            state_preprocess_pipeline,
        )

    def init_model(self):
        super().init_model()

        self.network = self.create_network(self.network_spec, self.action_spec).type(H.float_tensor)
        self.optimizer = self.optimizer_builder(self.network.parameters())

    def create_network(self, network_spec, action_spec):
        if action_spec['type'] == 'int':
            return CategoricalDistributionNetwork(network_spec, num_actions=action_spec['num_actions'])

    def get_action(self, obs, random_action):
        dist_param = self.network(H.Variable(obs, volatile=True))
        action = self.network.sample(dist_param, deterministic=(not random_action) or self.require_deterministic)
        return action[0]

    def save(self, timestep):
        pass

    def restore(self):
        pass
