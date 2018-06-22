import logging

import torch.nn as nn
import torch.nn.functional as F

from lwrl.models.networks.layers import layer_factory
import lwrl.utils.logging as L


class Network(nn.Module):
    def __init__(self, network_spec):
        super().__init__()
        self.layers = [
            layer_factory(**layer_spec) for layer_spec in network_spec
        ]
        self.network = nn.Sequential(*self.layers)

        logger = logging.getLogger(__name__)
        logger.info(L.begin_section('Model'))
        logger.info(self.network)

    def forward(self, x):
        return self.network(x)

    def output_size(self):
        return self.layers[-1].output_size()
