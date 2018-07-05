from lwrl import optimizers
from lwrl.optimizers import Optimizer


class MetaOptimizer(Optimizer):
    def __init__(self, parameters, optimizer, **kwargs):
        self.optimizer = optimizers.optimizer_factory(
            optimizer['type'], parameters, **optimizer['args'])

        super().__init__(parameters)
