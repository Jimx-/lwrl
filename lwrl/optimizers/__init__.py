from lwrl.optimizers.optimizer import Optimizer
from lwrl.optimizers.pth_optimizer import PytorchOptimizer
from lwrl.optimizers.meta_optimizer import MetaOptimizer
from lwrl.optimizers.multistep_optimizer import MultistepOptimizer

optimizer_dict = dict(
    Adam=
    lambda parameters, **kwargs: PytorchOptimizer('Adam', parameters, **kwargs),
    RMSprop=
    lambda parameters, **kwargs: PytorchOptimizer('RMSprop', parameters, **kwargs),
    multistep=MultistepOptimizer,
)


def optimizer_factory(type, parameters, **kwargs):
    return optimizer_dict[type](parameters, **kwargs)
