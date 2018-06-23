from lwrl.baselines.baseline import Baseline
from lwrl.baselines.network_baseline import NetworkBaseline
from lwrl.baselines.mlp_baseline import MLPBaseline

baseline_dict = dict(mlp=MLPBaseline, )


def baseline_factory(type, *args, **kwargs):
    return baseline_dict[type](*args, **kwargs)
