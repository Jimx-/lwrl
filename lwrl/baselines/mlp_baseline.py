from lwrl.baselines import NetworkBaseline


class MLPBaseline(NetworkBaseline):
    def __init__(self, size):
        network_spec = []

        for in_features, out_features in zip(size[:-1], size[1:]):
            network_spec.append(
                dict(
                    type='dense',
                    in_features=in_features,
                    out_features=out_features,
                    activation='relu'))

        super().__init__(network_spec)
