from lwrl.optimizers import MetaOptimizer


class MultistepOptimizer(MetaOptimizer):
    def __init__(self, parameters, optimizer, num_steps=10):
        self.num_steps = num_steps
        super().__init__(parameters, optimizer)

    def step(self, fn_loss, arguments, fn_reference, **kwargs):
        arguments['reference'] = fn_reference(**arguments)

        for _ in range(self.num_steps):
            self.optimizer.step(fn_loss=fn_loss, arguments=arguments)
