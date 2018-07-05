from lwrl.optimizers import Optimizer
import lwrl.utils.th_helper as H


class PytorchOptimizer(Optimizer):
    def __init__(self, optimizer, parameters, **kwargs):
        self.optimizer_type = optimizer
        self.optimizer = H.optimizer_dict[self.optimizer_type](parameters,
                                                               **kwargs)

    def step(self, fn_loss, arguments=None, grad=None, **kwargs):
        if arguments is not None:
            loss = fn_loss(**arguments)
        else:
            loss = fn_loss

        self.optimizer.zero_grad()
        loss.backward(grad)
        self.optimizer.step()
