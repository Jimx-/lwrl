import torch
from torch import autograd

use_cuda = torch.cuda.is_available()


def get_tensor_type(name):
    if use_cuda:
        return getattr(torch.cuda, name)
    return getattr(torch, name)


float_tensor = get_tensor_type('FloatTensor')
long_tensor = get_tensor_type('LongTensor')


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if use_cuda:
            data = data.cuda()
        super().__init__(data, *args, **kwargs)


optimizer_dict = {'RMSprop': torch.optim.RMSprop, 'Adam': torch.optim.Adam}
