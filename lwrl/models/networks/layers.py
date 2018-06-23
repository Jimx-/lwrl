import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.activation = getattr(
            F, activation) if activation is not None else None

    def forward(self, x):
        return self.activation(self.conv(x))


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.output = out_features
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = getattr(
            F, activation) if activation is not None else None

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.fc(x))
        return self.fc(x)

    def output_size(self):
        return (self.output, )


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


layer_dict = dict(conv2d=Conv2d, flatten=Flatten, dense=Dense)


def layer_factory(type, *args, **kwargs):
    return layer_dict[type](*args, **kwargs)
