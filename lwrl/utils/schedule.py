import numpy as np
import torch

import lwrl.utils.th_helper as H


class Schedule:
    def value(self, timestep, action_spec):
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, initial=1.0, final=0.1, steps=1000000):
        self.initial = initial
        self.final = final
        self.steps = steps

    def value(self, timestep, action_spec):
        frac = min(1.0, float(timestep) / self.steps)
        return self.initial + (self.final - self.initial) * frac


class OrnsteinUhlenbeckProcess(Schedule):
    def __init__(self, sigma=0.3, mu=0.0, theta=0.5):
        self.sigma = sigma
        self.mu = float(mu)
        self.theta = theta
        self.state = None

    def value(self, timestep, action_spec):
        normal_state = torch.randn(size=action_spec['shape'])

        if self.state is None:
            self.state = torch.full(action_spec['shape'], self.mu)

        self.state += self.theta * (
            self.mu - self.state) + self.sigma * normal_state

        return self.state.type(H.float_tensor)


def get_schedule(config):
    type_dict = {
        "linear": LinearSchedule,
        "ornstein_uhlenbeck": OrnsteinUhlenbeckProcess,
    }

    return type_dict[config['type']](**config['args'])
