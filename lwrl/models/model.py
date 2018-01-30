import random
import torch

from lwrl.utils import schedule
import lwrl.utils.th_helper as H
from lwrl.utils.saver import Saver


class Model:
    def __init__(
            self,
            state_space,
            action_space,
            exploration_schedule,
            optimizer_spec=None,
            saver_spec=None,
            discount_factor=0.99
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.num_actions = action_space.n

        self.exploration_schedule = schedule.get_schedule(exploration_schedule)

        if optimizer_spec is None:
            optimizer_spec = {
                "type": "Adam",
                "args": {
                    "lr": 0.00025,
                }
            }

        self.optimizer_builder = lambda params: \
            H.optimizer_dict[optimizer_spec['type']](params, **optimizer_spec['args'])

        self.discount_factor = discount_factor

        self.saver = None
        if saver_spec is not None:
            self.saver = Saver(**saver_spec)

        self.setup()

    def setup(self):
        self.init_model()

    def init_model(self):
        self.timestep = 0
        self.num_updates = 0

    def act(self, obs, random_action=True):
        raise NotImplementedError()

    def observe(self, obs, action, reward, done):
        self.timestep += 1

    def update(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_mask):
        self.num_updates += 1

    def save(self, timestep):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()
