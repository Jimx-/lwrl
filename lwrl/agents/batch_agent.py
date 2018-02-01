import numpy as np

from lwrl.agents import LearningAgent
from lwrl.utils.history import History


class BatchAgent(LearningAgent):
    def __init__(
            self,
            state_spec,
            action_spec,
            discount_factor,
            optimizer=None,
            history_length=1,
            batch_size=1000,
            keep_last_timestep=True,
            state_preprocess_pipeline=None
    ):
        self.history_length = history_length
        self.history = History(self.history_length)

        self.batch_size = batch_size
        self.keep_last_timestep = keep_last_timestep

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            discount_factor=discount_factor,
            optimizer=optimizer,
            state_preprocess_pipeline=state_preprocess_pipeline
        )

        self.obs_batch = None
        self.action_batch = None
        self.reward_batch = None
        self.terminal_batch = None
        self.next_obs_batch = None
        self.dim = None

        self.reset_batch()

    def act(self, obs, random_action=True):
        # fill in history on the beginning of an episode
        if self.history.empty():
            for _ in range(self.history_length):
                self.history.add(obs)

        obs = self.history.get()
        if self.obs_batch is None:
            self.obs_batch = np.empty((self.batch_size, *obs.shape), dtype=obs.dtype)
            self.obs_batch[0, ...] = obs

        if self.batch_count == 0 and not self.keep_last_timestep:
            self.obs_batch[0, ...] = obs

        return super().act(obs, random_action)

    def observe(self, obs, action, reward, done, training=False):
        super().observe(obs, action, reward, done, training)

        self.history.add(obs)

        obs = self.history.get()
        if self.next_obs_batch is None:
            self.action_batch = np.empty(self.batch_size, dtype=type(action))
            self.reward_batch = np.empty(self.batch_size, dtype=np.float32)
            self.next_obs_batch = np.empty((self.batch_size, *obs.shape), dtype=obs.dtype)
            self.terminal_batch = np.empty(self.batch_size, dtype=bool)
            self.dim = obs.shape

        if training:
            self.next_obs_batch[self.batch_count, ...] = obs
            self.action_batch[self.batch_count] = action
            self.reward_batch[self.batch_count] = reward
            self.terminal_batch[self.batch_count] = done

            if self.batch_count > 0:
                self.obs_batch[self.batch_count, ...] = self.next_obs_batch[self.batch_count - 1, ...]
            self.batch_count += 1

        if training and self.batch_count == self.batch_size:
            self.model.update(self.obs_batch, self.action_batch, self.reward_batch, self.next_obs_batch, self.terminal_batch)
            self.batch_count = 0

    def reset(self):
        self.history.reset()

    def reset_batch(self):
        self.batch_count = 0
