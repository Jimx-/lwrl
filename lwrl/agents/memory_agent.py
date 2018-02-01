from lwrl.agents import LearningAgent
from lwrl.memories import get_replay_memory
from lwrl.utils.history import History


class MemoryAgent(LearningAgent):
    def __init__(
            self,
            state_spec,
            action_spec,
            discount_factor,
            optimizer=None,
            memory=None,
            history_length=1,
            batch_size=32,
            learning_starts=10000,
            train_freq=4,
            state_preprocess_pipeline=None
    ):
        self.history_length = history_length
        self.history = History(self.history_length)

        self.batch_size = batch_size

        self.learning_starts = learning_starts
        self.train_freq = train_freq

        if memory is None:
            memory = 'sequential'
        self.replay_memory = get_replay_memory(memory)

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            discount_factor=discount_factor,
            optimizer=optimizer,
            state_preprocess_pipeline=state_preprocess_pipeline
        )

    def act(self, obs, random_action=True):
        # fill in history on the beginning of an episode
        if self.history.empty():
            for _ in range(self.history_length):
                self.history.add(obs)

        return super().act(self.history.get(), random_action)

    def observe(self, obs, action, reward, done, training=False):
        super().observe(obs, action, reward, done, training)

        self.history.add(obs)

        if training:
            self.replay_memory.add(obs, action, reward, done)

        if training and self.timestep > self.learning_starts and self.timestep % self.train_freq == 0 and \
                        self.replay_memory.size() > self.batch_size:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_mask = \
                self.replay_memory.sample(self.batch_size)
            self.model.update(obs_batch, action_batch, reward_batch, next_obs_batch, done_mask)

    def reset(self):
        self.history.reset()
