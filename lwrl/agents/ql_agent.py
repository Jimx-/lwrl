from lwrl.agents import MemoryAgent
from lwrl.models import QModel
from lwrl.models.networks import DeepQNetwork, DuelingDQN


class BaseQLearningAgent(MemoryAgent):
    def __init__(
            self,
            state_spec,
            action_spec,
            network_cls,
            network_spec,
            optimizer,
            memory,
            exploration_schedule,
            discount_factor,
            clip_error,
            update_target_freq,
            history_length,
            learning_starts,
            train_freq=1,
            batch_size=32,
            double_q_learning=False,
            saver_spec=None,
            state_preprocess_pipeline=None
    ):

        self.network_cls = network_cls
        self.network_spec = network_spec
        self.exploration_schedule = exploration_schedule
        self.optimizer= optimizer

        self.global_step = 0

        self.clip_error = clip_error
        self.update_target_freq = update_target_freq
        self.double_q_learning = double_q_learning

        self.saver_spec = saver_spec

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            discount_factor=discount_factor,
            optimizer=optimizer,
            memory=memory,
            history_length=history_length,
            batch_size=batch_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            state_preprocess_pipeline=state_preprocess_pipeline
        )

    def init_model(self):
        return QModel(
            state_spec=self.state_spec,
            action_spec=self.action_spec,
            network_cls=self.network_cls,
            network_spec=self.network_spec,
            exploration_schedule=self.exploration_schedule,
            optimizer=self.optimizer,
            saver_spec=self.saver_spec,
            discount_factor=self.discount_factor,
            clip_error=self.clip_error,
            update_target_freq=self.update_target_freq,
            double_q_learning=self.double_q_learning,
            state_preprocess_pipeline=self.state_preprocess_pipeline
        )


class QLearningAgent(BaseQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(network_cls=DeepQNetwork, *args, **kwargs)


class DuelingQLearningAgent(BaseQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(network_cls=DuelingDQN, *args, **kwargs)
