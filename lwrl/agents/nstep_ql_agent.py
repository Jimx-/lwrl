from lwrl.agents import BatchAgent
from lwrl.models import NStepQModel
from lwrl.models.networks import DeepQNetwork


class BaseNStepQLearningAgent(BatchAgent):
    def __init__(
            self,
            state_spec,
            action_spec,
            network_cls,
            network_spec,
            optimizer,
            exploration_schedule,
            discount_factor,
            clip_error,
            update_target_freq,
            history_length,
            batch_size=32,
            keep_last_timestep=True,
            double_q_learning=False,
            saver_spec=None,
            state_preprocess_pipeline=None
    ):
        self.network_cls = network_cls
        self.network_spec = network_spec
        self.exploration_schedule = exploration_schedule
        self.optimizer = optimizer

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
            history_length=history_length,
            batch_size=batch_size,
            keep_last_timestep=keep_last_timestep,
            state_preprocess_pipeline=state_preprocess_pipeline
        )

    def init_model(self):
        return NStepQModel(
            state_space=self.state_space,
            action_space=self.action_space,
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


class NStepQLearningAgent(BaseNStepQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(network_cls=DeepQNetwork, *args, **kwargs)
