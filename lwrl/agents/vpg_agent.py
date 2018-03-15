from lwrl.agents import BatchAgent
from lwrl.models import PGLogProbModel


class VPGAgent(BatchAgent):
    """
    Vanilla Policy Gradient Agent
    """
    def __init__(
            self,
            state_spec,
            action_spec,
            network_spec,
            optimizer,
            discount_factor,
            history_length,
            batch_size=1000,
            exploration_schedule=None,
            keep_last_timestep=True,
            saver_spec=None,
            state_preprocess_pipeline=None
    ):
        self.network_spec = network_spec
        self.exploration_schedule = exploration_schedule
        self.optimizer = optimizer

        self.global_step = 0

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
        return PGLogProbModel(
            state_spec=self.state_spec,
            action_spec=self.action_spec,
            network_spec=self.network_spec,
            exploration_schedule=self.exploration_schedule,
            optimizer=self.optimizer,
            saver_spec=self.saver_spec,
            discount_factor=self.discount_factor,
            state_preprocess_pipeline=self.state_preprocess_pipeline
        )
