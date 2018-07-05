from lwrl.agents import BatchAgent
from lwrl.models import PGProbRatioModel


class PPOAgent(BatchAgent):
    """
    Proximal Policy Optimization Agent
    """

    def __init__(self,
                 state_spec,
                 action_spec,
                 network_spec,
                 step_optimizer,
                 discount_factor,
                 history_length,
                 batch_size=1000,
                 exploration_schedule=None,
                 keep_last_timestep=True,
                 saver_spec=None,
                 state_preprocess_pipeline=None,
                 baseline_mode=None,
                 baseline_spec=None,
                 baseline_optimizer=None,
                 subsampling_fraction=0.1,
                 optimization_steps=50,
                 likelihood_ratio_clipping=None):
        self.network_spec = network_spec
        self.exploration_schedule = exploration_schedule
        self.baseline_mode = baseline_mode
        self.baseline_spec = baseline_spec
        self.baseline_optimizer = baseline_optimizer
        self.likelihood_ratio_clipping = likelihood_ratio_clipping

        if step_optimizer is None:
            step_optimizer = dict(type='Adam', args=dict(lr=1e-3, ))

        optimizer = dict(
            type='multistep',
            args=dict(
                optimizer=step_optimizer,
                num_steps=optimization_steps,
            ))
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
            state_preprocess_pipeline=state_preprocess_pipeline)

    def init_model(self):
        return PGProbRatioModel(
            state_spec=self.state_spec,
            action_spec=self.action_spec,
            network_spec=self.network_spec,
            exploration_schedule=self.exploration_schedule,
            optimizer=self.optimizer,
            saver_spec=self.saver_spec,
            discount_factor=self.discount_factor,
            state_preprocess_pipeline=self.state_preprocess_pipeline,
            baseline_mode=self.baseline_mode,
            baseline_spec=self.baseline_spec,
            baseline_optimizer=self.baseline_optimizer,
            likelihood_ratio_clipping=self.likelihood_ratio_clipping)
