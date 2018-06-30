from lwrl.agents import MemoryAgent
from lwrl.models import DDPGModel


class DDPGAgent(MemoryAgent):
    def __init__(self,
                 state_spec,
                 action_spec,
                 network_spec,
                 optimizer,
                 memory,
                 exploration_schedule,
                 discount_factor,
                 update_target_freq,
                 update_target_weight,
                 history_length,
                 learning_starts,
                 train_freq=1,
                 batch_size=32,
                 saver_spec=None,
                 critic_network_spec=None,
                 critic_optimizer=None,
                 state_preprocess_pipeline=None):

        self.network_spec = network_spec
        self.exploration_schedule = exploration_schedule
        self.optimizer = optimizer

        self.global_step = 0

        self.update_target_freq = update_target_freq
        self.update_target_weight = update_target_weight

        if critic_network_spec is None:
            critic_network_spec = network_spec
        if critic_optimizer is None:
            critic_optimizer = dict(type='Adam', args=dict(lr=1e-3, ))

        self.critic_network_spec = critic_network_spec
        self.critic_optimizer = critic_optimizer

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
            state_preprocess_pipeline=state_preprocess_pipeline)

    def init_model(self):
        return DDPGModel(
            state_spec=self.state_spec,
            action_spec=self.action_spec,
            network_spec=self.network_spec,
            exploration_schedule=self.exploration_schedule,
            optimizer=self.optimizer,
            saver_spec=self.saver_spec,
            discount_factor=self.discount_factor,
            update_target_freq=self.update_target_freq,
            update_target_weight=self.update_target_weight,
            critic_network_spec=self.critic_network_spec,
            critic_optimizer=self.critic_optimizer,
            state_preprocess_pipeline=self.state_preprocess_pipeline)
