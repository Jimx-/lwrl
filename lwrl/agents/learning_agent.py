from lwrl.agents import Agent


class LearningAgent(Agent):
    def __init__(
            self,
            state_spec,
            action_spec,
            discount_factor,
            optimizer=None,
            state_preprocess_pipeline=None
    ):
        self.discount_factor = discount_factor
        self.state_preprocess_pipeline = state_preprocess_pipeline

        super().__init__(state_spec, action_spec)
