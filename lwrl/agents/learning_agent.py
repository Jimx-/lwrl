from lwrl.agents import Agent
import lwrl.utils.th_helper as H


class LearningAgent(Agent):
    def __init__(
            self,
            state_spec,
            action_spec,
            discount_factor,
            optimizer_spec=None,
    ):
        self.discount_factor = discount_factor

        super().__init__(state_spec, action_spec)
