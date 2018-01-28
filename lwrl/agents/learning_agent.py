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
        super().__init__(state_spec, action_spec)

        if optimizer_spec is None:
            optimizer_spec = {
                "type": "Adam",
                "args": {
                    "lr": 0.00025,
                }
            }

        self.optimizer_builder = lambda params:\
            H.optimizer_dict[optimizer_spec['type']](params, **optimizer_spec['args'])

        self.discount_factor = discount_factor
