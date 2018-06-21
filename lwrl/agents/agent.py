from gym import spaces


class Agent:
    def __init__(self, state_spec, action_spec):
        self.state_spec = state_spec
        self.action_spec = action_spec

        self.timestep = None

        self.model = self.init_model()

    def init_model(self):
        raise NotImplementedError()

    def act(self, obs, random_action=True):
        action, self.timestep = self.model.act(obs, random_action)
        return action

    def observe(self, obs, action, reward, done, training=False):
        self.model.observe(obs, action, reward, done)

    def restore_model(self):
        self.model.restore()

    def reset(self):
        pass
