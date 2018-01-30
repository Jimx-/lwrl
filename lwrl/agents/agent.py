from gym import spaces

class Agent:
    def __init__(self, state_spec, action_spec):
        self.state_space = spaces.Box(**state_spec)
        self.action_space = spaces.Discrete(**action_spec)

        self.model = self.init_model()

    def init_model(self):
        raise NotImplementedError()

    def act(self, obs, random_action=True):
        return self.model.act(obs, random_action)

    def observe(self, obs, action, reward, done):
        self.model.observe(obs, action, reward, done)

    def restore_model(self):
        self.model.restore()
