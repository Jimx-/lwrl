

class Environment:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    @property
    def state_spec(self):
        raise NotImplementedError

    @property
    def action_spec(self):
        raise NotImplementedError
