from gym import spaces

class Agent:
    def __init__(self, state_spec, action_spec):
        self.state_space = spaces.Box(**state_spec)
        self.action_space = spaces.Discrete(**action_spec)
