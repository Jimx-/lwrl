import gym

from lwrl.environments import Environment


class OpenAIGymWrapper(Environment):
    def __init__(self, env, monitor_dir=None, monitor_force=False, visualize=False):
        self.env = env
        self.visualize = visualize

        if monitor_dir is not None:
            self.env = gym.wrappers.Monitor(self.env, monitor_dir, force=monitor_force)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.visualize:
            self.env.render()

        obs, reward, done, _ = self.env.step(action)
        return obs, reward, done

    @property
    def state_spec(self):
        space = self.env.observation_space
        if isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')

    @property
    def action_spec(self):
        space = self.env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), num_actions=space.n, type='int')


class OpenAIGym(OpenAIGymWrapper):
    def __init__(self, id, *args, **kwargs):
        env = gym.make(id)
        super().__init__(env, *args, **kwargs)
