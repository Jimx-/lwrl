import gym

from lwrl.environments import Environment


class OpenAIGym(Environment):
    def __init__(self, env_or_id, monitor_dir=None, monitor_force=False, visualize=False):
        self.env = env_or_id
        self.visualize = visualize

        if isinstance(env_or_id, str):
            self.env = gym.make(self.env)

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
            return dict(num_actions=space.n, type='int')
