import gym

from lwrl.environments.atari_wrapper import get_atari_env


env_dict = dict(
    atari=get_atari_env,
    gym=gym.make
)


def env_factory(type, *args, **kwargs):
    return env_dict[type](*args, **kwargs)
