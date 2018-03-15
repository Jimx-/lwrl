import gym

from lwrl.environments.environment import Environment
from lwrl.environments.openai_gym import OpenAIGym, OpenAIGymWrapper
from lwrl.environments.atari_wrapper import get_atari_env


env_dict = dict(
    atari=get_atari_env,
    gym=OpenAIGym,
)


def env_factory(type, *args, **kwargs):
    return env_dict[type](*args, **kwargs)
