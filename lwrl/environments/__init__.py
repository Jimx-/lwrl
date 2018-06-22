import gym
import logging

from lwrl.environments.environment import Environment
from lwrl.environments.openai_gym import OpenAIGym, OpenAIGymWrapper
from lwrl.environments.atari_wrapper import get_atari_env
from lwrl.utils.logging import begin_section

env_dict = dict(
    atari=get_atari_env,
    gym=OpenAIGym,
)


def env_factory(type, *args, **kwargs):
    logger = logging.getLogger(__name__)

    logger.info(begin_section('Environment'))
    logger.info('Creating enviroment of type {}'.format(type))
    logger.info('Configuration: {')
    for kv in kwargs.items():
        logger.info('\t{} = {}'.format(*kv))
    logger.info('}')

    env = env_dict[type](*args, **kwargs)

    for label, spec in (('State', env.state_spec), ('Action',
                                                    env.action_spec)):
        logger.info(label + ' spec: {')
        for kv in spec.items():
            logger.info('\t{} = {}'.format(*kv))
        logger.info('}')

    return env
