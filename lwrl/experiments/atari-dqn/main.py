import argparse

from lwrl.agents import QLearningAgent, DuelingQLearningAgent
from lwrl.environments import get_atari_env
from lwrl.utils import schedule, read_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file')
parser.add_argument('--save_dir', type=str, help='Directory to save the trained model', default=None)
parser.add_argument('--log_dir', type=str, help='Directory to save the training log', default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    config = read_config(args.config)

    exploration_schedule = schedule.get_schedule(config['exploration_schedule'])
    env = get_atari_env(**config['environment']['train'])

    test_env_conf = config['environment'].get('test')
    if test_env_conf is None:
        test_env = env
    else:
        test_env = get_atari_env(**test_env_conf)

    agent_type = QLearningAgent
    if config['dueling']:
        agent_type = DuelingQLearningAgent

    agent = agent_type(env, test_env, exploration_schedule, config, save_dir=args.save_dir)
    agent.train(logdir=args.log_dir)
