import argparse

from lwrl.agents import agent_factory
from lwrl.environments import env_factory
from lwrl.utils import read_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file')
parser.add_argument('--save_dir', type=str, help='Directory to save the trained model', default=None)
parser.add_argument('--log_dir', type=str, help='Directory to save the training log', default=None)
parser.add_argument('--is_train', type=bool, help='Whether to train or to test the model', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    config = read_config(args.config)

    env = env_factory(type=config['environment']['type'], **config['environment']['train'])

    test_env_conf = config['environment'].get('test')
    if test_env_conf is None:
        test_env = env
    else:
        test_env = env_factory(type=config['environment']['type'], **test_env_conf)

    agent = agent_factory(**config['agent'], env=env, test_env=test_env, save_dir=args.save_dir)

    if args.is_train:
        agent.train(logdir=args.log_dir)
    else:
        agent.restore()
        agent.test(render=True)
