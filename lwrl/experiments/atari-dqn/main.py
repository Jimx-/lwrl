import argparse

from lwrl.agents import agent_factory
from lwrl.environments import env_factory
from lwrl.utils import read_config
from lwrl.executions import Runner

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, help='Gym environment id')
parser.add_argument('--config', type=str, help='Path to the config file')
parser.add_argument('--save_dir', type=str, help='Directory to save the trained model', default=None)
parser.add_argument('--log_dir', type=str, help='Directory to save the training log', default=None)
parser.add_argument('--is_train', help='Whether to train or to test the model', action='store_true', default=False)
parser.add_argument('--visualize', help='Whether to visualize the gameplay', action='store_true', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    config = read_config(args.config)

    env = env_factory(type='atari', id=args.env_id, visualize=args.visualize)
    test_env = env_factory(type='atari', id=args.env_id, visualize=args.visualize, episodic_life=False, clip_reward=False)

    saver_spec = dict(
        save_dir=args.save_dir
    )

    agent = agent_factory(state_spec=env.state_spec, action_spec=env.action_spec, **config, saver_spec=saver_spec)
    runner = Runner(agent, env, test_env)

    if args.is_train:
        runner.train(logdir=args.log_dir)
    else:
        agent.restore_model()
        runner.test()
