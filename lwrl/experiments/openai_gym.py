import argparse

from lwrl.agents import agent_factory
from lwrl.environments import env_factory
from lwrl.executions import Runner
from lwrl.utils import read_config

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, help='Gym environment id')
parser.add_argument('--agent', type=str, help='Path to the agent config file')
parser.add_argument('--network', type=str, help='Path to the agent config file')
parser.add_argument('--save_dir', type=str, help='Directory to save the trained model', default=None)
parser.add_argument('--log_dir', type=str, help='Directory to save the training log', default=None)
parser.add_argument('--is_train', help='Whether to train or to test the model', action='store_true', default=False)
parser.add_argument('--visualize', help='Whether to visualize the gameplay', action='store_true', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    agent_config = read_config(args.agent)
    network_config = read_config(args.network)

    env = env_factory(type='gym', id=args.env_id, visualize=args.visualize)

    saver_spec = dict(
        save_dir=args.save_dir
    )

    agent = agent_factory(state_spec=env.state_spec, action_spec=env.action_spec, network_spec=network_config,
                          **agent_config, saver_spec=saver_spec)
    runner = Runner(agent, env)

    if args.is_train:
        runner.train(logdir=args.log_dir)
    else:
        agent.restore_model()
        runner.test()
