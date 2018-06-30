import argparse
import copy

import numpy as np

from lwrl.agents import agent_factory
from lwrl.environments import env_factory
from lwrl.executions import ThreadedRunner, threaded_agent_wrapper
from lwrl.utils import read_config

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_id', type=str, help='Gym environment id')
parser.add_argument(
    '-a', '--agent', type=str, help='Path to the agent config file')
parser.add_argument(
    '-n', '--network', type=str, help='Path to the network config file')
parser.add_argument(
    '-w', '--workers', type=int, help='Number of workers', default=16)
parser.add_argument(
    '-s',
    '--save_dir',
    type=str,
    help='Directory to save the trained model',
    default=None)
parser.add_argument(
    '--is_train',
    help='Whether to train or to test the model',
    action='store_true',
    default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    agent_config = read_config(args.agent)
    network_config = read_config(args.network)

    envs = [
        env_factory(type='atari', id=args.env_id, visualize=False)
        for _ in range(args.workers)
    ]
    test_envs = [
        env_factory(
            type='atari',
            id=args.env_id,
            visualize=False,
            episodic_life=False,
            clip_reward=False) for _ in range(args.workers)
    ]

    saver_spec = dict(save_dir=args.save_dir)

    agent_configs = []
    for i in range(args.workers):
        worker_config = copy.deepcopy(agent_config)
        worker_config['exploration_schedule']['args'][
            'final'] = np.random.choice(
                [0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
        agent_configs.append(worker_config)

    agent = agent_factory(
        **agent_configs[0],
        state_spec=envs[0].state_spec,
        action_spec=envs[0].action_spec,
        network_spec=network_config,
        saver_spec=saver_spec)

    agents = [agent]
    for config, env in zip(agent_configs[1:], envs[1:]):
        agent_type = config['type']
        del config['type']
        worker = threaded_agent_wrapper(agent_type)(
            model=agent.model,
            state_spec=env.state_spec,
            action_spec=env.action_spec,
            network_spec=network_config,
            **config)

        agents.append(worker)

    runner = ThreadedRunner(agents, envs, test_envs)

    if args.is_train:
        runner.train()
    else:
        agent.restore_model()
        runner.test(render=True)
