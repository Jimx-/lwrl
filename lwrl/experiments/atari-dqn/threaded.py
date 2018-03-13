import argparse
import copy
import numpy as np

from lwrl.agents import agent_factory
from lwrl.environments import env_factory
from lwrl.utils import read_config
from lwrl.executions import ThreadedRunner, threaded_agent_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='Path to the config file')
parser.add_argument('-w', '--workers', type=str, help='Number of workers', default=16)
parser.add_argument('-s', '--save_dir', type=str, help='Directory to save the trained model', default=None)
parser.add_argument('-l', '--log_dir', type=str, help='Directory to save the training log', default=None)
parser.add_argument('--is_train', type=bool, help='Whether to train or to test the model', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    config = read_config(args.config)

    envs = [env_factory(type=config['environment']['type'], **config['environment']['train'])
            for _ in range(args.workers)]

    test_env_conf = config['environment'].get('test')
    if test_env_conf is None:
        test_envs = envs
    else:
        test_envs = [env_factory(type=config['environment']['type'], **test_env_conf) for _ in range(args.workers)]

    saver_spec = dict(
        save_dir=args.save_dir
    )

    agent_config = config['agent']
    agent_configs = []
    for i in range(args.workers):
        worker_config = copy.deepcopy(agent_config)
        worker_config['exploration_schedule']['args']['final'] = np.random.choice([0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
        agent_configs.append(worker_config)

    agent = agent_factory(**agent_configs[0], saver_spec=saver_spec)
    agents = [agent]
    for config in agent_configs[1:]:
        agent_type = config['type']
        del config['type']
        worker = threaded_agent_wrapper(agent_type)(
            model=agent.model,
            **config
        )

        agents.append(worker)

    runner = ThreadedRunner(agents, envs, test_envs)

    if args.is_train:
        runner.train(logdir=args.log_dir)
    else:
        agent.restore_model()
        runner.test(render=True)
