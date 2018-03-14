import tqdm
import numpy as np
from tensorboard_logger import Logger
import threading

from lwrl.agents import agent_dict


class ThreadedRunner:
    def __init__(self, agents, envs, test_envs=None):
        self.agents = agents
        self.envs = envs
        self.test_envs = test_envs if test_envs is not None else self.envs

        self.stopped = False
        self.episode_list_lock = threading.Lock()

        assert len(self.agents) == len(self.envs)
        assert len(self.envs) == len(self.test_envs)

    def train(
            self,
            max_timestep=50000000,
            save_freq=100000,
            test_freq=1000,
            logdir=None,
            render=False,
            verbose=True
    ):
        if verbose:
            pbar = tqdm.tqdm(total=max_timestep)

        if logdir is not None:
            logger = Logger(logdir)

        self.episode_rewards = []
        self.stopped = False
        self.global_timestep = 0
        self.last_timestep = 0

        threads = [threading.Thread(target=self._train_single, args=(t, self.agents[t], self.envs[t]),
                                    kwargs=dict(
                                        render=render
                                    ))
                       for t in range(len(self.agents))]

        for t in threads:
            t.start()

        try:
            while any([t.is_alive() for t in threads]) and self.global_timestep < max_timestep:
                total_episodes = len(self.episode_rewards)
                if total_episodes == 0:
                    continue

                if total_episodes < 100:
                    avg_r = np.mean(self.episode_rewards)
                else:
                    avg_r = np.mean(self.episode_rewards[-101:-1])

                if verbose:
                    pbar.update(self.global_timestep - self.last_timestep)
                    self.last_timestep = self.global_timestep
                    pbar.set_description(
                        'Train: episode: {}, global steps: {}, episode score: {:.1f}, avg score: {:.2f}'.format(
                            total_episodes, self.global_timestep, self.episode_rewards[-1], avg_r
                        )
                    )

                if logdir is not None:
                    logger.log_value('episode_reward', self.episode_rewards[-1], self.global_timestep)
                    logger.log_value('avg_episode_reward', avg_r, self.global_timestep)

                    #if total_episodes % test_freq == 0:
                    #    test_score = self.test()
                    #    logger.log_value('test_score', test_score, self.global_timestep)

                if self.global_timestep % save_freq == 0:
                    self.agents[0].model.save(self.global_timestep)

        except KeyboardInterrupt:
            print('Received keyboard interrupt, stopping threads')

        self.stopped = True

        for t in threads:
            t.join()

    def _train_single(
            self,
            tid,
            agent,
            env,
            render=False,
    ):
        timestep = 0

        while not self.stopped:
            obs = env.reset()
            episode_reward = 0
            self.global_timestep = max(self.global_timestep, timestep)

            while True:
                if render:
                    env.render()

                action = agent.act(obs)

                # take action in the environment
                obs, reward, done = env.step(action)
                episode_reward += reward
                timestep += 1
                # observe the effect
                agent.observe(obs, action, reward, done, training=True)

                if done:
                    agent.reset()
                    self.episode_list_lock.acquire()
                    self.episode_rewards.append(episode_reward)
                    self.episode_list_lock.release()
                    break

                if self.stopped:
                    return


    def test(self, num_episodes=10, render=False):
        scores = []
        for episode in range(num_episodes):
            obs = self.test_envs[0].reset()
            done = False
            acc_reward = 0

            while not done:
                if render:
                    self.test_envs[0].render()

                action = self.agents[0].act(obs, random_action=False)
                obs, reward, done = self.test_envs[0].step(action)
                self.agents[0].observe(obs, action, reward, done)

                acc_reward += reward
                if done:
                    scores.append(acc_reward)
                    self.test_envs[0].reset()
                    self.agents[0].reset()

        return sum(scores) / float(num_episodes)


def threaded_agent_wrapper(agent_class):
    if isinstance(agent_class, str):
        agent_class = agent_dict[agent_class]

    class ThreadedAgent(agent_class):
        def __init__(self, model, **kwargs):
            self.model = model
            super().__init__(**kwargs)

        def init_model(self):
            return self.model

    return ThreadedAgent
