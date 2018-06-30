import logging

import numpy as np
import tqdm

from lwrl.utils.visualizer import Visualizer
import lwrl.utils.logging as L


class Runner:
    def __init__(self, agent, env, test_env=None):
        self.agent = agent
        self.env = env
        self.test_env = test_env if test_env is not None else self.env

    def train(self,
              max_timestep=50000000,
              save_freq=100000,
              test_freq=1000,
              log_freq=1,
              verbose=True):
        vis = Visualizer()
        logger = logging.getLogger(__name__)

        logger.info(L.begin_section('Training'))

        pbar = range(max_timestep)
        if verbose:
            pbar = tqdm.tqdm(pbar)

        obs = self.env.reset()
        episode_reward = 0
        episode_rewards = []

        episode_timesteps = []
        acc_episode_rewards = []
        acc_avg_episode_rewards = []
        for t in pbar:
            action = self.agent.act(obs)

            # take action in the environment
            obs, reward, done = self.env.step(action)
            episode_reward += reward
            # observe the effect
            self.agent.observe(obs, action, reward, done, training=True)

            if done:
                self.env.reset()
                self.agent.reset()
                episode_rewards.append(episode_reward)
                episode_timesteps.append(t)
                acc_episode_rewards.append(episode_reward)

                # log training status
                total_episodes = len(episode_rewards)
                if total_episodes < 100:
                    avg_r = np.mean(episode_rewards)
                else:
                    avg_r = np.mean(episode_rewards[-101:-1])
                acc_avg_episode_rewards.append(avg_r)

                if total_episodes % log_freq == 0:
                    logger.info(
                        'Reporting @ episode {}'.format(total_episodes))
                    logger.info('Episode {}:  total timestep:\t{}'.format(
                        total_episodes, t))
                    logger.info('Episode {}:  episode score:\t{}'.format(
                        total_episodes, episode_reward))
                    logger.info('Episode {}:  avg. eps. score:\t{}'.format(
                        total_episodes, avg_r))

                    vis.line(
                        'episode reward',
                        np.array(episode_timesteps),
                        np.array(acc_episode_rewards),
                        xlabel='Timestep',
                        append=True)
                    vis.line(
                        'average episode reward',
                        np.array(episode_timesteps),
                        np.array(acc_avg_episode_rewards),
                        xlabel='Timestep',
                        append=True)
                    episode_timesteps = []
                    acc_episode_rewards = []
                    acc_avg_episode_rewards = []

                if verbose:
                    pbar.set_description(
                        'Train: episode: {}, global steps: {}, episode score: {}, avg score: {}'.
                        format(total_episodes, t, episode_reward, avg_r))

                if total_episodes % test_freq == 0:
                    test_score = self.test()
                    vis.line(
                        'test score',
                        np.array([t]),
                        np.array([test_score]),
                        xlabel='Timestep',
                        append=True)

                    logger.info(
                        'Evaluating @ episode {}'.format(total_episodes))
                    logger.info('Episode {}:  test score:\t{}'.format(
                        total_episodes, test_score))

                episode_reward = 0

            if t % save_freq == 0:
                self.agent.model.save(t)

    def test(self, num_episodes=10):
        scores = []
        for episode in range(num_episodes):
            obs = self.test_env.reset()
            done = False
            acc_reward = 0

            while not done:
                action = self.agent.act(obs, random_action=False)
                obs, reward, done = self.test_env.step(action)
                self.agent.observe(obs, action, reward, done)

                acc_reward += reward
                if done:
                    scores.append(acc_reward)
                    self.test_env.reset()
                    self.agent.reset()

        return sum(scores) / float(num_episodes)
