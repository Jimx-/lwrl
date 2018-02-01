import tqdm
import numpy as np
from tensorboard_logger import Logger


class Runner:
    def __init__(self, agent, env, test_env=None):
        self.agent = agent
        self.env = env
        self.test_env = test_env if test_env is not None else self.env

    def train(
            self,
            max_timestep=50000000,
            save_freq=100000,
            test_freq=1000,
            logdir=None,
            render=False,
            verbose=True
    ):
        pbar = range(max_timestep)
        if verbose:
            pbar = tqdm.tqdm(pbar)

        if logdir is not None:
            logger = Logger(logdir)

        obs = self.env.reset()
        episode_reward = 0
        episode_rewards = []
        for t in pbar:
            if render:
                self.env.render()

            action = self.agent.act(obs)

            # take action in the environment
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            # observe the effect
            self.agent.observe(obs, action, reward, done, training=True)

            if done:
                self.env.reset()
                self.agent.reset()
                episode_rewards.append(episode_reward)

                # log training status
                total_episodes = len(episode_rewards)
                if verbose:
                    if total_episodes < 100:
                        avg_r = np.mean(episode_rewards)
                    else:
                        avg_r = np.mean(episode_rewards[-101:-1])
                    pbar.set_description(
                        'Train: episode: {}, global steps: {}, episode score: {:.1f}, avg score: {:.2f}, exploration rate: {:.3f}'.format(
                            total_episodes, t, episode_reward, avg_r, self.agent.model.exploration_schedule.value(t)
                        )
                    )

                if logdir is not None:
                    logger.log_value('episode_reward', episode_reward, t)
                    logger.log_value('avg_episode_reward', avg_r, t)

                    if total_episodes % test_freq == 0:
                        test_score = self.test()
                        logger.log_value('test_score', test_score, t)

                episode_reward = 0

            if t % save_freq == 0:
                self.agent.model.save(t)

    def test(self, num_episodes=10, render=False):
        scores = []
        for episode in range(num_episodes):
            obs = self.test_env.reset()
            done = False
            acc_reward = 0

            while not done:
                if render:
                    self.test_env.render()

                action = self.agent.act(obs, random_action=False)
                obs, reward, done, _ = self.test_env.step(action)
                self.agent.observe(obs, action, reward, done)

                acc_reward += reward
                if done:
                    scores.append(acc_reward)
                    self.test_env.reset()
                    self.agent.reset()

        return sum(scores) / float(num_episodes)
