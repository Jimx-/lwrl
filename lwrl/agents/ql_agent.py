import torch
import random
import tqdm
import numpy as np
from tensorboard_logger import Logger

from lwrl.agents import LearningAgent
from lwrl.memories import get_replay_memory
from lwrl.models import QModel
from lwrl.models.networks import DeepQNetwork, DuelingDQN
from lwrl.utils.history import History


class BaseQLearningAgent(LearningAgent):
    def __init__(
            self,
            env,
            test_env,
            state_spec,
            action_spec,
            network_cls,
            network_spec,
            optimizer_spec,
            memory,
            exploration_schedule,
            discount_factor,
            clip_error,
            update_target_freq,
            history_length,
            max_timesteps,
            learning_starts,
            train_freq=1,
            batch_size=32,
            double_q_learning=False,
            saver_spec=None,
            save_freq=100000,
            test_freq=1000
    ):

        self.env = env
        self.test_env = test_env
        self.history_length = history_length
        self.batch_size = batch_size

        self.network_cls = network_cls
        self.network_spec = network_spec
        self.exploration_schedule = exploration_schedule
        self.optimizer_spec = optimizer_spec

        self.replay_memory = get_replay_memory(memory)
        self.history = History(self.history_length)
        self.global_step = 0
        self.num_updates = 0

        self.clip_error = clip_error
        self.update_target_freq = update_target_freq
        self.double_q_learning = double_q_learning

        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.train_freq = train_freq

        self.save_freq = save_freq
        self.saver_spec = saver_spec
        self.test_freq = test_freq

        super().__init__(
            state_spec=state_spec,
            action_spec=action_spec,
            discount_factor=discount_factor,
            optimizer_spec=optimizer_spec
        )

    def init_model(self):
        return QModel(
            state_space=self.state_space,
            action_space=self.action_space,
            network_cls=self.network_cls,
            network_spec=self.network_spec,
            exploration_schedule=self.exploration_schedule,
            optimizer_spec=self.optimizer_spec,
            saver_spec = self.saver_spec,
            discount_factor=self.discount_factor,
            clip_error=self.clip_error,
            update_target_freq=self.update_target_freq,
            double_q_learning=self.double_q_learning
        )

    def act(self, obs, random_action=True):
        # fill in history on the beginning of an episode
        if self.history.empty():
            for _ in range(self.history_length):
                self.history.add(obs)

        return super().act(self.history.get(), random_action)

    def observe(self, obs, action, reward, done, learn=False):
        super().observe(obs, action, reward, done)

        self.history.add(obs)
        self.replay_memory.add(obs, action, reward, done)

        if learn:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_mask = \
                self.replay_memory.sample(self.batch_size)
            self.model.update(obs_batch, action_batch, reward_batch, next_obs_batch, done_mask)

    def train(self, logdir=None, render=False, verbose=True):
        pbar = range(self.max_timesteps)
        if verbose:
            pbar = tqdm.tqdm(pbar)

        if logdir is not None:
            logger = Logger(logdir)

        obs = self.env.reset()
        episode_reward = 0
        episode_rewards = []
        for t in pbar:
            self.global_step = t
            if render:
                self.env.render()

            # choose action
            if t > self.learning_starts:
                action = self.act(obs)
            else:
                action = self.env.action_space.sample()

            # take action in the environment
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            learn = t > self.learning_starts and t % self.train_freq == 0 and self.replay_memory.size() > self.batch_size
            # observe the effect
            self.observe(obs, action, reward, done, learn=learn)

            if done:
                self.env.reset()
                self.history.reset()
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
                            total_episodes, t, episode_reward, avg_r, self.model.exploration_schedule.value(t)
                        )
                    )

                if logdir is not None:
                    logger.log_value('episode_reward', episode_reward, t)
                    logger.log_value('avg_episode_reward', avg_r, t)

                    if total_episodes % self.test_freq == 0:
                        test_score = self.test()
                        logger.log_value('test_score', test_score, t)

                episode_reward = 0

            if t % self.save_freq == 0:
                self.model.save(t)

    def test(self, num_episodes=10, render=False):
        scores = []
        for episode in range(num_episodes):
            obs = self.test_env.reset()
            done = False
            acc_reward = 0

            while not done:
                if render:
                    self.test_env.render()

                action = self.act(obs, random_action=False)
                obs, reward, done, _ = self.test_env.step(action)
                self.observe(obs, action, reward, done, learn=False)

                acc_reward += reward
                if done:
                    scores.append(acc_reward)
                    self.test_env.reset()
                    self.history.reset()

        return sum(scores) / float(num_episodes)

class QLearningAgent(BaseQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(network_cls=DeepQNetwork, *args, **kwargs)


class DuelingQLearningAgent(BaseQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(network_cls=DuelingDQN, *args, **kwargs)
