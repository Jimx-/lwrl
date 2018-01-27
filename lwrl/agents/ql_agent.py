import torch
import random
import tqdm
import numpy as np
from tensorboard_logger import Logger

from lwrl.models import DeepQNetwork
from lwrl.memories import get_replay_memory
import lwrl.utils.th_helper as H
from lwrl.utils.history import History
from lwrl.utils.saver import Saver


class BaseQLearningAgent:
    def __init__(self, network_cls, env, test_env, exploration_schedule, config, save_dir=None, save_freq=100000,
                 test_freq=1000):
        self.env = env
        self.test_env = test_env
        self.exploration_schedule = exploration_schedule
        self.config = config
        self.history_length = config['history_length']
        self.batch_size = config['batch_size']

        if len(env.observation_space.shape) > 1:
            # raw image input
            frame_width, frame_height, frame_channel = env.observation_space.shape
            in_channels = frame_channel * self.history_length
        else:
            # one-dimensional input
            in_channels = env.observation_space.shape[0]

        # set up online network and target network
        self.num_actions = env.action_space.n
        self.q_network = network_cls(in_channels, self.num_actions).type(H.float_tensor)
        self.target_network = network_cls(in_channels, self.num_actions).type(H.float_tensor)

        optimizer_conf = config['optimizer']
        self.optimizer = H.optimizer_dict[optimizer_conf['type']](self.q_network.parameters(), **optimizer_conf['args'])

        self.replay_memory = get_replay_memory(config['memory'])
        self.history = History(self.history_length)
        self.global_step = 0
        self.num_updates = 0

        self.discount_factor = config['discount_factor']
        self.clip_error = config['clip_error']
        self.update_target_freq = config['update_target_freq']

        self.save_freq = save_freq
        self.saver = None
        if save_dir is not None and save_freq > 0:
            self.saver = Saver(save_dir)

        self.test_freq = test_freq

    def act(self, obs, random_action=True):
        # fill in history on the beginning of an episode
        if self.history.empty():
            for _ in range(self.history_length):
                self.history.add(obs)

        # epsilon-greedy action selection
        eps = self.exploration_schedule.value(self.global_step)
        if not random_action:
            eps = 0.05
        if random.random() < eps:
            return self.env.action_space.sample()
        else:
            obs = torch.from_numpy(self.history.get()).type(H.float_tensor).unsqueeze(0) / 255.0
            with torch.no_grad():
                return self.q_network(H.Variable(obs)).data.max(1)[1].cpu()[0]

    def observe(self, obs, action, reward, done, learn, keep_memory):
        self.history.add(obs)

        if keep_memory:
            self.replay_memory.add(obs, action, reward, done)

        if learn:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_mask = \
                self.replay_memory.sample(self.batch_size)
            obs_batch = H.Variable(torch.from_numpy(obs_batch).type(H.float_tensor) / 255.0)
            next_obs_batch = H.Variable(torch.from_numpy(next_obs_batch).type(H.float_tensor) / 255.0)
            action_batch = H.Variable(torch.from_numpy(action_batch).long())
            reward_batch = H.Variable(torch.from_numpy(reward_batch))
            neg_done_mask = H.Variable(torch.from_numpy(1.0 - done_mask).type(H.float_tensor))

            if H.use_cuda:
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()

            # minimize (Q(s, a) - (r + gamma * max Q(s', a'; w'))^2
            q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1)) # Q(s, a; w)
            next_max_q_values = self.target_network(next_obs_batch).detach().max(1)[0] # max Q(s', a'; w')
            next_q_values = neg_done_mask * next_max_q_values
            target_q_values = reward_batch + self.discount_factor * next_q_values # r + gamma * max Q(s', a'; w')
            td_error = target_q_values.unsqueeze(1) - q_values
            clipped_td_error = td_error.clamp(-self.clip_error, self.clip_error)
            grad = clipped_td_error * -1.0

            self.optimizer.zero_grad()
            q_values.backward(grad.data)

            self.optimizer.step()
            self.num_updates += 1

            # target network <- online network
            if self.num_updates % self.update_target_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, logdir=None, render=False, verbose=True):
        max_timesteps = self.config['max_timesteps']
        learning_starts = self.config['learning_starts']
        train_freq = self.config['train_freq']

        pbar = range(max_timesteps)
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
            if t > learning_starts:
                action = self.act(obs)
            else:
                action = self.env.action_space.sample()

            # take action in the environment
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            learn = t > learning_starts and t % train_freq == 0 and self.replay_memory.size() > self.batch_size
            # observe the effect
            self.observe(obs, action, reward, done, learn=learn, keep_memory=True)

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
                            total_episodes, t, episode_reward, avg_r, self.exploration_schedule.value(t)
                        )
                    )

                if logdir is not None:
                    logger.log_value('episode_reward', episode_reward, t)
                    logger.log_value('avg_episode_reward', avg_r, t)

                    if total_episodes % self.test_freq == 0:
                        test_score = self.test()
                        logger.log_value('test_score', test_score, t)

                episode_reward = 0

            if self.saver is not None and t % self.save_freq == 0:
                self.save(t)

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
                self.observe(obs, action, reward, done, learn=False, keep_memory=False)

                acc_reward += reward
                if done:
                    scores.append(acc_reward)
                    self.test_env.reset()
                    self.history.reset()

        return sum(scores) / float(num_episodes)

    def save(self, global_step):
        self.saver.save({
            'global_step': global_step,
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, global_step)

    def restore(self):
        checkpoint = self.saver.restore()
        self.global_step = checkpoint['global_step']
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class QLearningAgent(BaseQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(DeepQNetwork, *args, **kwargs)


class DuelingQLearningAgent(BaseQLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(DeepQNetwork, *args, **kwargs)