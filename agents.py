from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import cv2
import glob
import os
import six
import time
import numpy as np
import tensorflow as tf

from six.moves import range

from lab_rl.models import SimpleDQNModel
from lab_rl.replaymemories import SimpleReplayMemory
from lab_rl.helper import get_human_readable, get_softmax, print_stats

import logging
_logger = logging.getLogger(__name__)

# Ignore all numpy warnings during training
np.seterr(all='ignore')


class Agent(object):
    def __init__(self, args, rng, env, paths=None):
        _logger.info("Initializing Agent (type: %s, load_model: %s)" %
                     (args.agent, str(isinstance(args.load_model, six.string_types))))
        self.args = args
        self.rng = rng
        self.env = env
        self.paths = paths
        self.episode_reward = 0.0
        self.episode_losses = []
        self.epoch_rewards = []
        self.available_actions = self.env.count_actions()
        self.epsilon = self.args.epsilon_start
        self.tau = self.args.tau_start
        self.tau_diff = 0.0
        self.loss = 0

        self.epoch = 0  # epoch counter
        self.episode = 0  # episode counter

        self.step_current = 0  # total steps so far
        self.step_episode = 0  # steps per episode
        self.total_reward = 0  # total reward so far
        self.test_reward_best = 0  # best avg reward after epochs

        # Prepare training logs (written after every episode)
        self.csv_train_file = None
        self.csv_train_writer = None
        if not self.args.play:
            self.prepare_csv_train()

        # Prepare testing logs (written after every epoch)
        self.csv_test_file = None
        self.csv_test_writer = None
        if not self.args.play:
            self.prepare_csv_test()

        self.observation = None
        self.model = None
        self.model_name = None
        self.model_last = None
        self.model_input_shape = None
        # Model parameter
        self.session = None
        self.saver = None
        self.start_time = time.time()
        self.episode_start_time = self.start_time
        self.epoch_start_time = self.start_time
        self.batch_size = self.args.batch_size

    def prepare_csv_train(self):
        first_row = ('episode',
                     'time',
                     'duration',
                     'steps_total',
                     'steps',
                     'reward_total',
                     'reward',
                     'tau',
                     'loss',
                     'batch_size')
        self.csv_train_file = open(os.path.join(self.paths['log_path'], 'stats_train.csv'), "wb")
        self.csv_train_writer = csv.writer(self.csv_train_file)
        self.csv_write_train(first_row)

    def csv_write_train(self, row):
        self.csv_train_writer.writerow(row)
        self.csv_train_file.flush()

    def prepare_csv_test(self):
        first_row = ('epoch',
                     'time',
                     'duration',
                     'steps',
                     'episodes',
                     'reward_avg',
                     'steps_avg')
        self.csv_test_file = open(os.path.join(self.paths['log_path'], 'stats_test.csv'), "wb")
        self.csv_test_writer = csv.writer(self.csv_test_file)
        self.csv_write_test(first_row)

    def csv_write_test(self, row):
        self.csv_test_writer.writerow(row)
        self.csv_test_file.flush()

    def preprocess_input(self, img):
        if self.args.color_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.args.input_width, self.args.input_height))
        return np.reshape(img, self.model_input_shape)

    def get_action(self, *args):
        pass

    def step(self):
        s = self.preprocess_input(self.env.get_observation())
        a = self.get_action(s, self.tau)
        r = self.env.step(a)
        is_terminal = not self.env.is_running() or r % 2 == 1
        return s, a, r, is_terminal

    def episode_reset(self):
        self.episode += 1
        #print('Test Episode', self.episode)
        self.episode_reward = 0
        self.step_episode = 0
        self.episode_losses = []
        self.env.reset()
        self.episode_start_time = time.time()

    def episode_cleanup(self):
        self.epoch_rewards.append(self.episode_reward)
        self.total_reward += self.episode_reward
        self.loss = sum(self.episode_losses)/len(self.episode_losses)
        new_row = (self.episode,  # current episode
                   "{0:.1f}".format(time.time() - self.start_time),  # total time
                   "{0:.1f}".format(time.time() - self.episode_start_time),  # episode duration
                   self.step_current,  # total steps so far
                   self.step_episode,  # steps per episode
                   self.total_reward,  # total reward so far
                   self.episode_reward,  # reward per episode
                   "{0:.4f}".format(self.tau),  # current tau
                   "{0:.4f}".format(self.loss),  # avg loss per tau
                   self.batch_size  # current batch size used for training
                   )
        print('Train Episode', self.episode, '(steps:', self.step_current,')finished. Reward:', self.episode_reward)
        if not self.args.play:
            self.csv_write_train(new_row)

    def epoch_reset(self):
        self.epoch += 1
        self.epoch_rewards = []
        self.epoch_start_time = time.time()

    def epoch_cleanup(self):

        if self.step_current > 0:
            print_stats(self.step_current,
                        self.args.steps,
                        self.epoch_rewards,
                        time.time() - self.start_time)

        test_reward, test_steps = self.test(self.args.test_episodes)
        new_row = (self.episode,  # current epoch
                   "{0:.1f}".format(time.time() - self.start_time),  # total time
                   "{0:.1f}".format(time.time() - self.epoch_start_time),  # epoch duration
                   self.step_current,  # total steps so far
                   self.epoch,  # total episodes so far
                   "{0:.4f}".format(test_reward),  # avg reward per episode during testing
                   "{0:.4f}".format(test_steps)  # avg steps per episode during testing
                   )
        if self.test_reward_best <= test_reward:
            self.test_reward_best = test_reward
            # self.model_name = "DQN_{:04}".format(
            #    int(self.step_current / (self.args.backup_frequency * self.args.steps)))
            for old in glob.glob(os.path.join(self.paths['model_path'],'DQN_epoch_*')):
                os.remove(old)
            self.model_name = 'DQN_epoch_{:04}'.format(self.epoch)
            self.model_last = os.path.join(self.paths['model_path'], self.model_name)
            self.saver.save(self.session, self.model_last)
            _logger.info("Saved network after epoch %i (%i steps): %s" %
                         (self.epoch, self.step_current, self.model_name))
        if not self.args.play:
            self.csv_write_test(new_row)

    def test(self, episodes):
        print('TESTING')
        episode_rewards = []
        episode_steps = []
        save_video = False
        for episode in range(0, episodes):
            self.bla = episode
            if episode == 0 and self.args.save_video:
                save_video = True
            reward, steps = self.play(save_video)
            episode_rewards.append(reward)
            episode_steps.append(steps)
        return sum(episode_rewards)/episodes, sum(episode_steps)/episodes

    def play(self, save_video, num_episodes=1):
        # print('Test Episode', self.bla)
        out_video = None
        video_path = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_path = os.path.join(
                self.paths['video_path'],
                "video_" + self.model_name + ".avi")
            out_video = cv2.VideoWriter(video_path, fourcc, self.args.fps, (self.args.width, self.args.height))
        steps_total = 0
        reward_total = 0
        self.env.reset()
        while num_episodes != 0:
            if not self.env.is_running() or reward_total % 2 == 1:
                # print('Test Episode finished - Reward:', reward_total)
                if reward_total % 2 == 1:
                    self.env.reset()
                return reward_total, steps_total
            steps_total += 1
            state_raw = self.env.get_observation()
            state = self.preprocess_input(state_raw)
            action = self.get_action(state, self.args.tau_min)  # epsilon = 0.05, tau = 0.1
            for _ in range(self.args.frame_repeat):
                if self.args.show:
                    cv2.imshow("frame-test", state_raw)
                    cv2.waitKey(20)
                if save_video:
                    out_video.write(state_raw.astype('uint8'))
                reward = self.env.step(action, 1)
                reward_total += reward
                if not self.env.is_running() or reward_total % 2 == 1:
                    break
                state_raw = self.env.get_observation()
        if save_video:
            out_video.release()
            print("Saved video (fps:%i, size:%s) to: %s [%s]" %
                  (self.args.fps, str((self.args.width, self.args.height)),
                   video_path, get_human_readable(os.path.getsize(video_path))))
        if self.args.show:
            cv2.destroyAllWindows()


class SimpleDQNAgent(Agent):
    def __init__(self, args, rng, env, paths):
        # Call super class
        super(SimpleDQNAgent, self).__init__(args, rng, env, paths)
        print('Starting simple dqn agent.')

        # Prepare model
        tf.set_random_seed(self.args.random_seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        # Initiate tensorflow session
        self.session = tf.Session(config=config)
        self.model_input_shape = (self.args.input_width, self.args.input_height) + \
                                 (self.args.color_channels,)
        self.model = SimpleDQNModel(self.args,
                                    self.rng,
                                    self.session,
                                    self.model_input_shape,
                                    self.available_actions,
                                    self.paths['model_path'])
        self.memory = SimpleReplayMemory(self.args,
                                         self.rng,
                                         self.model_input_shape)

        self.saver = tf.train.Saver(max_to_keep=1000)
        if self.args.load_model is not None:
            self.saver.restore(self.session, self.args.load_model)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)
        if not self.args.play:
        # Backup initial model weights
            self.model_name = "DQN_0000"
            self.model_last = os.path.join(self.paths['model_path'], self.model_name)
            self.saver.save(self.session, self.model_last)

    def train_model(self):
        # train model with random batch from memory
        # if self.step_current % int((1/5)*self.args.steps) == 0:
        #    self.batch_size *= 2
        if self.memory.size > 2 * self.batch_size:
            s, a, r, s_prime, is_terminal = self.memory.get_batch()
            qs = self.model.get_qs(s)
            # print('Qs', qs[0])
            max_qs = np.max(self.model.get_qs(s_prime), axis=1)
            # print('a', a[0], 'r', r[0], 'gamma', self.args.gamma, 'q_s_prime', max_qs[0])
            qs[np.arange(qs.shape[0]), a] = r + (1 - is_terminal) * self.args.gamma * max_qs
            # print('Qs_updated', qs[0])
            return self.model.train(s, qs)
        return 0.0

    '''
    def update_epsilon(self, steps):
        # Update epsilon if necessary
        if steps > self.args.epsilon_decay * self.args.steps:
            return self.args.epsilon_min
        return self.args.epsilon_start - \
               steps * (self.args.epsilon_start - self.args.epsilon_min) / \
               (self.args.epsilon_decay * self.args.steps)
    '''

    def update_tau(self):
        # Update tau if necessary
        if self.tau <= self.args.tau_min:
            return self.args.tau_min
        return self.tau - self.tau * self.args.tau_decay

    def get_action(self, state, tau):
        """ Returns an action selected through softmax. """
        # print(self.step_current, self.tau, self.batch_size)
        return self.rng.choice(self.available_actions,
                               p=get_softmax(self.model.get_qs(state),
                                             tau))

    def train(self):
        self.epoch_cleanup()
        self.epoch_reset()
        print("TRAINING")
        self.episode_reset()
        for self.step_current in range(1, self.args.steps+1):
            self.step_episode += 1
            self.tau = self.update_tau()
            s, a, r, is_terminal = self.step()
            self.episode_reward += r
            self.memory.add(s, a, r, is_terminal)
            self.episode_losses.append(self.train_model())
            if not self.env.is_running() or is_terminal:
                self.episode_cleanup()
                self.episode_reset()
            if self.step_current % (self.args.backup_frequency * self.args.steps) == 0:
                self.epoch_cleanup()
                self.epoch_reset()
                if not self.step_current == self.args.steps:
                    print("TRAINING")
                    self.episode_reset()


class RandomAgent(Agent):
    """Simple random agent for DeepMind Lab."""

    def __init__(self, args, rng, env, paths):
        # Call super class
        super(RandomAgent, self).__init__(args, rng, env, paths)
        print('Starting random discretized agent.')

    def get_action(self, *args):
        """Gets an image state and a reward, returns an action."""
        return self.rng.randint(0, self.available_actions)

    def train(self):
        pass


class DummyAgent(Agent):
    """Simple dummy agent for DeepMind Lab."""

    def __init__(self, args, rng, env, paths):
        # Call super class
        super(DummyAgent, self).__init__(args, rng, env, paths)
        print('Starting dummy agent.')

    def get_action(self, *args):
        """Gets an image state and a reward, returns an action."""
        return 0

    def train(self):
        pass