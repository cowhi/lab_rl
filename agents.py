from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import cv2
import os
import six
import time
import numpy as np
import tensorflow as tf

from six.moves import range

from lab_rl.models import SimpleDQNModel
from lab_rl.replaymemories import SimpleReplayMemory
from lab_rl.helper import get_human_readable, print_stats

import logging
_logger = logging.getLogger(__name__)


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
        self.loss = 0

        self.epoch = 0  # epoch counter
        self.episode = 0  # episode counter

        self.step_current = 0  # total steps so far
        self.step_episode = 0  # steps per episode
        self.total_reward = 0  # total reward so far

        # Prepare episode logs
        self.csv_file = None
        self.csv_writer = None

        first_row = ('episode',
                     'time',
                     'duration',
                     'steps_total',
                     'steps',
                     'reward_total',
                     'reward',
                     'epsilon',
                     'loss')
        self.write_csv_init('episode_stats', first_row)

        self.observation = None
        self.model = None
        self.model_name = None
        self.model_last = None
        self.model_input_shape = None
        # Model parameter
        self.session = None
        self.saver = None
        self.start_time = time.time()
        self.episode_start_time = None

    def write_csv_init(self, target, row):
        self.csv_file = open(os.path.join(self.paths['log_path'], target + '.csv'), "wb")
        self.csv_writer = csv.writer(self.csv_file)
        self.write_csv(row)

    def write_csv(self, row):
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def preprocess_input(self, img):
        if self.args.color_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.args.input_width, self.args.input_height))
        return np.reshape(img, self.model_input_shape)

    def get_action(self, *args):
        pass

    def step(self):
        s = self.preprocess_input(self.env.get_observation())
        a = self.get_action(s, self.epsilon)
        r = self.env.step(a)
        is_terminal = not self.env.is_running()
        return s, a, r, is_terminal

    def episode_reset(self):
        self.episode += 1
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
                   "{0:.4f}".format(self.epsilon),  # current epsilon
                   "{0:.4f}".format(self.loss)  # avg loss per episode
                   )
        self.write_csv(new_row)

    def epoch_reset(self):
        self.epoch += 1
        self.epoch_rewards = []

    def epoch_cleanup(self):
        self.model_name = "DQN_{:04}".format(
            int(self.step_current / (self.args.backup_frequency * self.args.steps)))
        self.model_last = os.path.join(self.paths['model_path'], self.model_name)
        self.saver.save(self.session, self.model_last)
        _logger.info("Saved network after epoch %i (%i steps): %s" %
                     (self.epoch, self.step_current, self.model_name))
        print_stats(self.step_current,
                    self.args.steps,
                    self.epoch_rewards,
                    time.time() - self.start_time)
        # TODO do testrun and save stats (epoch, steps, episodes, test_reward_avg, test_step_avg)
        if self.args.save_video:
            self.play()

    def play(self):
        print("Starting playing.")
        out_video = None
        video_path = None
        if self.args.save_video:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_path = os.path.join(
                self.paths['video_path'],
                "video_" + self.model_name + ".avi")
            out_video = cv2.VideoWriter(video_path, fourcc, self.args.fps, (self.args.width, self.args.height))

        reward_total = 0
        num_episodes = 1
        self.env.reset()
        while num_episodes != 0:
            if not self.env.is_running():
                self.env.reset()
                print("Total reward: {}".format(reward_total))
                reward_total = 0
                num_episodes -= 1

            state_raw = self.env.get_observation()
            state = self.preprocess_input(state_raw)
            action = self.get_action(state, 0.05)

            for _ in range(self.args.frame_repeat):
                if self.args.show:
                    cv2.imshow("frame-test", state_raw)
                    cv2.waitKey(20)
                if self.args.save_video:
                    out_video.write(state_raw.astype('uint8'))
                reward = self.env.step(action, 1)
                reward_total += reward
                if not self.env.is_running():
                    break
                state_raw = self.env.get_observation()
        if self.args.save_video:
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
        # Backup initial model weights
        self.model_name = "DQN_0000"
        self.model_last = os.path.join(self.paths['model_path'], self.model_name)
        self.saver.save(self.session, self.model_last)

    def train_model(self):
        # train model with random batch from memory
        if self.memory.size > 2 * self.args.batch_size:
            s, a, r, s_prime, is_terminal = self.memory.get_batch()
            qs = self.model.get_qs(s)
            max_qs = np.max(self.model.get_qs(s_prime), axis=1)
            qs[np.arange(qs.shape[0]), a] = r + (1 - is_terminal) * self.args.gamma * max_qs
            return self.model.train(s, qs)
        return 0.0

    def update_epsilon(self, steps):
        # Update epsilon if necessary
        if steps > self.args.epsilon_decay * self.args.steps:
            return self.args.epsilon_min
        else:
            return self.args.epsilon_start - \
                  steps * (self.args.epsilon_start - self.args.epsilon_min) / \
                   (self.args.epsilon_decay * self.args.steps)

    def get_action(self, state, epsilon):
        if self.rng.rand() <= epsilon:
            return self.rng.randint(0, self.available_actions)
        else:
            # TODO add ability to work with q values
            return self.model.get_action(state)

    def train(self):
        print("Starting training.")
        self.epoch_reset()
        self.episode_reset()
        for self.step_current in range(1, self.args.steps+1):
            # self.step_current = step
            self.step_episode += 1
            self.epsilon = self.update_epsilon(self.step_current)
            s, a, r, is_terminal = self.step()
            self.episode_reward += r
            self.memory.add(s, a, r, is_terminal)
            self.episode_losses.append(self.train_model())
            if not self.env.is_running():
                self.episode_cleanup()
                self.episode_reset()
            if self.step_current % (self.args.backup_frequency * self.args.steps) == 0:
                self.epoch_cleanup()
                self.epoch_reset()


class DiscretizedRandomAgent(Agent):
    """Simple random agent for DeepMind Lab."""

    def __init__(self, args, env, rng, paths):
        # Call super class
        super(DiscretizedRandomAgent, self).__init__(args, rng, env, paths)
        print('Starting random discretized agent.')

    def get_action(self, *args):
        """Gets an image state and a reward, returns an action."""
        return self.rng.randint(0, self.available_actions)

    def train(self):
        pass