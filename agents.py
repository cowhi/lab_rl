from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import logging
import os
import sys

import time

import cv2
# import glob

from lab_rl.buffer import Buffer
from lab_rl.helper import get_human_readable, get_softmax, plot_experiment, print_stats
from lab_rl.models import SimpleDQNModel
from lab_rl.replaymemories import SimpleReplayMemory

import numpy as np

import six

from six.moves import range

import tensorflow as tf

_logger = logging.getLogger(__name__)

# Ignore all numpy warnings during training
np.seterr(all='ignore')


class Agent(object):
    def __init__(self, args, rng, env, paths=None):
        _logger.info("Initializing Agent (type: %s, load_model: %s)" %
                     (args.agent,
                      str(isinstance(args.load_model,
                                     six.string_types))))
        self.args = args
        self.rng = rng
        self.env = env
        self.paths = paths
        self.episode_reward = 0.0
        self.episode_losses = []
        self.epoch_rewards = []
        self.available_actions = self.env.count_actions()
        self.exploration_method = self.args.exploration_method
        self.epsilon = self.args.epsilon_start
        self.tau = self.args.tau_start
        self.tau_diff = 0.0
        self.loss = 0

        self.step_penalty = self.args.step_penalty
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
        self.target_model = None
        self.model_name = None
        self.model_last = None
        self.model_input_shape = None
        self.observation_shape = None
        # Model parameter
        self.session = None
        self.saver = None
        self.start_time = time.time()
        self.episode_start_time = self.start_time
        self.epoch_start_time = self.start_time
        self.batch_size = self.args.batch_size
        # Initialize a buffer to turn observations into states
        self.buffer = Buffer(self.args.sequence_length,
                             self.args.input_width,
                             self.args.input_height,
                             self.args.color_channels)
        # define learning parameter
        self.epsilons = self.generate_epsilons()
        self.taus = self.generate_taus()
        # if self.exploration_method == "epsilon":
        #    self.epsilons = self.generate_epsilons()
        #    self.tau = self.args.tau_start
        #elif self.exploration_method == "tau":
        #    self.taus = self.generate_taus()
        #    self.epsilon = self.args.epsilon_start
        #    #print(self.taus.shape)

    def generate_epsilons(self):
        if self.exploration_method == "epsilon":
            epsilon_decline = np.linspace(
                    self.args.epsilon_start,
                    self.args.epsilon_min,
                    int(self.args.epsilon_decay * self.args.steps))
            rest = self.args.epsilon_min * \
                    np.ones(self.args.steps - epsilon_decline.shape[0])
            return np.append(epsilon_decline, rest)
        else:
            return np.ones(self.args.steps)
    
    def generate_taus(self):
        if self.exploration_method == "tau":
            rate = ((self.args.tau_start/self.args.tau_min)**(1./(int(self.args.tau_decay*self.args.steps)))) - 1
            taus = []
            tau = self.args.tau_start
            for i in range(int(self.args.tau_decay*self.args.steps)):
                if tau > self.args.tau_min:
                    taus.append(tau)
                else:
                    break
                tau -= tau * rate
            for i in range(self.args.steps - len(taus)):
                taus.append(self.args.tau_min)
            return np.array(taus)
        else:
            return np.ones(self.args.steps)
    
    def prepare_csv_train(self):
        first_row = ('episode',
                     'time',
                     'duration',
                     'steps',
                     'reward',
                     'steps_total',
                     'reward_total',
                     'tau',
                     'epsilon',
                     'loss',
                     'batch_size')
        self.csv_train_file = open(
                os.path.join(self.paths['log_path'], 'stats_train.csv'), "wb")
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
        self.csv_test_file = open(
                os.path.join(self.paths['log_path'], 'stats_test.csv'), "wb")
        self.csv_test_writer = csv.writer(self.csv_test_file)
        self.csv_write_test(first_row)

    def csv_write_test(self, row):
        self.csv_test_writer.writerow(row)
        self.csv_test_file.flush()

    def preprocess_input(self, img):
        if self.args.color_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.args.input_width, self.args.input_height))
        # print("img:", img.shape)
        # print("model:", self.model_input_shape)
        # return np.reshape(img, self.model_input_shape)
        # print("observation:", self.observation_shape)
        return np.reshape(img, self.observation_shape)

    def get_action(self, *args):
        pass

    def step(self):
        o = self.preprocess_input(self.env.get_observation())
        self.buffer.add(o)
        s = self.buffer.get_state()
        a = self.get_action(s[None, ...])
        r = self.env.step(a)
        is_terminal = not self.env.is_running() or r % 2 == 1
        return s, a, r - self.step_penalty, is_terminal

    def episode_reset(self):
        self.episode += 1
        # print('Episode', self.episode, 'START')
        self.episode_reward = 0
        self.step_episode = 0
        self.episode_losses = []
        self.env.reset()
        self.episode_start_time = time.time()

    def episode_cleanup(self):
        # print('Episode', self.episode, 'END')
        self.epoch_rewards.append(self.episode_reward)
        self.total_reward += self.episode_reward
        # TODO: Figure out why len(self.episode_losses) is sometimes 0 
        if not len(self.episode_losses) == 0:
            self.loss = sum(self.episode_losses)/len(self.episode_losses)
        else:
            self.loss = 0.0
        new_row = (self.episode,  # current episode
                   "{0:.1f}".format(time.time() - self.start_time),
                   # total time
                   "{0:.1f}".format(time.time() - self.episode_start_time),
                   # episode duration
                   self.step_episode,  # steps per episode
                   self.episode_reward,  # reward per episode
                   self.step_current,  # total steps so far
                   self.total_reward,  # total reward so far
                   "{0:.4f}".format(self.tau),  # current tau
                   "{0:.5f}".format(self.epsilon),  # current epsilon
                   "{0:.4f}".format(self.loss),  # avg loss per batch
                   self.batch_size  # current batch size used for training
                   )
        # print(new_row)
        # print('Train Episode', self.episode,
        #       '(steps:', self.step_current,')finished.
        #       Reward:', self.episode_reward)
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

        new_row = (self.epoch,  # current epoch
                   "{0:.1f}".format(time.time() - self.start_time),
                   # total time
                   "{0:.1f}".format(time.time() - self.epoch_start_time),
                   # epoch duration
                   self.step_current,  # total steps so far
                   self.episode,  # total episodes so far
                   "{0:.4f}".format(test_reward),
                   # avg reward per episode during testing
                   "{0:.4f}".format(test_steps)
                   # avg steps per episode during testing
                   )
        # if self.test_reward_best <= test_reward:
        # self.test_reward_best = test_reward
        # self.model_name = "DQN_{:04}".format(
        #    int(self.step_current /
        #        (self.args.backup_frequency * self.args.steps)))
        # for old in glob.glob(
        #       os.path.join(self.paths['model_path'],'DQN_epoch_*')):
        #    os.remove(old)
        self.model_name = 'DQN_epoch_{:04}'.format(self.epoch)
        self.model_last = os.path.join(self.paths['model_path'],
                                       self.model_name)
        self.saver.save(self.session, self.model_last)
        _logger.info("Saved network after epoch %i (%i steps): %s" %
                     (self.epoch, self.step_current, self.model_name))
        if not self.args.play:
            self.csv_write_test(new_row)
            if self.epoch > 0:
                plot_experiment(self.paths['log_path'], 'stats_train', 'episode')
                plot_experiment(self.paths['log_path'], 'stats_test', 'epoch')

    def test(self, episodes):
        if self.exploration_method == "tau":
            backup_tau = self.tau
            self.tau = self.args.tau_min
        elif self.exploration_method == "epsilon":
            backup_epsilon = self.epsilon
            self.epsilon = self.args.epsilon_min
        
        print('TESTING')
        episode_rewards = []
        episode_steps = []
        for episode in range(0, episodes):
            save_video = False
            # self.bla = episode
            if episode == 0 and self.args.save_video:
                save_video = True
            # print('Test episode %i' % episode)
            reward, steps = self.play(save_video)
            episode_rewards.append(reward)
            episode_steps.append(steps)
        print_stats(sum(episode_steps),
                    sum(episode_steps),
                    episode_rewards,
                    time.time() - self.start_time)
        if self.exploration_method == "tau":
            self.tau = backup_tau
        elif self.exploration_method == "epsilon":
            self.epsilon = backup_epsilon
        
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
            out_video = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    self.args.fps,
                    (self.args.width, self.args.height))
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
            observation = self.preprocess_input(state_raw)
            self.buffer.add(observation)
            state = self.buffer.get_state()
            action = self.get_action(state[None, ...])
            # epsilon = 0.05, tau = 0.1
            # reward_old = reward_total
            for _ in range(self.args.frame_repeat):
                if self.args.show:
                    cv2.imshow("frame-test", state_raw)
                    cv2.waitKey(20)
                if save_video:
                    out_video.write(state_raw.astype('uint8'))
                reward = self.env.step(action, 1)
                reward_total += reward
                # if reward_total != reward_old:
                #    print("New reward:", reward, 'Sum:', reward_total)
                #    reward_old = reward_total
                if not self.env.is_running() or reward_total % 2 == 1:
                    break
                state_raw = self.env.get_observation()
        if save_video:
            out_video.release()
            print("Saved video (fps:%i, size:%s) to: %s [%s]" %
                  (self.args.fps,
                   str((self.args.width, self.args.height)),
                   video_path,
                   get_human_readable(os.path.getsize(video_path))))
        if self.args.show:
            cv2.destroyAllWindows()

'''
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
        self.model_input_shape = (self.args.input_width,
                                  self.args.input_height) + \
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
            self.model_name = self.args.load_model.split("/")[-1]
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)
        if not self.args.play:
            # Backup initial model weights
            self.model_name = "DQN_0000"
            self.model_last = os.path.join(self.paths['model_path'],
                                           self.model_name)
            self.saver.save(self.session, self.model_last)

    def train_model(self):
        # train model with random batch from memory
        # if self.step_current % int((1/5)*self.args.steps) == 0:
        #    self.batch_size *= 2
        if self.memory.size > 2 * self.batch_size:
            s, a, r, s_prime, is_terminal = self.memory.get_batch()
            qs = self.model.get_qs(s)
            # print('Qs', qs[0])
            max_qs_prime = np.max(self.model.get_qs(s_prime), axis=1)
            # print('a', a[0], 'r', r[0], 'gamma',
            #       self.args.gamma, 'q_s_prime', max_qs[0])
            qs[np.arange(qs.shape[0]), a] = r + (1 - is_terminal) * (
                    self.args.gamma * max_qs_prime)
            # print('Qs_updated', qs[0])
            return self.model.train(s, qs)
        return 0.0

    """
    def update_epsilon(self, steps):
        # Update epsilon if necessary
        if steps > self.args.epsilon_decay * self.args.steps:
            return self.args.epsilon_min
        return self.args.epsilon_start - \
               steps * (self.args.epsilon_start - self.args.epsilon_min) / \
               (self.args.epsilon_decay * self.args.steps)
    

    def update_tau(self):
        # Update tau if necessary
        if self.tau <= self.args.tau_min:
            return self.args.tau_min
        return self.tau - self.tau * self.args.tau_decay
    """
    
    def generate_taus(self):
        rate = ((self.args.tau_start/self.args.tau_min)**(1./(int(self.args.tau_decay*self.args.steps)))) - 1
        taus = []
        tau = self.args.tau_start
        for i in range(int(self.args.tau_decay*self.args.steps)):
            if tau > self.args.tau_min:
                taus.append(tau)
            else:
                break
            tau -= tau * rate
        for i in range(self.args.steps - len(taus)):
            taus.append(self.args.tau_min)
        return np.array(taus)
    
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
            if self.step_current % \
                    (self.args.backup_frequency * self.args.steps) == 0:
                if self.env.is_running() or not is_terminal:
                    self.episode_cleanup()
                self.epoch_cleanup()
                self.epoch_reset()
                if not self.step_current == self.args.steps:
                    print("TRAINING")
                    self.episode_reset()
'''

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


class DQNAgent(Agent):
    def __init__(self, args, rng, env, paths):
        # Call super class
        super(DQNAgent, self).__init__(args, rng, env, paths)
        print('Starting DQN agent.')

        # Check if no session is active
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
        if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) > 0:
            # print(len(variables), '#####  BEFORE RESETING  ####################################')
            # print('\n'.join([ str(variable) for variable in variables ]))
            tf.reset_default_graph()
            # tf.get_variable_scope().reuse_variables()
            self.session = None
            # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
            # print(len(variables), '#####  AFTER RESETING  #####################################')
            # print('\n'.join([ str(variable) for variable in variables ]))
            # print('############################################################')
        
        # Prepare model
        tf.set_random_seed(self.args.random_seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        # Initiate tensorflow session
        self.session = tf.Session(config=config)
        
        # self.model_input_shape = (self.args.input_width,
        #                          self.args.input_height) + \
        #                         (self.args.color_channels,)
        # print("Old:", self.model_input_shape)
        self.model_input_shape = (self.args.sequence_length,) + \
                                 (self.args.input_width, self.args.input_height) + \
                                 (self.args.color_channels,)
        self.observation_shape = (self.args.input_width,
                                  self.args.input_height) + \
                                 (self.args.color_channels,)
        # Replay memory
        self.memory = SimpleReplayMemory(self.args,
                                         self.rng,
                                         self.model_input_shape)
        # Policy network
        self.model = SimpleDQNModel(self.args,
                                    self.rng,
                                    self.session,
                                    self.model_input_shape,
                                    self.available_actions,
                                    self.paths['model_path'],
                                    'policy')
        # Target network
        self.target_model = SimpleDQNModel(self.args,
                                           self.rng,
                                           self.session,
                                           self.model_input_shape,
                                           self.available_actions,
                                           self.paths['model_path'],
                                           'target')
        
        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='policy'))
        if self.args.load_model is not None:
            self.saver.restore(self.session, self.args.load_model)
            self.model_name = self.args.load_model.split("/")[-1]
        else:
            # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # print(len(variables), '#####  BEFORE INIT  #####################################')
            # print('\n'.join([ str(variable) for variable in variables ]))
            # print('############################################################')
            init = tf.global_variables_initializer()
            self.session.run(init)
        # We want to have two similar networks
        self.copy_model_parameters(self.model.scope, self.target_model.scope)
        # if not self.args.play:
        #    # Backup initial model weights
        #    self.model_name = "DQN_epoch_0000"
        #    self.model_last = os.path.join(self.paths['model_path'],
        #                                   self.model_name)
        #    self.saver.save(self.session, self.model_last)

    def train_model(self):
        # train model with random batch from memory
        # if self.step_current % int((1/5)*self.args.steps) == 0:
        #    self.batch_size *= 2
        # TODO: add dynamics (combine 4 observations to one state)
        if self.memory.size > 2 * self.batch_size:
            s, a, r, s_prime, is_terminal = self.memory.get_batch()
            # values from current policy
            qs = self.model.get_qs(s)
            # print('Qs', qs[0])
            # TODO: values from target network!
            # max_qs = np.max(self.model.get_qs(s_prime), axis=1)
            max_qs = np.max(self.target_model.get_qs(s_prime), axis=1)
            # print('a', a[0], 'r', r[0], 'gamma',
            #       self.args.gamma, 'q_s_prime', max_qs[0])
            qs[np.arange(qs.shape[0]), a] = r + (1 - is_terminal) * (
                    self.args.gamma * max_qs)
            # print('Qs_updated', qs[0])
            # Training of current policy!
            return self.model.train(s, qs)
        return 0.0

    def copy_model_parameters(self, source, target):
        """ Copies the trainable network parameters from the
            current policy network to the target network
        """
        policy_params = [t for t in tf.trainable_variables()
                         if t.name.startswith(source)]
        policy_params = sorted(policy_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables()
                         if t.name.startswith(target)]
        target_params = sorted(target_params, key=lambda v: v.name)

        update_ops = []
        for policy_v, target_v in zip(policy_params, target_params):
            op = target_v.assign(policy_v)
            update_ops.append(op)

        self.session.run(update_ops)
    
    def get_action(self, state):
        """ Returns an action selected through softmax. """
        # TODO: Get actions from model
        if self.exploration_method == "epsilon":
            if self.rng.random_sample() < self.epsilon:
                # select random action
                return self.rng.choice(self.available_actions)
            else:
                # if not random choose action with highest Q-value
                return np.argmax(self.model.get_qs(state)) 
        if self.exploration_method == "tau":
            # print(self.step_current, self.tau, self.batch_size)
            return self.rng.choice(self.available_actions,
                                   p=get_softmax(self.model.get_qs(state),
                                                 self.tau))

    def train(self):
        self.epoch_cleanup()
        self.epoch_reset()
        print("TRAINING")
        self.episode_reset()
        self.run_test = False
        for self.step_current in range(1, self.args.steps+1):
            self.step_episode += 1
            # self.tau = self.update_tau()
            # print(self.step_current)
            # if self.exploration_method == "tau":
            #    self.tau = self.taus[self.step_current-1]
            # elif self.exploration_method == "epsilon":
            #    self.epsilon = self.epsilons[self.step_current-1]
            self.tau = self.taus[self.step_current-1]
            self.epsilon = self.epsilons[self.step_current-1]
            s, a, r, is_terminal = self.step()
            self.episode_reward += r
            self.memory.add(s, a, r, is_terminal)
            self.episode_losses.append(self.train_model())
            # End episode if necessary
            if not self.env.is_running() or \
                    is_terminal or \
                    self.step_current == self.args.steps:
                self.episode_cleanup()
                if self.run_test:
                    self.epoch_cleanup()
                    self.epoch_reset()
                    self.run_test = False
                    if not self.step_current == self.args.steps:
                        print("TRAINING")
                self.episode_reset()
            # Copy network weights from time to time
            if self.step_current % self.args.target_update_frequency == 0:
                self.copy_model_parameters(self.model.scope, self.target_model.scope)
            # End epoch if necessary
            if self.step_current % \
                    (self.args.backup_frequency * self.args.steps) == 0 or \
                    self.step_current == self.args.steps - 1:
                self.run_test = True
        self.session.close()
            

class ADAAPTAgent(Agent):
    """ This is static for max 3 sources. """
    def __init__(self, args, rng, env, paths):
        # Call super class
        super(ADAAPTAgent, self).__init__(args, rng, env, paths)
        print('Starting ADAAPT agent.')

        # Check if no session is active
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
        if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) > 0:
            # print(len(variables), '#####  BEFORE RESETING  ####################################')
            # print('\n'.join([ str(variable) for variable in variables ]))
            tf.reset_default_graph()
            # tf.get_variable_scope().reuse_variables()
            self.session = None
            # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
            # print(len(variables), '#####  AFTER RESETING  #####################################')
            # print('\n'.join([ str(variable) for variable in variables ]))
            # print('############################################################')
        
        # Prepare model
        tf.set_random_seed(self.args.random_seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        # Initiate tensorflow session
        self.session = tf.Session(config=config)
        # self.model_input_shape = (self.args.input_width,
        #                          self.args.input_height) + \
        #                         (self.args.color_channels,)
        # print("Old:", self.model_input_shape)
        self.model_input_shape = (self.args.sequence_length,) + \
                                 (self.args.input_width, self.args.input_height) + \
                                 (self.args.color_channels,)
        self.observation_shape = (self.args.input_width,
                                  self.args.input_height) + \
                                 (self.args.color_channels,)
        # Replay memory
        self.memory = SimpleReplayMemory(self.args,
                                         self.rng,
                                         self.model_input_shape)
        
        # Policy network
        self.model = SimpleDQNModel(
            self.args,
            self.rng,
            self.session,
            self.model_input_shape,
            self.available_actions,
            self.paths['model_path'],
            'policy')
        # Target network
        self.target_model = SimpleDQNModel(
            self.args,
            self.rng,
            self.session,
            self.model_input_shape,
            self.available_actions,
            self.paths['model_path'],
            'target')
        
        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='policy'))
        
        self.args.load_source1 = '/home/ruben/.lab/2018-04-16_14-13_task_3a_adaapt/run_00/models/DQN_epoch_0000'
        self.args.load_source2 = '/home/ruben/.lab/2018-04-16_14-13_task_3a_adaapt/run_00/models/DQN_epoch_0001'
        self.args.load_source3 = '/home/ruben/.lab/2018-04-16_14-13_task_3a_adaapt/run_00/models/DQN_epoch_0002'
        # Load source models first if available
        self.count_source_models = 0
        self.source_models = {}
        sources = [self.args.load_source1, self.args.load_source2, self.args.load_source3]
        for source in sources:
            if source is not None:
                self.count_source_models += 1
                # Load source
                self.saver.restore(self.session, source)
                # Build source network
                self.source_models["source_" + str(self.count_source_models)] = SimpleDQNModel(
                    self.args,
                    self.rng,
                    self.session,
                    self.model_input_shape,
                    self.available_actions,
                    self.paths['model_path'],
                    'source' + str(self.count_source_models))
                # Copy model parameter to source model
                self.copy_model_parameters('policy', 'source' + str(self.count_source_models))
        
        # Importance network
        self.importance_model = SimpleDQNModel(
            self.args,
            self.rng,
            self.session,
            self.model_input_shape,
            len(self.source_models) + 1,
            self.paths['model_path'],
            'importance')
        
        if self.args.load_model is not None:
            self.saver.restore(self.session, self.args.load_model)
            self.model_name = self.args.load_model.split("/")[-1]
        else:
            # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # print(len(variables), '#####  BEFORE INIT  #####################################')
            # print('\n'.join([ str(variable) for variable in variables ]))
            # print('############################################################')
            init = tf.global_variables_initializer()
            self.session.run(init)
        # We want to have two similar networks
        self.copy_model_parameters(self.model.scope, self.target_model.scope)
        # if not self.args.play:
        #    # Backup initial model weights
        #    self.model_name = "DQN_epoch_0000"
        #    self.model_last = os.path.join(self.paths['model_path'],
        #                                   self.model_name)
        #    self.saver.save(self.session, self.model_last)

    def train_model(self):
        # train model with random batch from memory
        # if self.step_current % int((1/5)*self.args.steps) == 0:
        #    self.batch_size *= 2
        # TODO: add dynamics (combine 4 observations to one state)
        if self.memory.size > 2 * self.batch_size:
            s, a, r, s_prime, is_terminal = self.memory.get_batch()
            # values from current policy
            qs = self.get_weighted_qs(s)
            # print('Qs', qs[0])
            # TODO: values from target network!
            # max_qs = np.max(self.model.get_qs(s_prime), axis=1)
            max_qs = np.max(self.target_model.get_qs(s_prime), axis=1)
            # print('a', a[0], 'r', r[0], 'gamma',
            #       self.args.gamma, 'q_s_prime', max_qs[0])
            qs[np.arange(qs.shape[0]), a] = r + (1 - is_terminal) * (
                    self.args.gamma * max_qs)
            # print('Qs_updated', qs[0])
            # Training of current policy!
            return self.model.train(s, qs)
        return 0.0
    
    def get_weighted_qs(self, s):
        # Get importance vector from imporance network
        importance = self.importance_model.get_action_probs(s)
        print(importance)
        # TODO: Get q values from each network
        qs = {}
        qs[0] = self.model.get_qs(s)
        for i in range(1, len(self.source_models)+1):
            qs[i] = self.source_models.get_qs(s)
        # TODO: Calculate weighted Q values
        weighted_qs = qs
        return weighted_qs

    def copy_model_parameters(self, source, target):
        """ Copies the trainable network parameters from the
            current policy network to the target network
        """
        policy_params = [t for t in tf.trainable_variables()
                         if t.name.startswith(source)]
        policy_params = sorted(policy_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables()
                         if t.name.startswith(target)]
        target_params = sorted(target_params, key=lambda v: v.name)

        update_ops = []
        for policy_v, target_v in zip(policy_params, target_params):
            op = target_v.assign(policy_v)
            update_ops.append(op)

        self.session.run(update_ops)
    
    def get_action(self, state):
        """ Returns an action selected through softmax. """
        # TODO: Get output from importance network
        # TODO: get all qs from source and policy
        # TODO: Calculate weighted q values
        # TODO: Perform argmax
        # TODO: Get actions from model
        if self.exploration_method == "epsilon":
            if self.rng.random_sample() < self.epsilon:
                # select random action
                return self.rng.choice(self.available_actions)
            else:
                # if not random choose action with highest Q-value
                return np.argmax(self.model.get_qs(state)) 
        if self.exploration_method == "tau":
            # print(self.step_current, self.tau, self.batch_size)
            return self.rng.choice(self.available_actions,
                                   p=get_softmax(self.model.get_qs(state),
                                                 self.tau))

    def train(self):
        self.epoch_cleanup()
        self.epoch_reset()
        print("TRAINING")
        self.episode_reset()
        self.run_test = False
        for self.step_current in range(1, self.args.steps+1):
            self.step_episode += 1
            # self.tau = self.update_tau()
            # print(self.step_current)
            # if self.exploration_method == "tau":
            #    self.tau = self.taus[self.step_current-1]
            # elif self.exploration_method == "epsilon":
            #    self.epsilon = self.epsilons[self.step_current-1]
            self.tau = self.taus[self.step_current-1]
            self.epsilon = self.epsilons[self.step_current-1]
            s, a, r, is_terminal = self.step()
            self.episode_reward += r
            self.memory.add(s, a, r, is_terminal)
            self.episode_losses.append(self.train_model())
            # End episode if necessary
            if not self.env.is_running() or \
                    is_terminal or \
                    self.step_current == self.args.steps:
                self.episode_cleanup()
                if self.run_test:
                    self.epoch_cleanup()
                    self.epoch_reset()
                    self.run_test = False
                    if not self.step_current == self.args.steps:
                        print("TRAINING")
                self.episode_reset()
            # Copy network weights from time to time
            if self.step_current % self.args.target_update_frequency == 0:
                self.copy_model_parameters(self.model.scope, self.target_model.scope)
            # End epoch if necessary
            if self.step_current % \
                    (self.args.backup_frequency * self.args.steps) == 0 or \
                    self.step_current == self.args.steps - 1:
                self.run_test = True
        self.session.close()