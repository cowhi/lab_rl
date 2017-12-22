from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import deepmind_lab
import six
import numpy as np

from collections import OrderedDict

import logging
_logger = logging.getLogger(__name__)


class Environment(object):

    ACTIONS = OrderedDict([
        ('look_left', np.array((-32, 0, 0, 0, 0, 0, 0), dtype=np.intc)),  # min/max: -512/0
        ('look_right', np.array((32, 0, 0, 0, 0, 0, 0), dtype=np.intc)),  # min/max: 0/512
        ('look_up', np.array((0, 10, 0, 0, 0, 0, 0), dtype=np.intc)),  # min/max: 0/512
        ('look_down', np.array((0, -10, 0, 0, 0, 0, 0), dtype=np.intc)),  # min/max: -512/0
        ('strafe_left', np.array((0, 0, -1, 0, 0, 0, 0), dtype=np.intc)),  # min/max: -1/0
        ('strafe_right', np.array((0, 0, 1, 0, 0, 0, 0), dtype=np.intc)),  # min/max: 0/1
        ('forward', np.array((0, 0, 0, 1, 0, 0, 0), dtype=np.intc)),  # min/max: 0/1
        ('backward', np.array((0, 0, 0, -1, 0, 0, 0), dtype=np.intc)),  # min/max: -1/0
        ('fire', np.array((0, 0, 0, 0, 1, 0, 0), dtype=np.intc)),  # min/max: 0/1
        ('jump', np.array((0, 0, 0, 0, 0, 1, 0), dtype=np.intc)),  # min/max: 0/1
        ('crouch', np.array((0, 0, 0, 0, 0, 0, 1), dtype=np.intc))  # min/max: 0/1
    ])

    def __init__(self, args, rng):
        _logger.info("Initializing Lab (Type: %s, FPS: %i, width: %i, height: %i, map: %s)" %
                     (args.env, args.fps, args.width, args.height, args.map))
        self.args = args
        self.rng = rng
        self.env = deepmind_lab.Lab(
            self.args.level_script,
            ["RGB_INTERLACED"],
            config={
                "fps": str(self.args.fps),
                "width": str(self.args.width),
                "height": str(self.args.height)
            })

        self.action_mapping = None
        self.actions = None

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def get_actions(self):
        actions = OrderedDict()
        for key, value in six.iteritems(self.action_mapping):
            actions[key] = self.ACTIONS[value]
        return actions

    def map_action(self, action):
        return self.actions[action]

    def count_actions(self):
        return len(self.actions)

    def get_observation(self):
        obs = self.env.observations()
        return cv2.cvtColor(obs["RGB_INTERLACED"], cv2.COLOR_RGB2BGR)

    def is_running(self):
        return self.env.is_running()

    def step(self, action, num_steps=None):
        if not num_steps:
            num_steps = self.args.frame_repeat
        return self.env.step(self.map_action(action),
                             num_steps=num_steps)

    @staticmethod
    def get_action_mapping():
        action_mapping = {
            0: 'look_left',
            1: 'look_right',
            2: 'look_up',
            3: 'look_down',
            4: 'strafe_left',
            5: 'strafe_right',
            6: 'forward',
            7: 'backward',
            8: 'fire',
            9: 'jump',
            10: 'crouch'}
        return action_mapping


class LabAllActions(Environment):
    def __init__(self, args, rng):
        # Call super class
        super(LabAllActions, self).__init__(args, rng)
        self.action_mapping = self.get_action_mapping()
        self.actions = self.get_actions()


class LabLimitedActions(Environment):
    def __init__(self, args, rng):
        # Call super class
        super(LabLimitedActions, self).__init__(args, rng)
        self.action_mapping = self.get_action_mapping()
        self.actions = self.get_actions()

    @staticmethod
    def get_action_mapping():
        action_mapping = {
            0: 'look_left',
            1: 'look_right',
            2: 'forward'}
        return action_mapping
