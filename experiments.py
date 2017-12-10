from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import numpy as np

_logger = logging.getLogger(__name__)


class Experiment(object):
    """
    Base class for all experiment implementations.
    """
    def __init__(self, args, paths):
        _logger.info("Initializing Experiment (Steps: %i)" %
                     args.steps)
        self.args = args
        self.paths = paths

        # Initialize important stats
        self.time_start = time.time()
        self.time_current = time.time()
        self.episodes = 0
        self.episodes_success = 0
        self.steps_current = 0
        self.steps_episode = 0
        self.reward_total = 0
        self.reward_episode = 0

        # Mersenne Twister pseudo-random number generator
        self.rng = np.random.RandomState(self.args.random_seed)

        # Initialize environment
        if self.args.env == 'LabLimitedActions':
            from cowhi.environments import LabLimitedActions
            self.env = LabLimitedActions(self.args, self.rng)
        elif self.args.env == 'LabAllActions':
            from cowhi.environments import LabAllActions
            self.env = LabAllActions(self.args, self.rng)

        # Initialize agent
        if self.args.agent == 'SimpleDQNAgent':
            from cowhi.agents import SimpleDQNAgent
            self.agent = SimpleDQNAgent(self.args, self.rng, self.env, self.paths)
        elif self.args.agent == 'DiscretizedRandomAgent':
            from cowhi.agents import DiscretizedRandomAgent
            self.agent = DiscretizedRandomAgent(self.args, self.rng, self.env, self.paths)

    def run(self):
        """ Organizes the training of the agent. """

        if not self.args.play:
            # Test and backup untrained agent
            self.agent.play()
            # Train the agent
            self.agent.train()

        # Test and backup final agent
        # TODO overrides last epoch
        self.agent.play()
