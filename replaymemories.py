from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import logging
_logger = logging.getLogger(__name__)


class ReplayMemory(object):
    def __init__(self, args, rng, input_shape):
        _logger.info("Initializing ReplayMemory (size: %i)" %
                     args.memory_size)
        self.args = args
        self.rng = rng
        
        # set the right size of the individual variables
        self.s = np.zeros((self.args.memory_size,) + input_shape, dtype=np.uint8)
        self.a = np.zeros(self.args.memory_size, dtype=np.int32)
        self.r = np.zeros(self.args.memory_size, dtype=np.float32)
        self.is_terminal = np.zeros(self.args.memory_size, dtype=np.float32)

        # initialize memory
        self.size = 0
        self.pos = 0


class SimpleReplayMemory(ReplayMemory):
    def __init__(self, args, rng, input_shape):
        # Call super class
        super(SimpleReplayMemory, self).__init__(args, rng, input_shape)

    def add(self, state, action, reward, isterminal):
        # save at correct position in memory
        self.s[self.pos, ...] = state
        self.a[self.pos] = action
        self.r[self.pos] = reward
        self.is_terminal[self.pos] = isterminal
        # update position and current memory size
        self.pos = (self.pos + 1) % self.args.memory_size
        self.size = min(self.size + 1, self.args.memory_size)

    def get_batch(self):
        s_ids = self.rng.randint(0, self.size - 2, self.args.batch_size).tolist()
        s_prime_ids = []
        for s_id in s_ids:
            s_prime_ids.append(s_id + 1)
        return (self.s[s_ids],
                self.a[s_ids],
                self.r[s_ids],
                self.s[s_prime_ids],
                self.is_terminal[s_ids])
    
    
class PrioritizedReplayMemory(SimpleReplayMemory):
    def __init__(self, args, rng, input_shape):
        # Call super class
        super(PrioritizedReplayMemory, self).__init__(args, rng)
        
        # set a factor for the importance of the prioritization
        self.priority_factor = self.args.priority_factor
        

    def add(self, state, action, reward, isterminal):
        # save at correct position in memory
        self.s[self.pos, ...] = state
        self.a[self.pos] = action
        self.r[self.pos] = reward
        self.is_terminal[self.pos] = isterminal
        # update position and current memory size
        self.pos = (self.pos + 1) % self.args.memory_size
        self.size = min(self.size + 1, self.args.memory_size)

    def get_batch(self):
        s_ids = self.rng.randint(0, self.size - 2, self.args.batch_size).tolist()
        s_prime_ids = []
        for s_id in s_ids:
            s_prime_ids.append(s_id + 1)
        return (self.s[s_ids],
                self.a[s_ids],
                self.r[s_ids],
                self.s[s_prime_ids],
                self.is_terminal[s_ids])

    
"""  
    prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
"""  