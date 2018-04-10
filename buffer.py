import numpy as np
import logging
_logger = logging.getLogger(__name__)


class Buffer(object):
    """ A buffer to collect observations until they form a state. """
    def __init__(self, sequence_length, width, height, color_channels):
        _logger.info("Initializing new object of type " + str(type(self).__name__))
        self.buffer = np.zeros((sequence_length,
                                width,
                                height,
                                color_channels), dtype=np.uint8)
        self.buffer_size = np.shape(self.buffer)
        
    def add(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        
    def get_state(self):
        return self.buffer
    
    def reset(self):
        self.buffer *= 0