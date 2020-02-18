import numpy as np
import torch
import torch.nn as nn
from neural_net import CosyneNet as cs


class Cosyne(object):
    def __init__(self, config):
        self.nn = cs(config['neural_net'])

    def __get_weight_shape(self):
        pass

    