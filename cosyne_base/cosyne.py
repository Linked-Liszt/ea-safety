import numpy as np
import torch
import torch.nn as nn
from neural_net import CosyneNet as cs


class Cosyne(object):
    def __init__(self, config):
        self.nn = cs(config['neural_net'])
        self.__get_param_shape()

    def __get_param_shape(self):
        parameters = self.nn.extract_parameters()
        self.param_sizes = []
        self.param_flattened_sizes = []
        for param in parameters:
            self.param_sizes.append(param.size())
            self.param_flattened_sizes.append(param.view(-1, 1).size()[0])

    def __reconstruct_params(self, flat_params):
        pass