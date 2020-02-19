import numpy as np
import torch
import torch.nn as nn
from  cosyne_base.neural_net import CosyneNet as cn


class Cosyne(object):
    def __init__(self, config):
        self.nn = cn(config['neural_net'])
        self.cosyne_config = config['cosyne']
        self._get_param_shape()
        self._init_subpopulations()


    def _init_subpopulations(self):
        self.subpopulations = np.random.rand(
                                self.num_parameters,
                                self.cosyne_config['pop_size']
                                )
        self.fitnesses = np.zeros(
                            (self.num_parameters, self.cosyne_config['pop_size'])
                            )
    
    def _construct_network(self, param_index):
        flat_params = self.subpopulations[:,param_index]
        self._insert_params(flat_params)

    def _insert_params(self, flat_params):
        reshaped_params = self._reconstruct_params(flat_params)
        self.nn.insert_parameters(reshaped_params)

    def _get_param_shape(self):
        parameters = self.nn.extract_parameters()
        self.param_sizes = []
        self.param_flattened_sizes = []
        for param in parameters:
            self.param_sizes.append(param.size())
            self.param_flattened_sizes.append(param.view(-1, 1).size()[0])
        self.num_parameters = np.sum(self.param_flattened_sizes)

    
    def _reconstruct_params(self, flat_params):
        #Remove last last indicie for the np.split function
        split_indicies = np.cumsum(self.param_flattened_sizes)[:-1]
        split_params = np.split(flat_params, split_indicies)
        
        #Take the flattened params and reshape them into original sizes
        reshaped_params = []
        for i in range(len(split_params)):
            reshaped_params.append(
                    np.reshape(split_params[i], 
                    self.param_sizes[i]))
        
        return reshaped_params