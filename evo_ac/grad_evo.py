import copy
import torch
import numpy as np


class EvoACEvoAlg(object):
    def __init__(self, config_dict):
        self.evo_config = config_dict['evo_ac']
        self.net_config = config_dict['neural_net']

        #CONSTANTS
        self.num_mutate = self.evo_config['recomb_nums']
        self.learning_rate = self.evo_config['learning_rate']
        self.mut_scale = self.evo_config['mut_scale']

    def set_params(self, params):
        self.params = params

    def set_grads(self, grads):
        self.grads = grads

    def set_fitnesses(self, fitnesses):
        self.fitnesses = np.array(fitnesses)
    
    def select_parents(self):
        argsorted = np.argsort(-self.fitnesses)
        self.parent_params = []
        self.parent_grads = []
        for pop_place in range(len(self.num_mutate)):
            pop_idx = argsorted[pop_place]
            self.parent_params.append(copy.deepcopy(self.params[pop_idx]))
            self.parent_grads.append(copy.deepcopy(self.grads[pop_idx]))
    
    def measure_diversity(self):
        num_parameters = 0
        parameter_diverage = 0
        for layer_idx in range(len(self.params[0])):
            layer_size = torch.prod(torch.tensor(self.params[0][layer_idx].size())).item()
            layer_max = self.params[0][layer_idx]
            layer_min = self.params[0][layer_idx]
            for pop_idx in range(len(self.params)):
                layer_max = torch.max(layer_max, self.params[pop_idx][layer_idx])
                layer_min = torch.min(layer_min, self.params[pop_idx][layer_idx])
            
            layer_diff = layer_max - layer_min
            num_parameters += layer_size
            parameter_diverage += torch.sum(layer_diff).item()

        return parameter_diverage / num_parameters
    
    def create_new_pop(self):
        self.select_parents()
        next_gen = []
        for parent_idx in range(len(self.num_mutate)):
            parent_count = self.num_mutate[parent_idx]
            for child_count in range(parent_count):
                child = []
                params = self.parent_params[parent_idx]
                grads = self.parent_grads[parent_idx]
                for i in range(len(params)):
                    child.append(self.mutate(params[i], grads[i]))
                next_gen.append(child)
        self.params = next_gen
        return next_gen
    

    def mutate(self, param, grad):
        adjusted_grad = self.learning_rate * grad

        locs = param - adjusted_grad
        scales = torch.abs(adjusted_grad) * self.mut_scale

        norm_dist = torch.distributions.normal.Normal(locs, scales)
        return norm_dist.sample()

    def end_generation(self):
        self.learning_rate = self.learning_rate * self.evo_config['lr_decay']