import copy
import torch
import numpy as np


class EvoACEvoAlg(object):
    def __init__(self, config_dict):
        self.evo_config = config_dict['evo_ac']
        self.net_config = config_dict['neural_net']

        #CONSTANTS
        self.num_mutate = self.evo_config['recomb_nums'] # [4,3,2,1]
        self.learning_rate = self.evo_config['learning_rate'] #1e-3
        self.mut_scale = self.evo_config['mut_scale'] #0.5

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