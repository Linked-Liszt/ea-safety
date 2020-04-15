import copy
import torch
import random
import numpy as np


class EvoACEvoAlg(object):
    def __init__(self, config_dict):
        self.evo_config = config_dict['evo_ac']
        self.net_config = config_dict['neural_net']

        #CONSTANTS
        self.num_mutate = self.evo_config['recomb_nums'] # [4,3,2,1]
        self.learning_rate = self.evo_config['learning_rate'] #1e-3
        self.lr_decay = self.evo_config['lr_decay']
        self.mut_scale = self.evo_config['mut_scale'] #0.5

        num_children = 0
        if self.evo_config['hold_elite']:
            num_children += 1
        num_children += sum(self.num_mutate)

        if num_children != self.evo_config['pop_size']:
            raise RuntimeError(f"Number children ({num_children}) created does" +
                                f" not match population size ({self.evo_config['pop_size']})")

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
        if self.evo_config['hold_elite']:
            next_gen.append(self.parent_params[0])

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
        learning_rate = random.uniform(self.learning_rate[0], self.learning_rate[1])
        adjusted_grad = learning_rate * grad

        if self.evo_config['mutation_type'] == "gauss":
            locs = param - adjusted_grad
            scales = torch.abs(adjusted_grad) * self.mut_scale

            norm_dist = torch.distributions.normal.Normal(locs, scales)
            return norm_dist.sample()

        elif self.evo_config['mutation_type'] == "uniform":
            locs = param - adjusted_grad
            mutation_amount = torch.abs(adjusted_grad) * self.mut_scale
            dist = torch.distributions.uniform.Uniform(-mutation_amount, mutation_amount)
            print(dist.sample())
            return locs + dist.sample()

    

    def decary_lr(self):
        self.learning_rate = self.lr_decay * self.learning_rate