import numpy as np
import torch
import torch.nn as nn
import random
from  cosyne_base.neural_net import CosyneNet as cn
import copy
import math
import pickle


class Cosyne(object):
    def __init__(self, config):
        self.nn = cn(config['neural_net'])
        self.cosyne_config = config['cosyne']
        self._get_param_shape()
        self._init_subpopulations()
        self.best_fitness = -math.inf
        self.log = []

    #eval_network takes the NN as a param and returns fitness
    def run(self, eval_network):
        self.gen_idx = 0
        while self._termination():
            for param_index in range(self.cosyne_config['pop_size']):
                self._construct_network(param_index)
                fitness = eval_network(self.nn)
                self._save_best_nn(fitness)
                self.fitnesses[:,param_index] = fitness
            self._recombination()

            self._save_data()
            self._print_info(self.gen_idx)
            self.gen_idx += 1

    def _save_best_nn(self, fitness):
        if fitness >= self.best_fitness:
            self.best_nn = copy.deepcopy(self.nn)
            self.best_fitness = fitness

    def _save_data(self):
        data_dict = {}
        data_dict['gen'] = self.gen_idx
        data_dict['fit_best'] = np.max(self.fitnesses[0])
        data_dict['fit_mean'] = np.mean(self.fitnesses[0])
        data_dict['fit_med'] = np.median(self.fitnesses[0])
        data_dict['fit_std'] = np.std(self.fitnesses[0])
        self.log.append(data_dict)

    def export_data(self):
        save_dict = {}
        save_dict['env'] = self.cosyne_config['env']
        save_dict['nn'] = self.best_nn
        save_dict['log'] = self.log
        pickle.dump(save_dict, open(self.cosyne_config['log_path'], 'wb')) 

    def _print_info(self, gen):
        curr_data_dict = self.log[-1]
        msg = f"Gen: {curr_data_dict['gen']}"
        msg += f"\nBest Fitness: {curr_data_dict['fit_best']:.2f}\tMedian: {curr_data_dict['fit_mean']:.2f}"
        msg += f"\nAvg: {curr_data_dict['fit_mean']:.2f}\tSTD: {curr_data_dict['fit_std']:.2f}"
        msg += '\n\n'
        print(msg)

    def _termination(self):
        if self.cosyne_config['terminate']['type'] == 'fit':
            return self.best_fitness < self.cosyne_config['terminate']['param']
        elif self.cosyne_config['terminate']['type'] == 'gen':
            return self.gen_idx < int(self.cosyne_config['terminate']['param'])

    def _recombination(self):
        for param_index in range(self.num_parameters):

            sorted_indices = np.argsort(self.fitnesses[param_index])
           
            sorted_pop = np.take_along_axis(
                            self.subpopulations[param_index],
                            sorted_indices, 0)

            sorted_fitnesses = np.take_along_axis(
                            self.fitnesses[param_index],
                            sorted_indices, 0)

            parents = np.flip(sorted_pop)[:self.cosyne_config['parent_count']]


            for replace_idx in range(self.cosyne_config['recomb_count']):
                new_value = self._mate_mutate(parents)
                sorted_pop[replace_idx] = new_value

            sorted_pop = self._permutate(sorted_pop, sorted_fitnesses)

            self.subpopulations[param_index] = sorted_pop

    def _permutate(self, population, fitnesses):
        permutate_markers = np.zeros(len(population))

        min_fit = np.min(fitnesses)
        max_fit = np.max(fitnesses)
        range_fit = max_fit - min_fit + 0.0001

        # Consider vectorization with numpy
        for pop_idx in range(len(population)):
            prob_permutate = 1 - (np.sqrt((fitnesses[pop_idx] - min_fit + 0.001)/range_fit))
            
            if random.uniform(0, 1) < self.cosyne_config['perm_mod'] * prob_permutate:
                permutate_markers[pop_idx] = 1

        #Returns tuple
        perm_indicies = np.where(permutate_markers == 1)[0]
        values = np.take(population, perm_indicies, 0)
        np.random.shuffle(values)

        for i in range(len(perm_indicies)):
            population[perm_indicies[i]] = values[i]

        return population

    def _mate_mutate(self, parents):
        if random.uniform(0, 1) < self.cosyne_config['mate_mutate_ratio']:
            return self._mutate(parents)
        else:
            return self._mate(parents)


    def _mutate(self, parents):
        new_val =  np.random.choice(parents) + np.random.normal(0, 0.2)
        return max(0.0, min(1.0, new_val)) #constrain to 0 and 1

    def _mate(self, parents):
        return np.mean(np.random.choice(parents, 2))


    def _init_subpopulations(self):
        self.subpopulations = np.random.rand(
                                self.num_parameters,
                                self.cosyne_config['pop_size'])

        self.fitnesses = np.zeros((self.num_parameters, 
                                    self.cosyne_config['pop_size']))

    def _construct_network(self, pop_index):
        flat_params = self.subpopulations[:,pop_index]
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