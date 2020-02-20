import unittest
import json
import numpy as np
import torch
from cosyne_base.cosyne import Cosyne as cs
import copy

class TestCosyne(unittest.TestCase):

    def setUp(self):
        self.config_dict = {
            "cosyne": {
                "pop_size": 10,
                "parent_count": 4,
                "recomb_count": 4,
                "mate_mutate_ratio": 0.5,
                "perm_mod": 0.4,
                "terminate":
                {
                    "type": "gen",
                    "param": 5
                }

            },

            "neural_net": {
                "layers": [
                    {
                        "type": "Linear",
                        "params": [5, 5], 
                        "kwargs": {"bias":True}
                    }, 
                    {
                        "type": "ReLU",
                        "params": [], 
                        "kwargs": {}
                    }, 
                    {
                        "type": "Linear",
                        "params": [5, 5], 
                        "kwargs": {"bias":True}
                    }
                ]
            }
        }

    def test_reconstruct_params(self):
        test_cs = cs(self.config_dict)
        test_flat_param_arr = np.random.rand(np.sum(test_cs.param_flattened_sizes))
        output_params = test_cs._reconstruct_params(test_flat_param_arr)
        for i in range(len(output_params)):
            self.assertEqual(np.shape(output_params[i]), test_cs.param_sizes[i])
    
    def test_extract_param_sizes(self):
        test_cs = cs(self.config_dict)
        expected_sizes = [(5, 5), (5,), (5, 5), (5,)]
        for i in range(len(test_cs.param_flattened_sizes)):
            self.assertEqual(expected_sizes[i], tuple(test_cs.param_sizes[i]))

    def test_extract_flat_param_sizes(self):
        test_cs = cs(self.config_dict)
        expected_flat_sizes = [25, 5, 25, 5]
        self.assertEqual(expected_flat_sizes, test_cs.param_flattened_sizes)

    def test_insert_params(self):
        test_cs = cs(self.config_dict)
        test_flat_param_arr = np.zeros(np.sum(test_cs.param_flattened_sizes))
        test_cs._insert_params(test_flat_param_arr)
        self.assertTrue(torch.all(torch.eq(test_cs.nn.layers[0].weight, torch.zeros([5, 5]))))

    def test_init_subpopulations(self):
        test_cs = cs(self.config_dict)
        self.assertEqual(np.shape(test_cs.subpopulations), (60, 10))
        self.assertEqual(np.shape(test_cs.fitnesses), (60, 10))

    def test_construct_network(self):
        test_cs = cs(self.config_dict)
        test_cs._construct_network(0)
        first_25_params = test_cs.subpopulations[:,0][:25]
        first_25_params = first_25_params.reshape(5, 5)
        actual_tensor = test_cs.nn.layers[0].weight.data.numpy()
        self.assertTrue(np.allclose(first_25_params, actual_tensor))

    def test_forward_pass_after_construction(self):
        test_cs = cs(self.config_dict)
        test_cs._construct_network(0)
        test_input = np.random.rand(5)
        test_cs.nn.forward(torch.from_numpy(test_input).float())

    def test_smoke_recombination(self):
        test_cs = cs(self.config_dict)
        test_cs._recombination()
        test_cs._construct_network(0)
        test_input = np.random.rand(5)
        test_cs.nn.forward(torch.from_numpy(test_input).float())

    def test_smoke_recombination_output_sizes(self):
        test_cs = cs(self.config_dict)
        test_cs._recombination()
        self.assertEqual(np.shape(test_cs.subpopulations), (60, 10))
        self.assertEqual(np.shape(test_cs.fitnesses), (60, 10))

    def test_permutation_preserved_elements(self):
        test_cs = cs(self.config_dict)
        population = test_cs.subpopulations[0]
        output = test_cs._permutate(copy.deepcopy(population), test_cs.fitnesses[0])
        population = np.sort(population)
        output = np.sort(output)
        self.assertTrue(np.allclose(population, output))
    
    def dummy_eval(self, dummy_param):
        return 1.0

    def test_smoke_fitness_term(self):
        custom_dict = copy.deepcopy(self.config_dict)
        custom_dict['cosyne']['terminate']['type'] = 'gen'
        custom_dict['cosyne']['terminate']['param'] = -5
        test_cs = cs(custom_dict)
        test_cs.run(self.dummy_eval)

    def test_smoke_run(self):
        test_cs = cs(self.config_dict)
        test_cs.run(self.dummy_eval)