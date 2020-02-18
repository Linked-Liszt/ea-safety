import unittest
import json
import numpy as np
import torch
from cosyne_base.cosyne import Cosyne as cs

class TestCosyne(unittest.TestCase):

    def setUp(self):
        self.config_dict = {
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

