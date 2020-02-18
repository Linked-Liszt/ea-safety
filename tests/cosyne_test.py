import unittest
import json
import numpy as np
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
        

if __name__ == '__main__':
    unittest.main()