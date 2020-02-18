from cosyne_base.neural_net import CosyneNet as cn
from cosyne_base.cosyne import Cosyne as cs
import json
import numpy as np
import sys
import torch

switcher = 1

config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

if switcher == 0:
    test_cn = cn(config_dict)

    test_input = torch.from_numpy(np.random.rand(5)).float()

    #print(test_cn)
    #print(test_cn.forward(test_input))

    print(test_cn.extract_parameters())

elif switcher == 1:
    test_cs = cs(config_dict)
    print(test_cs.param_sizes)
    print(test_cs.param_flattened_sizes)
