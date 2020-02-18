from neural_net import CosyneNet as cn
import json
import numpy as np
import sys
import torch

config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

test_cn = cn(config_dict)

test_input = torch.from_numpy(np.random.rand(5)).float()

#print(test_cn)
#print(test_cn.forward(test_input))

print(test_cn.extract_weights())