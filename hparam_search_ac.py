import numpy as np
import gym
import json
import sys
from evo_ac.runner import EvoACRunner
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)


    for hparam_run_idx in range(300):
        config_dict['experiment']['log_name'] = f'evo_ac_hparam_{hparam_run_idx}'
        config_dict['evo_ac']['learning_rate'] = random.uniform(1e-4, 1e-7)
        config_dict['neural_net']['learning_rate'] = random.uniform(1e-3, 1e-7)
        config_dict['evo_ac']['mut_scale'] = random.uniform(0.2, 1.0)

        shared_size = random.randint(0, 1)
        hidden_size = random.randint(1, 8) * 64
        shared = policy = value = None
        if shared_size == 0:
            shared = [
                {
                    "type": "Linear",
                    "params": [8, hidden_size], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": "ReLU",
                    "params": [], 
                    "kwargs": {}
                }
                ]
        elif shared_size == 1:
            shared = [
                {
                    "type": "Linear",
                    "params": [8, hidden_size], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": "ReLU",
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [hidden_size, hidden_size], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": "ReLU",
                    "params": [], 
                    "kwargs": {}
                }
                ]

        value_size = random.randint(0, 1)
        if value_size == 0:
            value = [
                {
                    "type": "Linear",
                    "params": [hidden_size, 1], 
                    "kwargs": {"bias":True}
                }
                ]
        elif value_size == 1:
            value = [
                {
                    "type": "Linear",
                    "params": [hidden_size, hidden_size], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": "ReLU",
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [hidden_size, 1], 
                    "kwargs": {"bias":True}
                }
                ]
        

        policy_size = random.randint(0, 1)
        if policy_size == 0:
            policy = [
                {
                    "type": "Linear",
                    "params": [hidden_size, 4], 
                    "kwargs": {"bias":True}
                }
                ]
        elif policy_size == 1:
            policy = [
                {
                    "type": "Linear",
                    "params": [hidden_size, hidden_size], 
                    "kwargs": {"bias":True}
                }, 
                {
                    "type": "ReLU",
                    "params": [], 
                    "kwargs": {}
                },
                {
                    "type": "Linear",
                    "params": [hidden_size, 4], 
                    "kwargs": {"bias":True}
                }
                ]


        config_dict['neural_net']['shared'] = shared
        config_dict['neural_net']['policy'] = policy
        config_dict['neural_net']['value'] = value

        runner = EvoACRunner(config_dict)
        runner.run_experiment()