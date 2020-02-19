from cosyne_base.cosyne import Cosyne as cs
import gym
import torch
import json
import sys

ENVIRONMENT = 'CartPole-v0'
gym_env = gym.make(ENVIRONMENT)


def eval_cartpole(nn):
    fitness = 0
    obs = gym_env.reset()
    
    while True:       
        action = nn.forward(torch.from_numpy(obs).float())
        #argmax
        action = action.max(0)[1].item()

        obs, reward, done, hazards = gym_env.step(action) 
        fitness += reward
        
        if done:
            break
            
    return fitness


#print(gym_env.action_space)
#print(gym_env.observation_space)
#print(gym_env.action_space.sample())

config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

cosyne = cs(config_dict)
cosyne.run(eval_cartpole)