from cosyne_base.cosyne import Cosyne as cs
import gym
import torch
import json
import sys

ENVIRONMENT = 'CartPole-v1'
gym_env = gym.make(ENVIRONMENT)


def eval_cartpole(nn):
    fitness = 0
    
    for i in range(2):
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

def demo_best_net(nn):
    obs = gym_env.reset()

    while True:
        gym_env.render()
        action = nn.forward(torch.from_numpy(obs).float())
        #argmax
        action = action.max(0)[1].item()

        obs, reward, done, hazards = gym_env.step(action) 
        
        if done:
            break
            

#print(gym_env.action_space)
#print(gym_env.observation_space)
#print(gym_env.action_space.sample())

config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

cosyne = cs(config_dict)
cosyne.run(eval_cartpole)
demo_best_net(cosyne.best_nn)
gym_env.close()