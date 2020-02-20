import gym
import torch
import json
import sys
import pickle

def demo_best_net(nn):
    obs = gym_env.reset()
    fitness = 0
    while True:
        gym_env.render()
        action = nn.forward(torch.from_numpy(obs).float())
        #argmax
        action = action.max(0)[1].item()

        obs, reward, done, hazards = gym_env.step(action) 
        fitness += reward
        
        if done:
            break
            
    print(f"Demo Fitness: {fitness}")

nn_path = sys.argv[1]

nn_dict = pickle.load(open(nn_path, 'rb'))

gym_env = gym.make(nn_dict['env'])

demo_best_net(nn_dict['nn'])