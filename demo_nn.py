import gym
import torch
import json
import sys
import pickle

nn_path = sys.argv[1]

nn_dict = pickle.load(open(nn_path, 'rb'))

ENVIRONMENT = nn_dict['env']
gym_env = gym.make(ENVIRONMENT)


#For performance
if ENVIRONMENT == 'CartPole-v1':
    ENV_SWITCHER = 0
elif ENVIRONMENT == 'Ant-v2':
    ENV_SWITCHER = 1

def demo_best_net(nn):
    obs = gym_env.reset()
    fitness = 0
    while True:
        gym_env.render()
        action = nn.forward(torch.from_numpy(obs).float())

        if ENV_SWITCHER == 0:
            #argmax
            action = action.max(0)[1].item()
        elif ENV_SWITCHER == 1:
            action = action.detach().numpy()
            action -= 0.5
            action *= 2

        print(action)
        obs, reward, done, hazards = gym_env.step(action) 
        fitness += reward
        
        if done:
            break
            
    print(f"Demo Fitness: {fitness}")



demo_best_net(nn_dict['nn'])