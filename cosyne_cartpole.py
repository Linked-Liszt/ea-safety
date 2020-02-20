from cosyne_base.cosyne import Cosyne as cs
import gym
import torch
import json
import sys
import pickle

ENVIRONMENT = 'CartPole-v1'
gym_env = gym.make(ENVIRONMENT)


def eval_cartpole(nn):
    fitness = 0.01
    
    for i in range(5):
        obs = gym_env.reset()
        if fitness > 500.0 * i:
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


def save_best_network(nn):
    save_dict = {}
    save_dict['env'] = ENVIRONMENT
    save_dict['nn'] = nn
    save_dict['log'] = nn.log
    pickle.dump(save_dict, open('output_nets/cartpole.p', 'wb'))

#print(gym_env.action_space)
#print(gym_env.observation_space)
#print(gym_env.action_space.sample())
config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

cosyne = cs(config_dict)
cosyne.run(eval_cartpole)
print(cosyne.best_fitness)
demo_best_net(cosyne.best_nn)
gym_env.close()
save_best_network(cosyne.best_nn)