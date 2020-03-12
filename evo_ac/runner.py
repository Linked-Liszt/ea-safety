import gym
import numpy as np
from evo_ac.model import EvoACModel
import torch
from evo_ac.storage import EvoACStorage


class EvoACRunner(object):
    def __init__(self, config):
        self.config_evo = config['evo_ac']
        self.config_net = config['neural_net']
        
        self.model = EvoACModel(config)
       
        self.env = gym.make(self.config_evo['env'])


        obs_size = np.prod(np.shape(self.env.observation_space))
        num_pop = self.config_evo['pop_size']
        max_ep_steps = self.env._max_episode_steps
        value_coeff = self.config_evo['value_coeff']
        entropy_coff = self.config_evo['entropy_coeff']

        self.storage = EvoACStorage(num_pop)

    def train(self):
        for gen_idx in range(self.config_evo['num_gens']):
            self.storage.reset_storage()
            #TODO entropies

            for pop_idx in range(self.config_evo['pop_size']):
                obs = self.env.reset()

                fitness = 0

                while True:
                    action, log_p_a, entropy, value = self.model.get_action(self.storage.obs2tensor(obs), pop_idx)

                    obs, reward, done, info = self.env.step(action.cpu().numpy())
                    fitness += reward

                    self.storage.insert(pop_idx, reward, action, log_p_a, value)
                
                    if done:
                        break
                
                self.storage.insert_fitness(pop_idx, fitness)
            self.model.opt.zero_grad()
            loss = self.storage.get_loss()
            loss.backward()
            self.model.opt.step()

            # Evo Alg Here

            print(self.storage.fitnesses)

            