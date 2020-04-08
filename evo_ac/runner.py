import gym
import numpy as np
from evo_ac.model import EvoACModel
import torch
from evo_ac.storage import EvoACStorage
from evo_ac.grad_evo import EvoACEvoAlg


class EvoACRunner(object):
    def __init__(self, config):
        self.config = config
        self.config_evo = config['evo_ac']
        self.config_net = config['neural_net']
        self.config_exp = config['experiment']
        
        
       
        self.env = gym.make(self.config_evo['env'])

        
    def train(self):
        for run_idx in range(self.config_exp['num_runs']):
            self.reset_experiment()
            for gen_idx in range(self.config_exp['num_gens']):
                self.storage.reset_storage()
                #TODO entropies

                for pop_idx in range(self.config_evo['pop_size']):
                    obs = self.env.reset()

                    fitness = 0

                    while True:
                        action, log_p_a, entropy, value = self.model.get_action(self.storage.obs2tensor(obs), pop_idx)

                        obs, reward, done, info = self.env.step(action.cpu().numpy())
                        fitness += reward

                        self.storage.insert(pop_idx, reward, action, log_p_a, value, entropy)
                    
                        if done:
                            break
                    
                    self.storage.insert_fitness(pop_idx, fitness)
                self.model.opt.zero_grad()
                loss, policy_loss_log, value_loss_log = self.storage.get_loss()
                loss.backward()
                self.evo.set_grads(self.model.extract_grads())
                self.model.opt.step()
                self.evo.set_fitnesses(self.storage.fitnesses)

                new_pop = self.evo.create_new_pop()

                self.model.insert_params(new_pop)

                print(self.storage.fitnesses)

    def reset_experiment(self):
        obs_size = np.prod(np.shape(self.env.observation_space))
        num_pop = self.config_evo['pop_size']
        max_ep_steps = self.env._max_episode_steps
        value_coeff = self.config_evo['value_coeff']
        entropy_coff = self.config_evo['entropy_coeff']

        print("NEW")

        self.storage = EvoACStorage(num_pop, self.config)
        self.model = EvoACModel(self.config)
        self.evo = EvoACEvoAlg(self.config)
        self.evo.set_params(self.model.extract_params())