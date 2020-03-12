import gym
import numpy as np
from evo_ac.model import EvoACModel
import torch
from evo_ac.storage import EvoACStorage
from evo_ac.grad_evo import EvoACEvoAlg
from evo_ac.logger import EvoACLogger
import pickle


class EvoACRunner(object):
    def __init__(self, config):
        self.config = config
        self.config_evo = config['evo_ac']
        self.config_net = config['neural_net']
        self.config_exp = config['experiment']

        self.env = gym.make(self.config_exp['env'])
        
        self.logger = EvoACLogger(config)
        

    def reset_run(self, run_id):
        self.run_id = run_id
        num_pop = self.config_evo['pop_size']
        value_coeff = self.config_evo['value_coeff']
        entropy_coff = self.config_evo['entropy_coeff']

        self.model = EvoACModel(self.config)

        self.storage = EvoACStorage(num_pop)

        self.evo = EvoACEvoAlg(self.config)
        self.evo.set_params(self.model.extract_params())


    def run_experiment(self):
        for run_idx in range(self.config_exp['num_runs']):
            self.reset_run(run_idx)
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
                
                self.logger.save_fitnesses(self.storage.fitnesses, gen_idx)

                self.model.opt.zero_grad()
                loss = self.storage.get_loss()
                loss.backward()
                self.evo.set_grads(self.model.extract_grads())
                self.model.opt.step()
                self.evo.set_fitnesses(self.storage.fitnesses)

                new_pop = self.evo.create_new_pop()

                self.model.insert_params(new_pop)
            
            self.logger.end_run()
