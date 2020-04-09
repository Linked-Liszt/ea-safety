import gym
import numpy as np
from evo_ac.model import EvoACModel
import torch
from evo_ac.storage import EvoACStorage
from evo_ac.grad_evo import EvoACEvoAlg
from evo_ac.logger import EvoACLogger


class EvoACRunner(object):
    def __init__(self, config):
        self.config = config
        self.config_evo = config['evo_ac']
        self.config_net = config['neural_net']
        self.config_exp = config['experiment']
        
        
       
        self.env = gym.make(self.config_exp['env'])
        self.logger = EvoACLogger(config)

        
    def train(self):
        for run_idx in range(self.config_exp['num_runs']):
            self.reset_experiment()
            self.timesteps = 0
            for self.gen_idx in range(self.config_exp['num_gens']):
                self.storage.reset_storage()
                

                for pop_idx in range(self.config_evo['pop_size']):
                    obs = self.env.reset()

                    fitness = 0

                    while True:
                        runner_pop = pop_idx
                        if 'single' in self.config_exp and self.config_exp['single']:
                            runner_pop = 0

                        action, log_p_a, entropy, value = self.model.get_action(self.storage.obs2tensor(obs), runner_pop)

                        self.timesteps += 1

                        obs, reward, done, info = self.env.step(action.cpu().numpy())
                        fitness += reward

                        self.storage.insert(pop_idx, reward, action, log_p_a, value, entropy)

                    
                        if done:
                            break
                    
                    self.storage.insert_fitness(pop_idx, fitness)
                
                if 'single' in self.config_exp and self.config_exp['single']:
                    self.update_single()
                else:
                    self.update_evo_ac()

            self.logger.end_run()
        self.logger.end_experiment()

                
    def update_single(self):
        self.model.opt.zero_grad()
        loss, policy_loss_log, value_loss_log = self.storage.get_loss()
        loss.backward()
        self.model.opt.step()
        self.logger.save_fitnesses(self.model, self.storage.fitnesses, policy_loss_log, 
                                            value_loss_log, self.gen_idx, self.timesteps)
        self.logger.print_data(self.gen_idx)

    def update_evo_ac(self):
        self.model.opt.zero_grad()
        loss, policy_loss_log, value_loss_log = self.storage.get_loss()
        loss.backward()
        self.evo.set_grads(self.model.extract_grads())
        
        self.model.opt.step()

        self.evo.set_fitnesses(self.storage.fitnesses)

        new_pop = self.evo.create_new_pop()

        self.logger.save_fitnesses(self.model, self.storage.fitnesses, policy_loss_log, 
                                    value_loss_log, self.gen_idx, self.timesteps)
        self.logger.print_data(self.gen_idx)

        self.model.insert_params(new_pop)


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