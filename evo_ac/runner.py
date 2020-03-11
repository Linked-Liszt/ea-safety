import gym
import numpy as np
from evo_ac.model import EvoACModel
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
        is_cuda = False #TODO auto-detect
        value_coeff = self.config_evo['value_coeff']
        entropy_coff = self.config_evo['entropy_coeff']

        self.storage = EvoACStorage(max_ep_steps, num_pop, obs_size, is_cuda,
                                        entropy_coff)

    def train(self):
        for gen_idx in range(self.config_evo['num_gens']):
            self.storage.reset_buffers()
            for pop_idx in range(self.config_evo['pop_size']):
                obs = self.env.reset()
                
                self.storage.states[0][pop_idx].copy_(self.storage.obs2tensor(obs))
                episode_entropy = 0
                step = 0

                while True:
                    action, log_p_a, entropy, value = self.model.get_action(obs, pop_idx)
                    episode_entropy += entropy

                    obs, rewards, done, info = self.env.step(action.cpu().numpy())

                    #TODO: Understand this call
                    self.storage.log_episode_rewards(info)
                    self.storage.insert(step, pop_idx, rewards, obs, action, log_p_a, value, done)
                    step += 1
                
            # Calculate loss here
            