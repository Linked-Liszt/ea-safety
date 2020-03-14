import numpy as np
import torch
import torch.nn as nn
# TODO: Consider refactor to functional versions
import torch.nn.functional as F


# TODO: Handle variable episode sizes
class EvoACStorage(object):
    def __init__(self, pop_size, value_coeff,
                 entropy_coeff, reward_discount=0.99):
        """

        :param max_episode_steps: number of steps after the policy gets updated
        :param num_pop: number of environments to train on parallel
        :param obs_shape: shape of a frame as a tuple
        :param n_stack: number of frames concatenated
        :param is_cuda: flag whether to use CUDA
        """
        super().__init__()

        self.pop_size = pop_size

        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.reward_discount = reward_discount

        # initialize the buffers with zeros
        self.reset_storage()

    def reset_storage(self):
        self.actions = [[] for _ in range(self.pop_size)]
        self.log_probs = [[] for _ in range(self.pop_size)]
        self.rewards = [[] for _ in range(self.pop_size)]
        self.values = [[] for _ in range(self.pop_size)]
        self.fitnesses = [0] * self.pop_size
        

    def obs2tensor(self, obs):
        # 1. reorder dimensions for nn.Conv2d (batch, ch_in, width, height)
        # 2. convert numpy array to _normalized_ FloatTensor
        tensor = torch.from_numpy(obs.astype(np.float32))
        return tensor

    
    def insert(self, pop_idx, reward, action, log_prob, value):
        self.rewards[pop_idx].append(reward)
        self.actions[pop_idx].append(action)
        self.log_probs[pop_idx].append(log_prob)
        self.values[pop_idx].append(value)

    def insert_fitness(self, pop_idx, fitness):
        self.fitnesses[pop_idx] = fitness
    
    def _discount_rewards(self):
        self.discounted_rewards = [[] for _ in range(self.pop_size)]
        for pop_idx in range(self.pop_size):
            reward = 0
            for r in self.rewards[pop_idx][::-1]:
                reward = r + self.reward_discount * reward
                self.discounted_rewards[pop_idx].insert(0, reward)
    
    def get_loss(self):
        # TODO: Double check sequencing
        self._discount_rewards()
        value_losses = []
        policy_losses = []
        for pop_idx in range(self.pop_size):
            for step_idx in range(len(self.rewards[pop_idx])):
                value = self.values[pop_idx][step_idx]
                reward = self.discounted_rewards[pop_idx][step_idx]

                advantage = reward - value.item()

                value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward])))

                policy_losses.append((-self.log_probs[pop_idx][step_idx] * advantage))
    
        policy_loss_total = torch.stack(policy_losses).sum()
        value_loss_total = torch.stack(value_losses).sum()
        loss = (policy_loss_total * self.value_coeff) + value_loss_total #- self.entropy_coeff * entropy
        return loss, policy_loss_total.item(), value_loss_total.item()
