import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

POP_SIZE = 10


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.population = nn.ModuleList()
        for _ in range(POP_SIZE):
            self.population.append(nn.Linear(128, 2))     

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.reset_storage()


    def reset_storage(self):
        self.saved_actions = [[] for _ in range(POP_SIZE)]
        self.rewards = [[] for _ in range(POP_SIZE)]
        self.fitnesses = [0] * POP_SIZE

    def extract_params(self):
        extracted_parameters = []
        for layer in self.population:
            layer_params = []
            for name, parameter in layer.named_parameters():
                layer_params.append(parameter.detach())
            extracted_parameters.append(layer_params)
        return extracted_parameters

    def insert_params(self, incoming_params):
        with torch.no_grad():
            for pop_idx in range(len(self.population)):
                params_idx = 0
                individual = self.population[pop_idx]
                state_dict = individual.state_dict()
                for name, parameter in individual.named_parameters():
                    state_dict[name] = incoming_params[pop_idx][params_idx]
                    params_idx += 1
                individual.load_state_dict(state_dict)

    def extract_grads(self):
        pass

    def forward(self, x, pop_idx):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.population[pop_idx](x))

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state, pop_idx):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state, pop_idx)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions[pop_idx].append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    #loss = torch.stack()
    loss = None
    for pop_idx in range(POP_SIZE):
        R = 0
        saved_actions = model.saved_actions[pop_idx]
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in model.rewards[pop_idx][::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        if loss == None:
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        else:
            loss += torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        
    
    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    model.reset_storage()


    
    extracted_parameters = model.extract_params()
    model.insert_params(extracted_parameters)
    """
    for individual in extracted_parameters:
        for param in individual:
            print(param.size())
    print()
    """
    


def main():
    running_reward = 10

    # run inifinitely many episodes
    for i_episode in count(1):

        for pop_idx in range(POP_SIZE):
            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't 
            # infinite loop while learning
            for t in range(1, 10000):

                # select action from policy
                action = select_action(state, pop_idx)

                # take the action
                state, reward, done, _ = env.step(action)

                if args.render:
                    env.render()

                model.rewards[pop_idx].append(reward)
                ep_reward += reward
                if done:
                    break
            model.fitnesses[pop_idx] = ep_reward
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        print(model.fitnesses)
        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()