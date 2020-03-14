import argparse
import gym
import numpy as np
import copy
from itertools import count
from collections import namedtuple
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=False, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

log_directory = 'checkpoints'
log_name = 'lunarlander'
env_name = 'LunarLander-v2'
env = gym.make(env_name)

if args.seed != False:
    env.seed(args.seed)
    torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

POP_SIZE = 10
NUM_LAYERS = 2
HIDDEN_SIZE = 128


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, HIDDEN_SIZE)

        # actor's layer
        self.population = []
        for _ in range(POP_SIZE):
            individual = nn.ModuleList()
            for _ in range(NUM_LAYERS - 1):
                individual.append(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
            individual.append(nn.Linear(HIDDEN_SIZE, 4))
            self.population.append(individual) 

        # critic's layer
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

        # action & reward buffer
        self.reset_storage()

    def reset_storage(self):
        self.saved_actions = [[] for _ in range(POP_SIZE)]
        self.rewards = [[] for _ in range(POP_SIZE)]
        self.fitnesses = [0] * POP_SIZE

    

    def extract_params(self):
        extracted_parameters = []
        for individual in self.population:
            layer_params = []
            for layer in individual:
                for name, parameter in layer.named_parameters():
                    layer_params.append(parameter.detach())
            extracted_parameters.append(layer_params)
        return extracted_parameters

    def insert_params(self, incoming_params):
        with torch.no_grad():
            for pop_idx in range(len(self.population)):
                params_idx = 0
                individual = self.population[pop_idx]
                for layer in individual:
                    state_dict = layer.state_dict()
                    for name, parameter in layer.named_parameters():
                        state_dict[name] = incoming_params[pop_idx][params_idx]
                        params_idx += 1
                    layer.load_state_dict(state_dict)

    def extract_grads(self):
        extracted_grads = []
        for individual in self.population:
            layer_grads = []
            for layer in individual:
                for name, parameter in layer.named_parameters():
                    layer_grads.append(parameter.grad.detach())
            extracted_grads.append(layer_grads)
        return extracted_grads

    def forward(self, x, pop_idx):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        a = x

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        for i in range(len(self.population[pop_idx]) - 1):
            a = F.relu(self.population[pop_idx][i](a))

        action_prob = F.softmax(self.population[pop_idx][-1](a))

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


class CheckpointSaver(object):
    def __init__(self, directory, name, env):
        self.directory = directory
        self.name = name
        self.checkpoint_counter = 0
        self.env = env
        self.log = []


    def save_fitnesses(self, fitnesses, gen):
        data_dict = {}
        data_dict['gen'] = gen
        data_dict['fit_best'] = np.max(fitnesses)
        data_dict['fit_mean'] = np.mean(fitnesses)
        data_dict['fit_med'] = np.median(fitnesses)
        data_dict['fit_std'] = np.std(fitnesses)
        self.log.append(data_dict)


    def export_data(self, module):
       
        data_path = self.directory + '/' + self.name + '-' + str(self.checkpoint_counter) + '.p'
        save_dict = {}
        save_dict['env'] = self.env
        save_dict['nn'] = None
        save_dict['log'] = self.log
        pickle.dump(save_dict, open(data_path, 'wb'))
        self.checkpoint_counter += 1

checksaver = CheckpointSaver(log_directory, log_name, env_name)

class EvoAlg(object):
    def __init__(self, pararms, grads, fitnesses):
        self.params = pararms
        self.grads = grads
        self.fitnesses = np.array(fitnesses)

        #CONSTANTS
        self.num_mutate = [4,3,2,1]
        self.learning_rate = 1e-3
        self.scale_weight = 0.5

    
    def select_parents(self):
        argsorted = np.argsort(-self.fitnesses)
        self.parent_params = []
        self.parent_grads = []
        for pop_place in range(len(self.num_mutate)):
            pop_idx = argsorted[pop_place]
            self.parent_params.append(copy.deepcopy(self.params[pop_idx]))
            self.parent_grads.append(copy.deepcopy(self.grads[pop_idx]))
    
    
    def create_new_pop(self):
        self.select_parents()
        next_gen = []
        for parent_idx in range(len(self.num_mutate)):
            parent_count = self.num_mutate[parent_idx]
            for child_count in range(parent_count):
                child = []
                params = self.parent_params[parent_idx]
                grads = self.parent_grads[parent_idx]
                for i in range(len(params)):
                    child.append(self.mutate(params[i], grads[i]))
                next_gen.append(child)
        return next_gen
    
    def mutate(self, param, grad):
        adjusted_grad = self.learning_rate * grad

        locs = param - adjusted_grad
        scales = torch.abs(adjusted_grad) * self.scale_weight

        norm_dist = torch.distributions.normal.Normal(locs, scales)
        return norm_dist.sample()


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

        print(torch.stack(policy_losses).sum().item())
        
    
    # reset gradients
    optimizer.zero_grad()

    
    extracted_parameters = model.extract_params()
    
    # perform backprop
    loss.backward()

    extracted_grads = model.extract_grads()

    optimizer.step()

    pop_gen = EvoAlg(extracted_parameters, extracted_grads, model.fitnesses)
    new_pop = pop_gen.create_new_pop()
    
    model.insert_params(new_pop)



    """
    LR = 0.0001

    for pop_idx in range(len(extracted_parameters)):
        for param_idx in range(len(extracted_parameters[pop_idx])):
            extracted_parameters[pop_idx][param_idx] += (LR * extracted_grads[pop_idx][param_idx])
    #print()
    
    model.insert_params(extracted_parameters)
    """

    # reset rewards and action buffer
    model.reset_storage()


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
        checksaver.save_fitnesses(model.fitnesses, i_episode)
        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            checksaver.export_data(model)
        
        

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()