import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, config_dict):
        super(Policy, self).__init__()
        self.config_dict = config_dict
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

    def _init_layers(self):
        pass 


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