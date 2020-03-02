import numpy as np
import torch
import torch.nn as nn



class CosyneNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        self.__add_layers()
        

    def forward(self, x):
        lstm_count = 0
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):
                x, states = layer(x, (self.hidden_states[lstm_count], self.cell_states[lstm_count]))
                self.hidden_states[lstm_count] = states[0]
                self.cell_states[lstm_count] = states[1]
            else:
                x = layer(x)
        return x

    def init_recurrent_states(self):
        self.hidden_states = []
        self.cell_states = []
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):
                num_layers = layer.num_layers
                batch = 1
                hidden_size = layer.hidden_size
                self.hidden_states.append(torch.zeros(num_layers, batch, hidden_size)) 
                self.cell_states.append(torch.zeros(num_layers, batch, hidden_size)) 

    def extract_parameters(self):
        parameters = []
        for layer in self.layers:
            for name, parameter in layer.named_parameters():
                parameters.append(parameter)

        return parameters

    def extract_layer_sizes(self):
        input_sizes = []
        for layer in self.layers:
            for name, parameter in layer.named_parameters():
                if isinstance(layer, nn.LSTM):
                    input_sizes.append(layer.hidden_size)
                else:
                    input_sizes.append(layer.in_features)
        return input_sizes

    def insert_parameters(self, params):
        params_idx = 0
        for layer in self.layers:
            state_dict = layer.state_dict()
            for name, parameter in layer.named_parameters():
                state_dict[name] = torch.from_numpy(params[params_idx]).double()
                params_idx += 1
            layer.load_state_dict(state_dict)

    """
    Begin Inner Facing Methods
    """
    def __add_layers(self):
        for layer_config in self.config["layers"]:
            self.layers.append(self.__add_layer(
                layer_config['type'], 
                layer_config['params'],
                layer_config['kwargs']))
    
    def __add_layer(self, layer_type, layer_params, layer_kwargs):
        #Finds the pytorch relevant function and calls it with params and kwargs
        return getattr(nn, layer_type)(*layer_params, **layer_kwargs)