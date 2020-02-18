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
        for layer in self.layers:
            x = layer(x)
        return x

    def extract_parameters(self):
        parameters = []
        for layer in self.layers:
            for name, parameter in layer.named_parameters():
                parameters.append(parameter)

        return parameters

    def insert_parameters(self, weights):
        pass

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