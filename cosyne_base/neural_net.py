import numpy as np
import torch
import torch.nn as nn



class CosyneNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()

        for layer_config in config["layers"]:
            self.layers.append(self.__add_layer(
                layer_config['type'], 
                layer_config['params'],
                layer_config['kwargs']))

        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def extract_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                weights.append(layer.weight)

        return weights

    def insert_weights(self, weights):
        pass

    """
    Begin Inner Facing Methods
    """
    
    def __add_layer(self, layer_type, layer_params, layer_kwargs):
        #Finds the pytorch relevant function and calls it with params and kwargs
        return getattr(nn, layer_type)(*layer_params, **layer_kwargs)