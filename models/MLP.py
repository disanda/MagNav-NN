#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    '''
    Initialization of the model weights and biases.
    
    Arguments:
    - `m` : model layer
    
    Returns:
    - None
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class MLP(torch.nn.Module):
    """
    Class to create MLP model.
    """
    def __init__(self, seq_len, channels, hidden_features=16):
        """
        Initialize the model.

        Arguments:
        - `seq_length`  : number of time steps in an input sequence
        - `channels`    : number of input features

        Returns:
        - None
        """
        super(MLP, self).__init__()
        self.architecture = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(seq_len*channels,hidden_features), # # 20 * 15 = 300
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features,hidden_features//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features//2,1),
        )
        self.architecture.apply(init_weights)
        
        
    def forward(self, x):
        """
        Forward input sequence in the model.

        Arguments:
        - `x`  : input sequence

        Returns:
        - `logits` : model prediction
        """
        logits = self.architecture(x)
        return logits

class MLP_3L(torch.nn.Module):
    """
    Class to create MLP model.
    """
    def __init__(self, seq_len, channels, hidden_features=16):
        """
        Initialize the model.

        Arguments:
        - `seq_length`  : number of time steps in an input sequence
        - `channels`    : number of input features

        Returns:
        - None
        """
        super(MLP_3L, self).__init__()
        self.architecture = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(seq_len*channels,hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features,hidden_features//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features//2,hidden_features//4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features//4,1),
        )
        self.architecture.apply(init_weights)
        
        
    def forward(self, x):
        """
        Forward input sequence in the model.

        Arguments:
        - `x`  : input sequence

        Returns:
        - `logits` : model prediction
        """
        logits = self.architecture(x)
        return logits
    
    
class MLP_3Lv2(torch.nn.Module):
    """
    Class to create MLP model.
    """
    def __init__(self, seq_len, channels, hidden_features=16):
        """
        Initialize the model.

        Arguments:
        - `seq_length`  : number of time steps in an input sequence
        - `channels`    : number of input features

        Returns:
        - None
        """
        super(MLP_3Lv2, self).__init__()
        self.architecture = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(seq_len*channels,hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features,hidden_features//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features//2,hidden_features//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features//2,1),
        )
        self.architecture.apply(init_weights)
        
        
    def forward(self, x):
        """
        Forward input sequence in the model.

        Arguments:
        - `x`  : input sequence

        Returns:
        - `logits` : model prediction
        """
        logits = self.architecture(x)
        return logits