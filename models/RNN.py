#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    """
    Class to create LSTM model.
    """
    def __init__(self, seq_len, hidden_size, num_layers, num_LSTM, num_linear, num_neurons, device):
        """
        Initialize the model.

        Arguments:
        - `seq_length`  : number of time steps in an input sequence
        - `hidden_size` : number of features in the hidden state h (must be a list)
        - `num_layers`  : number of recurrent layers in an LSTM layer (must be a list)
        - `num_LSTM`    : number of different LSTM layers
        - `num_linear`  : number of linear layers
        - `num_neurons` : number of neurons for linear layers (must be a list)
        - `device`      : device on which to train

        Returns:
        - None
        """
        super(LSTM, self).__init__()
        self.num_LSTM = num_LSTM
        self.num_linear = num_linear
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstms = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.device = device
        
        for k in range(num_LSTM):
            if k == 0:
                self.lstms.append(nn.LSTM(seq_len, hidden_size[0], num_layers[0], batch_first=True))
                continue
                
            self.lstms.append(nn.LSTM(hidden_size[k-1], hidden_size[k], num_layers[k], batch_first=True))
            
        for n in range(num_linear):
            if n == 0:
                self.linears.append(nn.Linear(hidden_size[-1], num_neurons[0]))
                continue
            
            self.linears.append(nn.Linear(num_neurons[n-1], num_neurons[n]))
        
        self.linears.append(nn.Linear(num_neurons[-1],1))
            
        for k in range(len(self.lstms)):
            if k == 1:
                continue
            
            nn.init.kaiming_normal_(self.lstms[k]._parameters['weight_ih_l0'])
            nn.init.kaiming_normal_(self.lstms[k]._parameters['weight_hh_l0'])
            if self.lstms[k].bias is not None:
                nn.init.constant_(self.lstms[k]._parameters['bias_ih_l0'], 0)
                nn.init.constant_(self.lstms[k]._parameters['bias_hh_l0'], 0)
        
        for k in range(num_linear):
            nn.init.kaiming_normal_(self.linears[k].weight)
            if self.linears[k].bias is not None:
                nn.init.constant_(self.linears[k].bias, 0)

            
    def forward(self, x):
        """
        Forward input sequence in the model.

        Arguments:
        - `x`  : input sequence

        Returns:
        - `logits` : model prediction
        """
        for k, lstm_k in enumerate(self.lstms):
            if k == 0:
                h = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to(self.device)
                c = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to(self.device)
                out, (h,c) = lstm_k(x, (h,c))
                continue

            out, (h,c) = lstm_k(out, (h,c)) #outputs, (hidden, cell) = lstm(input_sequence, (hidden, cell))

        out = out[0][:, -1, :] #只要seq这一个轴的最后一个
        # print(out[0].shape) #[BATCH_SIZE,in_features,hidden_features]
        # print(len(out[1])) 
        # print(out[1][0].shape) # [1,bs,h_f]
        # print(out[1][0].shape) # [1,bs,h_f]
        # print(h.shape) # [layers,bs,h_f]
        # print(c.shape) # [layers,bs,h_f]
        
        for k, linear_k in enumerate(self.linears):
            if k == self.num_linear:
                out = linear_k(out)
                return out
            
            out = F.relu(linear_k(out))

class GRU(torch.nn.Module):
    """
    Class to create GRU model.
    """
    def __init__(self, seq_length, hidden_size, num_layers, device): # SEQ_LEN, len(features)-2, 3
        """
        Initialize the model.

        Arguments:
        - `seq_length`  : number of time steps in an input sequence
        - `hidden_size` : number of features in the hidden state h (must be a list)
        - `num_layers`  : number of recurrent layers in a GRU layer (must be a list)
        - `device`      : device on which to train

        Returns:
        - None
        """
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.gru = torch.nn.GRU(seq_length, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0)
        self.fc = torch.nn.Linear(hidden_size,1)
    
    def forward(self, x):
        """
        Forward input sequence in the model.

        Arguments:
        - `x`  : input sequence (BATCH_SIZE, features, sequences)

        Returns:
        - `logits` : model prediction
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, h1 = self.gru(x,h0)
        #print(out.shape) #torch.Size([bs, input_features, hidden_size])
        #print(h1.shape)  #torch.Size([layers, bs, input_features])
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# from ncps.torch import CfC

# # in_features = 20 
# # out_features = 1
# # units = 16 # out_features or wiring
# # time = 10
# # batch_size = 32

# rnn = CfC(in_features,out_features)

# x = torch.randn(batch_size, time, in_features) # (batch_size, time, in_features)
# h0 = torch.zeros(batch_size,out_features) # (batch_size, units)
# output, hn = rnn(x,h0)

# print(output.shape) # torch.Size([32, 10, 50]) [batch_size, time, out_features]
# print(hn.shape)     # torch.Size([32, 50]) [batch_size, out_features]